"""LangGraph 状态定义 —— 图中流动的核心数据结构。

本文件的角色：
1. 定义 AgentState（TypedDict），它是各 Agent 节点间传递数据的"共享内存"
2. 提供辅助工厂函数，在图入口处创建初始状态

设计决策（为什么用 TypedDict 而非 Pydantic BaseModel）：
- LangGraph 官方推荐 TypedDict 作为状态容器——它在序列化到 Checkpointer（M8 的 PostgreSQL）
  时直接用 JSON，不需要额外的模型转换逻辑
- total=False 允许多起点执行：例如 RAG 命中 chord_only 时从 Agent 2 起步，
  midi_notes 字段自然为空，不强制填充
- 字段值用 Python 原生类型（dict / list / str / int）而非 Pydantic model：
  各 Agent node 内部导入 schemas.py 的 Pydantic model 做校验 → .model_dump() 写 State
  → 下游 node 用 .model_validate(state["key"]) 恢复 —— 序列化边界清晰，零损耗

消息累加机制：
- messages 字段使用 LangGraph 内置的 add_messages reducer
- 每次 LLM 调用产生的新消息自动追加到列表，不会被后续节点覆盖
- 这是实现 Agent 3 / Agent 5"与 LLM 对话"能力的关键基础设施
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class AgentState(TypedDict, total=False):
    """多 Agent 指弹谱生成系统的全局状态。

    每个 Agent 节点阅读整个状态，修改属于它职责范围的 key 后写回。
    不负责的 key 保持原样，LangGraph 引擎自动合并。

    注意：状态中所有复杂对象均以 dict 形式存储（Pydantic model 经 .model_dump() 序列化）。
    """

    # =========================================================================
    # 输入层 —— 图入口（API 路由 / RAG 路由）写入，后续节点只读
    # =========================================================================

    midi_path: str
    """上传的 MIDI 文件在服务器上的临时路径。Agent 1 从中读取音符序列。"""

    song_name: str
    """用户搜索的歌名文本。RAG 检索未命中 full_tab 时，作为备选路由的输入。"""

    difficulty: str
    """演奏难度 —— 'beginner' | 'intermediate' | 'advanced'。
    由前端配置面板传入，影响 Agent 3 的指法生成策略（把位范围、和弦复杂度）。"""

    style: str
    """指弹风格 —— 'jpop' | 'american_folk' | 'pop_adaptation'。
    影响 Agent 3 的模板选择 + RAG 风格样例检索。"""

    tuning: list[str]
    """吉他定弦，6 个元素的音高名称列表。
    默认标准定弦 ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']，用户可在配置面板自定义。"""

    capo: int
    """变调夹品位。0=不用，>0=建议品位（由 Agent 4 校验后给出推荐）。"""

    # =========================================================================
    # 路由层 —— API 的硬编码路由逻辑写入（非 Agent 节点）
    # =========================================================================

    rag_hit_type: str
    """RAG 检索命中类型 —— '' | 'full_tab' | 'chord_only'。
    空字符串表示走 MIDI 完整链路；'full_tab' 直接返回（跳过 Agent 链路）；
    'chord_only' 跳至 Agent 2 起（不走 Agent 1 的 MIDI 解析）。"""

    # =========================================================================
    # Agent 1（旋律解析）产出 —— midi_parser.py 写入
    # =========================================================================

    bpm: int
    """整曲速度（每分钟节拍数）。Agent 1 从 MIDI 的 tempo 标记中提取，
    传入 Agent 2 供和弦分桶窗口计算使用。"""

    midi_notes: list[dict]
    """全轨音符序列（含伴奏）。每个元素为 MidiNote.model_dump() 的 dict。
    这是 Agent 2（和声编排）的输入——和声分析需要完整的和声信息。"""

    melody_notes: list[dict]
    """仅主旋律轨音符（ADR-001 P1）。midi_parser 启发式识别后单独分离，
    每个元素为 MidiNote.model_dump() 的 dict（is_melody=True）。
    Agent 3（指法生成）直接使用此列表作为旋律线，不再内部猜测。
    若 midi_parser 无法识别主旋律轨，此列表为空——下游使用旧 top_note 策略回退。"""

    # =========================================================================
    # Agent 2.5（LLM 编曲决策）产出 —— nodes.py 写入（ADR-001 P2）
    # =========================================================================

    arrangement_plan: dict
    """LLM 编曲决策结果，即 ArrangementPlan.model_dump() 的 dict。
    包含按段落划分的 density / bass_style / melody_register / techniques / dynamic。
    若 LLM 调用失败，此字段不存在——下游 generate_tab() 使用默认全曲统一参数。"""

    # =========================================================================
    # Agent 2（和声编排）产出 —— music21_wrapper.py 写入
    # =========================================================================

    harmony: dict
    """和声分析结果，即 HarmonyAnalysis.model_dump() 的 dict。
    包含 key（调性）、bpm（速度）、chord_progression（和弦进行序列）、
    time_signature（拍号）。作为 Agent 3（指法生成）的输入之一。"""

    # =========================================================================
    # Agent 3（指法生成）产出 —— tab_generator.py 写入
    # =========================================================================

    tab_data: dict
    """完整指弹谱数据，即 TabData.model_dump() 的 dict。
    包含 measures（小节列表）、tuning、capo、tempo、key、style 等。
    这是整个系统的核心产出，可直接送给 alphaTab 渲染或导出 .gp5。"""

    # =========================================================================
    # Agent 4（物理校验）产出 —— tab_validator.py 写入
    # =========================================================================

    validation: dict
    """物理校验结果，即 ValidationResult.model_dump() 的 dict。
    包含 is_valid（是否通过）、errors（具体问题列表）、warnings（警告）、
    capo_recommendation（变调夹推荐）。若校验不通过，Agent 3 读取 errors 做局部修正。"""

    validation_retry_count: int
    """校验→回退的重试计数器。初始值为 0，每次不通过后 +1。
    设置上限（如 3 次）防止死循环：达到上限后标记 error 并终止。"""

    # =========================================================================
    # Agent 5（修改理解器）产出 —— 用户 QA 指令驱动
    # =========================================================================

    modify_instruction: str
    """用户输入的自然语言修改指令，如 '副歌简化一点'、'把前奏升八度'。
    由 POST /modify 接口写入，Agent 5 读取后解析为结构化修改目标。"""

    modification_plan: dict
    """Agent 5（LLM）解析 modify_instruction 后产出的结构化修改计划。
    即 ModificationPlan.model_dump() 的 dict，含 summary + operations 列表。
    Agent 3（caller=agent_5）直接读取此字段执行操作，不再重复调 LLM。"""

    modified_tab_data: dict
    """修改后的指弹谱数据（同 TabData 结构）。
    Agent 5 定位修改范围后，调用 Agent 3 的指法生成逻辑对局部重新计算，
    其他未修改的小节保持原样。"""

    # =========================================================================
    # 流程控制 —— 所有节点共用
    # =========================================================================

    caller: str
    """调用链标记 —— Agent 3 据此判断走哪条分支。
    'agent_2'=首次生成(确定性)
    'agent_4'=校验回退(LLM 读 validation.errors → operations → 重生成)
    'agent_5'=修改路径(确定性读取 Agent 5 已解析的 modification_plan → 执行重生成)
    由上一个节点在返回 dict 时写入，Agent 3 读取后分流。"""

    messages: Annotated[list[AnyMessage], add_messages]
    """LLM 对话历史。使用 add_messages reducer，每次 invoke 自动追加而非覆盖。
    Agent 3 和 Agent 5 的 LLM 决策通过此字段与 DeepSeek 交互。
    空列表时 LangGraph 自动初始化为 []。"""

    error: str
    """全局错误信息。任何节点在不可恢复的异常发生时写入，图据此终止执行。
    空字符串 = 无错误，正常运行。"""

    status: str
    """当前流程状态标记，用于 LangSmith Trace 中追踪执行到哪个节点。
    典型值：'init' → 'parsing_midi' → 'analyzing_harmony' →
    'generating_tab' → 'validating' → 'completed' → 'modifying'"""


# =========================================================================
# 辅助函数
# =========================================================================


def create_initial_state(
    *,
    midi_path: str = "",
    song_name: str = "",
    difficulty: str = "beginner",
    style: str = "jpop",
    tuning: list[str] | None = None,
    capo: int = 0,
) -> AgentState:
    """创建图的初始状态（工厂函数）。

    所有可选字段填入安全的默认值，调用方只传自己关心的参数，
    避免每次 app.invoke() 时重复手写长长一串默认值。
    """
    if tuning is None:
        tuning = ["E2", "A2", "D3", "G3", "B3", "E4"]

    return {
        "midi_path": midi_path,
        "song_name": song_name,
        "difficulty": difficulty,
        "style": style,
        "tuning": tuning,
        "capo": capo,
        "rag_hit_type": "",
        "validation_retry_count": 0,
        "status": "init",
        "error": "",
    }

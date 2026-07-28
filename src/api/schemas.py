"""Pydantic 数据模型 —— 多 Agent 系统的类型契约。

本文件的角色：
1. 定义各 Agent 节点的输入/输出 Schema（类型边界，node 间传数据时靠模型校验防出错）
2. 为 FastAPI 的 Swagger UI (/docs) 提供自动生成的请求/响应文档
3. 不与 LangGraph State 直接耦合——State 中用 dict 承载序列化后的数据（见 state.py），
   各 node 内部用本文件的 Pydantic model 做校验 → .model_dump() 写入 State

分为三层：枚举 → 音乐数据层 → 聚合输出 + API 交互层。
"""

from typing import Literal

from enum import Enum
from pydantic import BaseModel, Field


# =============================================================================
# 第一层：枚举（全局复用，所有模块共享同一套定义）
# =============================================================================


class Difficulty(str, Enum):
    """演奏难度。对应前端配置面板的下拉选项，影响 Agent 3 的指法生成策略。"""

    BEGINNER = "beginner"       # 初级：优先低把位、开放和弦、简单节奏
    INTERMEDIATE = "intermediate"  # 中级：允许中把位、横按、中等跨度
    ADVANCED = "advanced"      # 高级：全把位、复杂技巧、大跨度


class Style(str, Enum):
    """指弹风格。影响 Agent 3 的模板选择 + RAG 检索的样式匹配。"""

    JPOP = "jpop"              # 日系指弹（押尾光太郎式打击+旋律线）
    AMERICAN_FOLK = "american_folk"  # 美式指弹（Travis picking、低音交替）
    POP_ADAPTATION = "pop_adaptation"  # 流行改编（简化编排、突出人声旋律代换）


class Technique(str, Enum):
    """吉他常用技巧标注。Agent 3 生成指法时按模板规则自动附加，最终显示在谱面上。"""

    HAMMER_ON = "H"            # 击弦
    PULL_OFF = "P"             # 勾弦
    BEND = "B"                 # 推弦
    SLIDE = "S"                # 滑弦
    ARTIFICIAL_HARMONIC = "A.H."  # 人工泛音
    NONE = "none"              # 无特殊技巧


# =============================================================================
# 第二层：音乐基本数据（对应用到的基石概念，与 music21 / MIDI 语义对齐）
# =============================================================================


class MidiNote(BaseModel):
    """单个音符 —— 从 MIDI 文件中解析出来的最小单元。

    Agent 1（旋律解析）产出 List[MidiNote]，传递给 Agent 2（和声编排）。
    """

    midi_number: int = Field(..., ge=0, le=127, description="MIDI 音符号（0-127，中央 C = 60）")
    start_time: float = Field(..., ge=0, description="音符起始时间（quarter lengths，music21 offset）")
    duration: float = Field(..., gt=0, description="音符持续时长（quarter lengths）")
    velocity: int = Field(default=64, ge=0, le=127, description="按键力度")
    track: int = Field(default=0, ge=0, description="MIDI 音轨编号")
    channel: int = Field(default=0, ge=0, le=15, description="MIDI 通道号")
    is_melody: bool = Field(default=False, description="ADR-001 P1：是否来自主旋律轨（midi_parser 启发式识别）")


class Chord(BaseModel):
    """单个和弦 —— Agent 2（和声编排）的核心分析单元。

    通过对一段旋律的音符集合做 music21 和弦识别，输出此结构。
    """

    name: str = Field(..., description="和弦标准名称，如 'Cmaj7'、'Dm'、'G7'")
    root: str = Field(..., description="根音音名，如 'C'、'F#'")
    quality: str = Field(..., description="和弦性质，如 'maj7'、'm'、'7'、'dim'")
    midi_numbers: list[int] = Field(..., description="构成和弦的所有 MIDI 音符号")
    start_time: float = Field(..., ge=0, description="和弦起始时间（quarter lengths）")
    duration: float = Field(..., gt=0, description="和弦覆盖时长（quarter lengths），即到下一和弦的间隔")


class TabNote(BaseModel):
    """六线谱上的单个音符 —— Agent 3（指法生成）的核心输出单元。

    每个 TabNote 代表一根弦上的一个确定位置（哪根弦、第几品），
    可直接驱动 alphaTab 渲染。
    """

    string: int = Field(..., ge=1, le=6, description="吉他弦编号（1=高音E，6=低音E）")
    fret: int = Field(..., ge=0, le=24, description="品位编号（0=空弦，最大 24 品）")
    start_time: float = Field(..., ge=0, description="音符起始时间（quarter lengths）")
    duration: float = Field(..., gt=0, description="音符持续时长（quarter lengths）")
    technique: Technique = Field(default=Technique.NONE, description="演奏技巧标注（H/P/B/S/A.H./none）")
    voice: str = Field(default="", description="声部标注：melody=旋律 / inner=内声部 / bass=低音。空字符串=未标记")


class TabMeasure(BaseModel):
    """六线谱的一个小节 —— TabData 的基本组成单位。

    一个小节包含若干 TabNote，配合拍号控制 alphaTab 的格子排版。
    """

    number: int = Field(..., ge=0, description="小节序号（从 0 或 1 开始）")
    notes: list[TabNote] = Field(default_factory=list, description="该小节内的所有音符")
    time_signature: tuple[int, int] = Field(
        default=(4, 4), description="拍号（分子=每小节拍数，分母=以几分音符为一拍）"
    )


class ValidationError(BaseModel):
    """物理校验失败的具体条目 —— Agent 4（物理校验）产出的错误列表单元。

    每条错误精确到"哪个小节、哪根弦、第几品、什么原因"，
    便于 Agent 3 在回退时缩小修正范围。
    """

    measure: int = Field(..., description="出错的小节序号")
    string: int = Field(default=0, description="出错的弦号（0=全局/多弦问题）")
    fret: int = Field(default=0, description="出错的品位")
    description: str = Field(..., description="人类可读的错误描述，如'品位 15 超出初级难度限制'")
    severity: str = Field(default="error", description="严重程度：'error'（阻断）或 'warning'（可放行）")


# =============================================================================
# 第三层：聚合输出模型（各 Agent 节点的完整产出 + API 请求/响应）
# =============================================================================


class HarmonyAnalysis(BaseModel):
    """Agent 2（和声编排）的完整输出。

    将整首曲子的和弦进行 + 调性 + 速度打包，传递给 Agent 3 用于指法决策。
    """

    key: str = Field(..., description="调性名称，如 'C major'、'A minor'，由 music21 分析得出")
    bpm: int = Field(..., ge=1, le=300, description="每分钟节拍数，优先取 MIDI 自身的 tempo")
    chord_progression: list[Chord] = Field(..., description="整曲的和弦进行序列（时间排序）")
    time_signature: tuple[int, int] = Field(
        default=(4, 4), description="拍号，优先取 MIDI 自身的拍号"
    )


class TabGenerationConfig(BaseModel):
    """用户配置聚合 —— 传给 Agent 3（指法生成）作为行为参数。

    该结构汇集了前端配置面板的三个选项（难度/风格/定弦），
    外加一个可空的变调夹推荐位（Agent 4 可能回填）。
    """

    difficulty: Difficulty = Field(default=Difficulty.BEGINNER, description="演奏难度（当前统一人手极限约束 fret≤15；QA 修改时可触发 per-measure 品位上限分级）")
    style: Style = Field(..., description="指弹风格")
    tuning: list[str] = Field(
        default_factory=lambda: ["E2", "A2", "D3", "G3", "B3", "E4"],
        description="吉他定弦（6根弦的标准音高名称），默认标准定弦 EADGBE",
    )
    capo: int = Field(default=0, ge=0, le=12, description="变调夹品位（0=不用，最大 12 品）")
    melody_source: Literal["top_note", "highest_density"] = Field(
        default="top_note",
        description=(
            "旋律提取回退策略（仅当 P1 MIDI 主旋律轨识别失败时使用）："
            "'top_note'=每拍取最高音（适合钢琴独奏MIDI）；"
            "'highest_density'=跟踪音符密度最高的音高线（更接近人声旋律线）"
        ),
    )


class TabData(BaseModel):
    """一份完整的指弹谱 —— Agent 3 的输出、Agent 5 的修改目标、alphaTab 的渲染输入。

    这是整个系统的"核心产出物"：包含所有小节信息、元数据（调性/速度/定弦/变调夹），
    可直接序列化为 alphaTab 支持的格式进行渲染，也可导出为 .gp5 文件。
    """

    measures: list[TabMeasure] = Field(..., description="小节列表（按时间顺序排列）")
    tuning: list[str] = Field(..., description="定弦（6 根弦的音高名称）")
    capo: int = Field(default=0, ge=0, le=12, description="变调夹品位")
    tempo: int = Field(default=120, ge=1, le=300, description="演奏速度（BPM）")
    key: str = Field(default="C major", description="调性")
    style: str = Field(default="jpop", description="指弹风格")
    techniques_used: list[Technique] = Field(
        default_factory=list, description="本谱使用的所有技巧类型（去重），用于前端标注图例"
    )


class ValidationResult(BaseModel):
    """Agent 4（物理校验）的完整输出。

    校验通过 → is_valid=True，TAB 可直接返回给前端。
    校验不通过 → is_valid=False，errors 列表送给 Agent 3 做局部修正（回退循环）。
    capo_recommendation 为智能变调夹推荐：如果发现大量音符集中在低把位但难以按弦，
    建议用户使用变调夹来优化指法舒适度。
    """

    is_valid: bool = Field(..., description="是否通过所有物理约束检查")
    errors: list[ValidationError] = Field(
        default_factory=list, description="校验不通过的具体错误条目"
    )
    warnings: list[str] = Field(
        default_factory=list, description="校验通过但存在隐患的提示（如高把位连用可能影响音色）"
    )
    capo_recommendation: int | None = Field(
        default=None, description="智能变调夹推荐品位（None=不推荐，0=移除变调夹，>0=建议品位）"
    )


# =============================================================================
# 修改操作模型（Agent 3 回退 + Agent 5 修改理解 共用）
# =============================================================================


class ModificationOperation(BaseModel):
    """原子修改操作 —— LLM 输出的结构化修正单元。

    5 种操作类型，覆盖所有用户自然语言指令的语义空间：
      - adjust_difficulty: 局部难度升降（"副歌简化"→beginner）
      - change_density:    声部密度调整（"太脏了"→sparse、"太单薄"→rich）
      - transpose:         音高位移（"低八度"→-12）
      - reassign_string:   弦分配偏好（"减少换弦"→adjacent_strings）
      - switch_technique:  技巧替换（"加滑音"→slide、"别用击弦"→no_hammer）
    """

    op: Literal[
        "adjust_difficulty", "change_density", "transpose",
        "reassign_string", "switch_technique",
    ] = Field(..., description="操作类型")
    scope: str = Field(
        default="entire_song",
        description="作用范围（自然语言描述，由 _resolve_scope 转为小节索引）："
                    "'entire_song'/'chorus'/'bridge'/'verse'/'measure_5-8' 等",
    )
    # --- op-specific parameters (only the relevant one is used) ---
    difficulty: str | None = Field(default=None, description="[adjust_difficulty] beginner/intermediate/advanced")
    density: str | None = Field(default=None, description="[change_density] sparse/normal/rich")
    semitones: int | None = Field(default=None, description="[transpose] 半音位移量，-12=低八度")
    constraint: str | None = Field(default=None, description="[reassign_string] low_position/adjacent_strings")
    technique: str | None = Field(default=None, description="[switch_technique] hammer/pull_off/slide/remove")


# =============================================================================
# ADR-001 P2：LLM 编曲决策模型
# =============================================================================

# 封闭选项集字面量（LLM 只能从这些值中选择，Pydantic 在 parse 阶段拒绝越界值）
Density = Literal["sparse", "medium", "full"]
BassStyle = Literal["root_only", "alternating", "travis_picking"]
MelodyRegister = Literal["low", "mid", "high"]
Dynamic = Literal["ppp", "pp", "p", "mp", "mf", "f", "ff", "fff"]
TechniqueChoice = Literal["hammer_on", "pull_off", "slide", "harmonic", "strumming"]


class SectionPlan(BaseModel):
    """单个段落的编排配置 —— LLM 为每个乐段选择参数组合。

    所有字段的合法值均为封闭集合，LLM 不能创造新选项。
    """

    measure_start: int = Field(..., ge=1, description="段落起始小节（1-indexed）")
    measure_end: int = Field(..., ge=1, description="段落结束小节（1-indexed，含）")
    label: str = Field(default="", description="段落标签，如 'intro' / 'verse' / 'chorus' / 'bridge'")
    density: Density = Field(default="medium", description="音符填充密度（sparse < medium < full）")
    bass_style: BassStyle = Field(default="alternating", description="低音行进模式")
    melody_register: MelodyRegister = Field(default="mid", description="旋律优先音域")
    techniques: list[TechniqueChoice] = Field(
        default_factory=list, description="该段落使用的技巧（可为空列表）"
    )
    dynamic: Dynamic = Field(default="mf", description="力度标记（影响内声部音符数和力度）")


class ArrangementPlan(BaseModel):
    """LLM 编曲决策的完整输出 —— 歌曲全部段落的编排计划。

    由 Agent 2.5 产生，传递给 Agent 3 的 generate_tab()。
    """

    sections: list[SectionPlan] = Field(..., min_length=1, description="按时间顺序排列的段落列表")
    summary: str = Field(..., description="编排计划的人读摘要，供前端展示")


class ModificationPlan(BaseModel):
    """LLM 输出的完整修改计划 —— 含一组有序的原子操作。

    由 Agent 3（回退修正）和 Agent 5（用户修改）共用此格式。
    summary 用于前端展示 + changes_summary 回写。
    """

    summary: str = Field(..., description="修改计划的人读摘要，如'副歌简化为初级指法，减少内声部'")
    operations: list[ModificationOperation] = Field(
        default_factory=list, description="有序原子操作列表（按此顺序依次执行）"
    )


# =============================================================================
# API 交互模型（里程碑 6 正式启用，此处先定义以保证 Schema 稳定性）
# =============================================================================


class UploadResponse(BaseModel):
    """POST /upload 的响应体。

    source 字段说明数据来源，便于前端区分处理逻辑：
    - "midi_pipeline" → Agent 链路产出，含完整 TabData
    - "rag_full_tab"   → RAG 直出完整指弹谱，跳过 Agent 链路
    - "direct_gp5"     → 用户上传 .gp5 文件，直接返回
    """

    tab_id: str = Field(..., description="生成的谱面唯一标识（UUID）")
    tab_data: TabData | None = Field(default=None, description="TAB 谱数据（midi_pipeline / rag_full_tab 时有值）")
    source: str = Field(..., description="数据来源路由标记")
    message: str = Field(default="", description="给前端的人性化提示信息")


class ModifyRequest(BaseModel):
    """POST /modify 的请求体。

    用户在 QA 输入框输入自然语言指令 → 前端以此结构发送给后端。
    Agent 5（修改理解器）解析 instruction 并定位修改范围。
    """

    tab_id: str = Field(..., description="当前正在编辑的谱面 ID")
    instruction: str = Field(..., min_length=1, description="自然语言修改指令，如'副歌简化一点'")


class ModifyResponse(BaseModel):
    """POST /modify 的响应体。返回修改后的谱面 + 修改摘要供前端提示。"""

    tab_id: str = Field(..., description="谱面 ID（同请求中的 tab_id）")
    modified_tab_data: TabData = Field(..., description="修改后的完整 TAB 谱数据")
    changes_summary: str = Field(..., description="修改内容的自然语言摘要，如'已将第8-16小节简化为初级指法'")

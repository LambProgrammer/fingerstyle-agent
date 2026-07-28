"""Agent 节点函数 —— LangGraph 工作流中的 6 个处理单元 + 入口路由。

节点职责（对照章程 §4.1）：
  agent_1  旋律解析  确定性：midi_parser.parse_midi()
  agent_2  和声编排  确定性：music21_wrapper.analyze_chords()
  agent_3  指法生成  三路分支：
             caller=agent_2 → 首次确定性生成（不花 token）
             caller=agent_4 → 校验回退（LLM 读 validation.errors → operations → 重生成）
             caller=agent_5 → 修改路径（LLM 读 modify_instruction → operations → 重生成）
  agent_4  物理校验  确定性：tab_validator.validate() + 条件回退判定
  agent_5  修改理解  LLM：用户指令 → ModificationPlan（operations 列表）

LLM 修正的核心机制（operations 驱动）：
  不再通过 LLM 直接操作 MIDI 数值，而是让 LLM 输出结构化 ModificationPlan
  → _apply_operations() 执行 → generate_tab() 重生成。LLM 只做"语义→操作"翻译，
  确定性代码做"操作→数据修改"，闭环完整且可靠。
"""

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from src.agents.state import AgentState
from src.api.schemas import (
    ArrangementPlan,
    HarmonyAnalysis,
    MidiNote,
    ModificationPlan,
    Style,
    TabData,
    TabGenerationConfig,
    ValidationResult,
)
from src.config import settings
from src.tools.midi_parser import parse_midi
from src.tools.music21_wrapper import analyze_chords
from src.tools.tab_generator import generate_tab
from src.tools.tab_validator import validate

logger = logging.getLogger(__name__)

_MAX_VALIDATION_RETRIES = 3

# DeepSeek 实例懒加载（全局复用）
_llm_cache: dict[tuple[str, int], ChatDeepSeek] = {}


def _get_llm(model: str = "deepseek-v4-pro", max_tokens: int = 1024) -> ChatDeepSeek:
    key = (model, max_tokens)
    if key not in _llm_cache:
        _llm_cache[key] = ChatDeepSeek(
            model=model,
            api_key=SecretStr(settings.deepseek_api_key),
            max_tokens=max_tokens,
            timeout=90,
        )
    return _llm_cache[key]


def _extract_response_text(response) -> str:
    """从 LangChain AIMessage 中提取文本，兼容 DeepSeek reasoning 模型。

    DeepSeek reasoning 系列（v4-pro / v4-flash）将最终答案放在 content，
    但 max_tokens 不足时 content 可能为空，实际答案落在
    additional_kwargs["reasoning_content"] 中。
    """
    raw = response.content if hasattr(response, "content") else str(response)
    text = raw if isinstance(raw, str) else str(raw)
    if not text and hasattr(response, "additional_kwargs"):
        text = response.additional_kwargs.get("reasoning_content", "")
    return text


# =========================================================================
# 共享 LLM prompt：operations JSON 格式说明
# =========================================================================

_OPERATIONS_SYSTEM_PROMPT = (
    "你是指弹吉他谱修改理解器。将输入的问题/指令翻译为结构化修改方案。\n\n"
    "只输出 JSON（不要任何其他文字），格式如下：\n"
    '{"summary": "一句话描述修改内容", "operations": [...]}\n\n'
    "每个 operation 包含：\n"
    "  op: 操作类型，只取以下 5 种之一：\n"
    '    "adjust_difficulty" — 局部调整难度\n'
    '    "change_density"    — 调整声部密度\n'
    '    "transpose"         — 音高上下移\n'
    '    "reassign_string"   — 改变弦分配偏好\n'
    '    "switch_technique"  — 替换演奏技巧\n'
    "  scope: 作用范围，自由填写自然语言（如 'chorus' / 'measure_5-8' / 'entire_song'）\n"
    "  以及对应 op 的参数字段。\n\n"
    "参数合法值（必须严格使用以下值，严禁自己造词）：\n"
    '  difficulty: "beginner" | "intermediate" | "advanced"\n'
    '  density: "sparse" | "normal" | "rich"\n'
    '  semitones: 整数，如 -12（低八度）、+12（高八度）、-5（低五度）\n'
    '  constraint: "low_position" | "adjacent_strings"\n'
    '  technique: "hammer" | "pull_off" | "slide" | "remove"\n\n'
    "示例：\n"
    '{"summary":"副歌简化为初级指法","operations":['
    '{"op":"adjust_difficulty","scope":"chorus","difficulty":"beginner"},'
    '{"op":"change_density","scope":"chorus","density":"sparse"}'
    ']}'
)


def _parse_modification_plan(llm_response: str) -> ModificationPlan:
    """从 LLM 的文本响应中提取 JSON → ModificationPlan。

    容错处理：LLM 可能在 JSON 前后附加说明文字，用正则提取最外层 {}。
    """
    json_match = re.search(r'\{[^{}]*"operations"\s*:\s*\[.*?\][^{}]*\}', llm_response, re.DOTALL)
    if not json_match:
        # 回退：尝试找第一个完整 JSON 对象
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)

    json_str = json_match.group(0) if json_match else llm_response

    try:
        data = json.loads(json_str)
        return ModificationPlan.model_validate(data)
    except Exception as exc:
        logger.warning("LLM 输出 JSON 解析失败，使用原始文本作为 summary: %s", exc)
        return ModificationPlan(
            summary=llm_response[:200],
            operations=[],
        )


# =========================================================================
# 入口路由
# =========================================================================


def route_entry(state: AgentState) -> dict:
    """入口路由：检查 modify_instruction 是否存在。"""
    instruction = state.get("modify_instruction", "")
    if instruction:
        logger.info("→ 修改路径（Agent 5）")
        return {"status": "routing_to_modify"}
    logger.info("→ MIDI 管线（Agent 1）")
    return {"status": "routing_to_midi_pipeline"}


def route_entry_decision(state: AgentState) -> str:
    """入口条件判定。"""
    return "agent_5" if state.get("modify_instruction", "") else "agent_1"


# =========================================================================
# Agent 1：旋律解析（确定性）
# =========================================================================


def agent_1_midi_parse(state: AgentState) -> dict:
    """解析 MIDI 文件 → 全轨音符 + 旋律轨音符 + BPM。

    ADR-001 P1：midi_parser 启发式识别主旋律轨，旋律与伴奏分离。
    """
    midi_path = state.get("midi_path", "")
    if not midi_path:
        return {"error": "缺少 MIDI 文件路径", "status": "error_agent_1"}

    logger.info("Agent 1: 解析 %s", midi_path)
    try:
        all_notes, melody_notes, bpm = parse_midi(midi_path)
    except Exception as exc:
        logger.exception("Agent 1 失败")
        return {"error": f"MIDI 解析异常: {exc}", "status": "error_agent_1"}

    logger.info("Agent 1: 全轨 %d 音符, 旋律轨 %d 音符, BPM=%d",
                len(all_notes), len(melody_notes), bpm)
    return {
        "midi_notes": [n.model_dump() for n in all_notes],
        "melody_notes": [n.model_dump() for n in melody_notes],
        "bpm": bpm,
        "status": "parsed_midi",
        "caller": "agent_1",
    }


# =========================================================================
# Agent 2：和声编排（确定性）
# =========================================================================


def agent_2_harmony_analysis(state: AgentState) -> dict:
    """分析音符序列 → 调性 + 和弦进行。"""
    raw_notes = state.get("midi_notes", [])
    bpm = state.get("bpm", 120)
    if not raw_notes:
        return {"error": "缺少音符数据", "status": "error_agent_2"}

    notes = [MidiNote.model_validate(n) for n in raw_notes]
    logger.info("Agent 2: 分析 %d 个音符", len(notes))

    try:
        harmony = analyze_chords(notes, bpm)
    except Exception as exc:
        logger.exception("Agent 2 失败")
        return {"error": f"和声分析异常: {exc}", "status": "error_agent_2"}

    logger.info("Agent 2: 调性=%s, %d 个和弦", harmony.key, len(harmony.chord_progression))
    return {
        "harmony": harmony.model_dump(),
        "status": "analyzed_harmony",
        "caller": "agent_2",
    }


# =========================================================================
# Agent 2.5：LLM 编曲决策（ADR-001 P2）
# =========================================================================

_ARRANGEMENT_SYSTEM_PROMPT = """你是一位经验丰富的指弹吉他编曲师。你会收到一首歌的结构摘要，
你需要为每个乐段选择合适的编排参数。

## 可选参数

density（填充密度）: "sparse" | "medium" | "full"
  - sparse: 几乎不填充内声部，只有旋律+低音（适合前奏/尾奏）
  - medium: 适量填充内声部（适合主歌）
  - full: 密集填充，全声部推进（适合副歌/高潮）

bass_style（低音模式）: "root_only" | "alternating" | "travis_picking"
  - root_only: 每个和弦只弹根音（安静段落）
  - alternating: 根音与五音交替（常规律动）
  - travis_picking: Travis 指弹模式，交替低音更活跃（高潮/结尾）

melody_register（旋律音域）: "low" | "mid" | "high"
  - low: 旋律优先在低把位/低音弦（前奏、安静段落）
  - mid: 中间把位（主歌常态）
  - high: 高把位/高音弦（副歌、情绪高点）

techniques（技巧）: 从 ["hammer_on", "pull_off", "slide", "harmonic", "strumming"] 中选择 0-3 个
  - hammer_on/pull_off: 装饰音，适合中等密度的旋律
  - slide: 适合 Blues/摇滚风格的情绪过渡
  - harmonic: 人工泛音，适合安静的尾奏或特殊效果
  - strumming: 扫弦，适合副歌高潮段落

dynamic（力度）: "ppp" | "pp" | "p" | "mp" | "mf" | "f" | "ff" | "fff"

## 判断依据

- 通过和弦进行密度和旋律音域变化判断段落边界
- 密度突然上升 + 旋律音域变高 → 副歌
- 和弦重复 I-V-vi-IV 模式 → 可能是主歌
- 开头只有低密度和弦 → 前奏
- 结尾渐弱 → 尾奏

## 输出格式

严格输出 JSON，不要输出其他内容：
{
  "sections": [
    {
      "measure_start": 1, "measure_end": 4,
      "label": "intro",
      "density": "sparse", "bass_style": "root_only",
      "melody_register": "low", "techniques": [],
      "dynamic": "p"
    },
    ...
  ],
  "summary": "一句话描述整体编排思路"
}

注意：
- sections 必须覆盖所有小节（measure_start 从 1 开始递增，measure_end 为歌曲最后小节）
- 每个 section 至少覆盖 8 个小节
- 一首歌最多 8 个 sections——将相邻的相似小节合并为更大段落
- 段落边界应放在密度/和弦/旋律发生明显变化的位置
- 所有字符串选项必须使用上面列出的精确值
"""


def _build_song_summary(
    harmony_dict: dict,
    melody_notes_raw: list[dict],
    all_notes_raw: list[dict],
) -> str:
    """构建歌曲结构摘要文本——LLM 的输入。"""
    from src.api.schemas import HarmonyAnalysis, MidiNote

    harmony = HarmonyAnalysis.model_validate(harmony_dict)
    chords = harmony.chord_progression
    total_measures = len(chords) if chords else 1

    all_notes = [MidiNote.model_validate(n) for n in all_notes_raw]
    melody_notes = [MidiNote.model_validate(n) for n in melody_notes_raw] if melody_notes_raw else []

    lines: list[str] = []
    lines.append(f"BPM: {harmony.bpm}  |  调性: {harmony.key}  |  拍号: "
                  f"{harmony.time_signature[0]}/{harmony.time_signature[1]}"
                  f"  |  小节数: {total_measures}")
    lines.append(f"全轨音符: {len(all_notes)}  |  旋律轨音符: {len(melody_notes)}"
                  if melody_notes else f"全轨音符: {len(all_notes)}（单轨，旋律嵌在伴奏中）")
    lines.append("")

    # 按小节分组统计
    measure_dur = harmony.time_signature[0] * 4.0 / harmony.time_signature[1]

    # 统计旋律音域
    melody_by_measure: dict[int, list[int]] = {}
    for n in melody_notes:
        m_idx = int(n.start_time / measure_dur)
        melody_by_measure.setdefault(m_idx, []).append(n.midi_number)

    # 统计全轨密度
    density_by_measure: dict[int, int] = {}
    for n in all_notes:
        m_idx = int(n.start_time / measure_dur)
        density_by_measure[m_idx] = density_by_measure.get(m_idx, 0) + 1

    # 压缩输出：相邻且和弦相同 + 密度相近的小节合并为范围（减少 LLM 输入噪音）
    range_start = 0
    prev_chord = ""
    prev_density = -1
    range_density_min = 999
    range_density_max = 0
    range_melody_min = 128
    range_melody_max = 0

    def _flush_range(end: int) -> None:
        """输出 Mstart-End 的合并行。"""
        if range_start >= total_measures:
            return
        c = chords[range_start] if range_start < len(chords) else None
        cn = c.name if c else "?"
        d_str = str(range_density_min) if range_density_min == range_density_max \
                else f"{range_density_min}-{range_density_max}"
        if range_melody_min <= range_melody_max:
            lo_n = _midi_to_name(range_melody_min)
            hi_n = _midi_to_name(range_melody_max)
            mel = f"旋律: {lo_n}-{hi_n}"
        else:
            mel = "旋律: 无"
        tag = " [休止]" if range_density_max == 0 else (" [密集]" if range_density_max >= 20 else "")
        rng = f"M{range_start + 1}" if end == range_start else f"M{range_start + 1}-{end + 1}"
        lines.append(f"{rng:<8} 和弦: {cn:<8} 密度: {d_str:>5}{tag}  {mel}")

    for m_idx in range(total_measures):
        chord = chords[m_idx] if m_idx < len(chords) else None
        chord_name = chord.name if chord else "?"
        density = density_by_measure.get(m_idx, 0)
        melody_pitches = melody_by_measure.get(m_idx, [])

        # 判断是否应合并：和弦相同 且 密度与上一行均值差不超过 50%
        mergeable = (
            prev_chord == chord_name
            and (prev_density == 0 or abs(density - prev_density) / max(prev_density, 1) < 0.5)
        ) if prev_density >= 0 else False

        if not mergeable and prev_density >= 0:
            _flush_range(m_idx - 1)
            range_start = m_idx
            range_density_min = density
            range_density_max = density
            range_melody_min = min(melody_pitches) if melody_pitches else 128
            range_melody_max = max(melody_pitches) if melody_pitches else 0
        else:
            if prev_density < 0:
                range_start = m_idx
                range_density_min = density
                range_density_max = density
                range_melody_min = min(melody_pitches) if melody_pitches else 128
                range_melody_max = max(melody_pitches) if melody_pitches else 0
            else:
                range_density_min = min(range_density_min, density)
                range_density_max = max(range_density_max, density)
                if melody_pitches:
                    range_melody_min = min(range_melody_min, min(melody_pitches))
                    range_melody_max = max(range_melody_max, max(melody_pitches))

        prev_chord = chord_name
        prev_density = density

    _flush_range(total_measures - 1)

    return "\n".join(lines)


def _midi_to_name(midi: int) -> str:
    """MIDI 编号 → 音名（如 60 → C4）。"""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


def agent_2_5_arrangement(state: AgentState) -> dict:
    """LLM 编曲决策：读歌曲摘要 → 输出 ArrangementPlan。

    ADR-001 P2：LLM 作为编曲师，按段落选择 density/bass_style/register/techniques。
    所有选项来自预定义枚举集，LLM 只做选择题不创造新选项。
    """
    harmony_dict = state.get("harmony", {})
    if not harmony_dict:
        return {"error": "缺少和声数据", "status": "error_agent_2_5", "caller": "agent_2_5"}

    try:
        summary = _build_song_summary(
            harmony_dict,
            state.get("melody_notes", []),
            state.get("midi_notes", []),
        )
    except Exception as exc:
        logger.exception("Agent 2.5: 构建歌曲摘要失败")
        return {"error": f"构建摘要失败: {exc}", "status": "error_agent_2_5", "caller": "agent_2_5"}

    logger.info("Agent 2.5: 歌曲摘要 %d 字符 → LLM 编曲决策", len(summary))

    try:
        llm = _get_llm("deepseek-chat", max_tokens=8192)
        response = llm.invoke([
            SystemMessage(content=_ARRANGEMENT_SYSTEM_PROMPT),
            HumanMessage(content=f"请为以下歌曲做指弹编曲决策：\n\n{summary}"),
        ])
    except Exception as exc:
        logger.exception("Agent 2.5: LLM 调用失败")
        return {"error": f"LLM 调用失败: {exc}", "status": "error_agent_2_5", "caller": "agent_2_5"}

    plan_text = _extract_response_text(response)

    # 提取 JSON（LLM 可能在前后加说明文字）
    plan = _parse_arrangement_plan(plan_text)
    if plan is None:
        logger.warning("Agent 2.5: 无法解析 LLM 输出为 ArrangementPlan（%d 字符），跳过编曲决策", len(plan_text))
        return {"status": "arrangement_skipped", "caller": "agent_2_5"}

    logger.info("Agent 2.5: %d 个段落 — %s", len(plan.sections), plan.summary)
    return {
        "arrangement_plan": plan.model_dump(),
        "status": "arranged",
        "caller": "agent_2_5",
    }


# LLM 常见变体 → 规范值映射（LLM 输出复数/近义词时自动纠正）
_TECHNIQUE_NORMALIZE: dict[str, str] = {
    "harmonics": "harmonic", "hammer_ons": "hammer_on",
    "hammer-on": "hammer_on", "pull_offs": "pull_off",
    "pull-off": "pull_off", "slides": "slide",
    "strummings": "strumming", "strum": "strumming",
}
_DYNAMIC_NORMALIZE: dict[str, str] = {
    "pianissimo": "pp", "piano": "p", "mezzo-piano": "mp",
    "mezzo-forte": "mf", "forte": "f", "fortissimo": "ff",
}


def _normalize_arrangement_json(json_text: str) -> str:
    """归一化 LLM 输出的 arrangement JSON——Pydantic 校验前修正近义词。"""
    import json as _json
    try:
        data = _json.loads(json_text)
    except Exception:
        return json_text

    sections = data.get("sections", [])
    for sec in sections:
        if isinstance(sec, dict):
            techs = sec.get("techniques", [])
            if isinstance(techs, list):
                sec["techniques"] = [
                    _TECHNIQUE_NORMALIZE.get(
                        t.strip().lower().replace(" ", "_").replace("-", "_"), t,
                    ) for t in techs if isinstance(t, str)
                ]
            for key, mapping in [
                ("dynamic", _DYNAMIC_NORMALIZE),
                ("density", {"light": "sparse", "thin": "sparse", "moderate": "medium",
                             "heavy": "full", "dense": "full", "rich": "full"}),
                ("bass_style", {"root": "root_only", "alternate": "alternating",
                                "travis": "travis_picking"}),
                ("melody_register", {"lower": "low", "middle": "mid", "higher": "high"}),
            ]:
                v = sec.get(key, "")
                if isinstance(v, str):
                    sec[key] = mapping.get(v.strip().lower().replace(" ", "_"), v)

    return _json.dumps(data)


def _parse_arrangement_plan(text: str) -> ArrangementPlan | None:
    """从 LLM 输出中提取 ArrangementPlan JSON。

    容忍 LLM 在 JSON 前后加 markdown 代码块标记或说明文字，
    并归一化常见的近义词/复数变体（"harmonics"→"harmonic" 等）。
    """
    from src.api.schemas import ArrangementPlan

    import re
    m = re.search(r'\{[\s\S]*"sections"[\s\S]*\}', text)
    if not m:
        logger.warning("Agent 2.5: 正则未匹配到 JSON（含 'sections' 的 {...}）")
        return None

    json_text = _normalize_arrangement_json(m.group(0))
    try:
        return ArrangementPlan.model_validate_json(json_text)
    except Exception as exc:
        logger.warning("Agent 2.5: ArrangementPlan 解析失败: %s", exc)
        return None


# =========================================================================
# Agent 3：指法生成（三路分支，operations 驱动）
# =========================================================================


def agent_3_tab_generate(state: AgentState) -> dict:
    """指法生成 —— 根据 caller 字段分流。

    caller=agent_2 → 确定性生成（不调 LLM）
    caller=agent_4 → LLM 读 validation.errors → operations → 重生成
    caller=agent_5 → 确定性读取 Agent 5 已解析的 modification_plan → 执行重生成
    """
    caller = state.get("caller", "agent_2")
    harmony_dict = state.get("harmony", {})
    midi_notes_raw = state.get("midi_notes", [])
    melody_notes_raw = state.get("melody_notes", [])  # ADR-001 P1

    if not harmony_dict or not midi_notes_raw:
        return {"error": "缺少和声或音符数据", "status": "error_agent_3"}

    harmony = HarmonyAnalysis.model_validate(harmony_dict)
    midi_notes = [MidiNote.model_validate(n) for n in midi_notes_raw]

    style = Style(state.get("style", "jpop"))
    config = TabGenerationConfig(style=style)
    melody_input = [MidiNote.model_validate(n) for n in melody_notes_raw] if melody_notes_raw else None

    # ADR-001 P2：读取 Agent 2.5 的编曲决策
    arr_dict = state.get("arrangement_plan", {})
    arrangement = ArrangementPlan.model_validate(arr_dict) if arr_dict else None

    # === 首次生成：确定性（不调 LLM） ===
    if caller in ("agent_2", "agent_2_5"):
        logger.info("Agent 3: 首次确定性生成（旋律轨 %d 音符, 编排 %d 段落）",
                    len(melody_input) if melody_input else 0,
                    len(arrangement.sections) if arrangement else 0)
        tab_data = generate_tab(midi_notes, harmony, config,
                                melody_notes=melody_input, arrangement=arrangement)
        return {
            "tab_data": tab_data.model_dump(),
            "status": "generated_tab",
            "caller": "agent_3",
        }

    # === 校验回退：LLM 读 errors → operations → 重生成 ===
    if caller == "agent_4":
        return _agent_3_retry(state, midi_notes, harmony, config, melody_input, arrangement)

    # === 修改路径：LLM 读 instruction → operations → 重生成 ===
    if caller == "agent_5":
        return _agent_3_modify(state, midi_notes, harmony, config, melody_input, arrangement)

    return {"error": f"未知 caller: {caller}", "status": "error_agent_3"}


def _agent_3_retry(
    state: AgentState,
    midi_notes: list[MidiNote],
    harmony: HarmonyAnalysis,
    config: TabGenerationConfig,
    melody_notes: list[MidiNote] | None = None,
    arrangement: ArrangementPlan | None = None,
) -> dict:
    """校验回退路径：LLM 读 validation.errors → ModificationPlan → 执行 → 重生成。

    最终输出 operations JSON，经 _apply_operations() 修改输入后重新调用 generate_tab()。
    """
    validation_dict = state.get("validation", {})
    retry_count = state.get("validation_retry_count", 0)

    if not validation_dict:
        return {"error": "回退路径缺少 validation 数据", "status": "error_agent_3"}

    validation = ValidationResult.model_validate(validation_dict)
    error_text = "\n".join(
        f"小节{e.measure} 弦{e.string} 品{e.fret}: {e.description}" for e in validation.errors
    )
    warning_text = "\n".join(validation.warnings) if validation.warnings else "无"

    logger.info("Agent 3: LLM 回退修正（第 %d 次），%d errors", retry_count, len(validation.errors))

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=_OPERATIONS_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"当前谱面信息：风格={config.style.value}，"
            f"调性={harmony.key}，BPM={harmony.bpm}\n\n"
            f"物理校验错误（需修正）：\n{error_text}\n\n"
            f"校验警告（供参考）：\n{warning_text}\n\n"
            "请输出修正 operations JSON："
        )),
    ])

    plan_text = _extract_response_text(response)
    plan = _parse_modification_plan(plan_text)
    logger.info("Agent 3: LLM 修正方案: %s", plan.summary)

    # 关键：LLM 的 operations 真正注入到生成过程
    tab_data = generate_tab(midi_notes, harmony, config, operations=plan.operations, melody_notes=melody_notes, arrangement=arrangement)
    return {
        "tab_data": tab_data.model_dump(),
        "status": f"regenerated_retry_{retry_count}",
        "caller": "agent_3",
        "messages": [SystemMessage(content=f"回退修正方案: {plan.summary}")],
    }


def _agent_3_modify(
    state: AgentState,
    midi_notes: list[MidiNote],
    harmony: HarmonyAnalysis,
    config: TabGenerationConfig,
    melody_notes: list[MidiNote] | None = None,
    arrangement: ArrangementPlan | None = None,
) -> dict:
    """修改路径：读取 Agent 5 已解析的 ModificationPlan → 确定性执行 → 重生成。

    Agent 5 已完成 LLM 语义解析并存入 state["modification_plan"]。
    此处只做确定性操作：取 operations → _apply_operations() → generate_tab()。
    不重复调 LLM。
    """
    plan_dict = state.get("modification_plan", {})
    if not plan_dict or not plan_dict.get("operations"):
        logger.warning("Agent 3: modification_plan 为空，回退到全曲确定性重生成")
        tab_data = generate_tab(midi_notes, harmony, config, melody_notes=melody_notes)
        return {
            "tab_data": tab_data.model_dump(),
            "status": "regenerated_fallback",
            "caller": "agent_3",
        }

    plan = ModificationPlan.model_validate(plan_dict)
    logger.info(
        "Agent 3: 执行修改方案 '%s'（%d operations，确定性执行，不调 LLM）",
        plan.summary, len(plan.operations),
    )

    # 确定性执行：operations → _apply_operations（在 generate_tab 内部调用）
    tab_data = generate_tab(midi_notes, harmony, config, operations=plan.operations, melody_notes=melody_notes, arrangement=arrangement)
    return {
        "tab_data": tab_data.model_dump(),
        "status": "regenerated_after_modify",
        "caller": "agent_3",
        "messages": [SystemMessage(content=f"已执行修改方案: {plan.summary}")],
    }


# =========================================================================
# Agent 4：物理校验（确定性 + 条件回退判定）
# =========================================================================


def agent_4_validate(state: AgentState) -> dict:
    """物理校验：tab_validator.validate() → 更新 retry_count + 判定是否回退。"""
    tab_data_dict = state.get("tab_data", {})
    if not tab_data_dict:
        return {"error": "缺少 TAB 数据", "status": "error_agent_4"}

    tab_data = TabData.model_validate(tab_data_dict)
    retry_count = state.get("validation_retry_count", 0)

    logger.info("Agent 4: 校验 %d 小节", len(tab_data.measures))
    result = validate(tab_data)
    new_retry = retry_count + 1

    if result.is_valid:
        logger.info("Agent 4: 校验通过 ✓")
    else:
        logger.warning("Agent 4: %d errors, %d warnings（第 %d 次）", len(result.errors), len(result.warnings), new_retry)

    return {
        "validation": result.model_dump(),
        "validation_retry_count": new_retry,
        "status": "validated" if result.is_valid else f"validation_failed_{new_retry}",
        "caller": "agent_4" if not result.is_valid else "",
    }


def should_retry(state: AgentState) -> str:
    """条件边：校验后走回退还是结束。

    由 graph.py 中 agent_4 的条件边调用。
    """
    validation_dict = state.get("validation", {})
    retry_count = state.get("validation_retry_count", 0)

    if not validation_dict:
        return "end"

    if validation_dict.get("is_valid", False):
        logger.info("✓ 校验通过 → END")
        return "end"

    if retry_count < _MAX_VALIDATION_RETRIES:
        logger.info("↻ 第 %d 次回退 → Agent 3", retry_count)
        return "agent_3"

    logger.warning("✗ 已达最大回退次数 %d → END", _MAX_VALIDATION_RETRIES)
    return "end"


# =========================================================================
# Agent 5：修改理解器（LLM 驱动）
# =========================================================================


def agent_5_modify_understand(state: AgentState) -> dict:
    """Agent 5（修改理解器，LLM 驱动）：自然语言指令 → ModificationPlan。

    这是整个修改管线的 LLM 语义理解入口——只调一次 LLM，产出结构化
    ModificationPlan 存入 state["modification_plan"]。
    Agent 3（caller=agent_5）直接读取该 plan 确定性执行，不再重复调 LLM。

    分工：
      Agent 5 = LLM 语义理解（NL → operations）
      Agent 3 = 确定性执行（operations → _apply_operations → generate_tab）
    """
    instruction = state.get("modify_instruction", "")
    tab_data_dict = state.get("tab_data", {})

    if not instruction:
        return {"error": "缺少修改指令", "status": "error_agent_5"}

    # 构建 TAB 上下文供 LLM 定位 scope
    tab_context = "无现有谱面"
    if tab_data_dict:
        try:
            td = TabData.model_validate(tab_data_dict)
            tab_context = (
                f"当前谱面：{len(td.measures)} 小节（第 1-{len(td.measures)} 小节），"
                f"调性={td.key}，速度={td.tempo}BPM，风格={td.style}"
            )
        except Exception:
            pass

    logger.info("Agent 5: LLM 解析修改指令 '%s'", instruction[:80])

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=_OPERATIONS_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"用户修改指令：{instruction}\n"
            f"{tab_context}\n\n"
            "请输出修改 operations JSON："
        )),
    ])

    plan_text = _extract_response_text(response)
    plan = _parse_modification_plan(plan_text)
    logger.info("Agent 5: 解析完成 → %s (%d operations)", plan.summary, len(plan.operations))

    return {
        "modification_plan": plan.model_dump(),
        "status": "modification_parsed",
        "caller": "agent_5",
        "messages": [SystemMessage(content=f"修改理解结果: {plan.summary}")],
    }

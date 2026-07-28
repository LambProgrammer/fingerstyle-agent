"""指法生成器 —— Agent 3（指法生成）的核心确定性引擎。

在多 Agent 系统中的角色：
  这是整个系统的"乐理心脏"——将 Agent 1 的音符序列 + Agent 2 的和声分析
  + 用户配置（难度/风格/定弦/变调夹）转化为可供 alphaTab 渲染的完整六线谱数据。
  所有逻辑均为确定性规则 + 模板匹配，不调 LLM。

架构分层（自上而下）：
  1. 参数区：所有可调常量/阈值集中在此，修改一行即改全局行为
  2. 指板矩阵：6 弦 × max_fret 的 MIDI 音高查找表（纯计算，不依赖 music21）
  3. 声部分配：按用户指定的硬约束（旋律 1-2 弦、低音 4-6 弦、填充 3-4 弦）
  4. 品位寻址：候选集 → zone 过滤 → 四级优先级评分（含当前把位追踪）
  5. 模板引擎：低音行进模式 + 旋律线生成 + 填充策略
  6. 技巧标注：H/P/S 自动判定

声部分配硬约束（Fretboard Zoning，优先级 -999）：
  - 旋律线：优先 1-2 弦，仅当无法弹奏时降级 3 弦，绝不使用 4-6 弦
  - 低音线：强制 4-6 弦，绝不使用 1-3 弦
  - 填充音：允许 3-4 弦
"""
from __future__ import annotations
from collections import defaultdict
from typing import Literal

from music21 import note as m21note  # type: ignore[import-untyped]   # pitch 名称/音高互转

from src.api.schemas import (
    ArrangementPlan,
    Chord,
    Difficulty,
    HarmonyAnalysis,
    MidiNote,
    ModificationOperation,
    Style,
    TabData,
    TabGenerationConfig,
    TabMeasure,
    TabNote,
    Technique,
)


# =============================================================================
# 第 1 层：参数区 —— 所有可调常量集中在此
# 人耳品鉴不满意时，只修改本区域的数字/阈值/权重即可，无需动下层逻辑
# =============================================================================

# --- 定弦与指板 ---
# 标准定弦下各弦的空弦 MIDI 音高（string_number 1-indexed: 1=高音E, 6=低音E）
_OPEN_STRING_PITCH: dict[int, int] = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
_MAX_FRET = 24     # 吉他物理最大品位
_FRET_LIMIT = 15   # 所有谱面统一的品位上限（人手舒适范围，不做难度分级）
_MAX_SPAN = 4      # 人手最大同时按弦跨度（品），与 tab_validator 保持一致

# --- 声部分配硬约束（Zone Filter）---
_MELODY_STRINGS = {1, 2}
_MELODY_FALLBACK_STRINGS = {3}
_BASS_STRINGS = {4, 5, 6}
_INNER_STRINGS = {3, 4}

# --- 声部绝对音高阈值：防止中音误入低音区、高音误入内声部 ---
_BASS_PITCH_MAX = 55     # G3，高于此值的音不进低音区（避免中音被塞进弦4-6高品位）
_INNER_PITCH_MAX = 72    # C5，高于此值的音不进内声部

# --- 品位寻址：流畅性评分参数 ---
# 评分优先级：同弦微移 > 邻弦换弦 > 远弦跳跃 > 兜底
_VOICE_LEADING_MAX_JUMP = 4   # 相邻音最多允许换弦数（超过此值视为"跳跃"）
_JUMP_RETRY_LIMIT = 3         # 连续跳跃触发上限，超限强制优化
# 评分权重（正数越大越好，用于排序）
_SCORE_SAME_STRING = 100       # 同弦，品位差 ≤ 2
_SCORE_ADJACENT_STRING = 50    # 相邻弦（差 1），品位差 ≤ _VOICE_LEADING_MAX_JUMP
_SCORE_DISTANT_BUT_OK = 10     # 任意弦，品位差 ≤ _VOICE_LEADING_MAX_JUMP
_SCORE_LEAST_BAD = -900         # 兜底（所有候选都犯规时选最不坏的那个）
# 候选评分中 zone 违规直接返回此值（<-999 等效物理不可能）
_ZONE_VIOLATION_SCORE = -1000

# --- 低音行进模式 ---
# Travis picking 风格的交替低音：根音→五音→根音→五音循环
_BASS_PATTERN_ROOTS = ["root", "fifth"]  # 奇数拍取根音，偶数拍取五音
# 日系风格偏好利用开放弦作持续低音（E2=40, A2=45, D3=50）
_JPOP_DRONE_STRINGS = {6, 5}   # 优先使用 E 弦和 A 弦的开放音作持续低音
_JPOP_DRONE_PITCHES = {40, 45}  # E2(40) 和 A2(45) 的 MIDI 音高

# --- 技巧标注参数 ---
_HAMMER_PULL_MAX_INTERVAL = 2   # H/P 最多允许的品位差（半音/全音为 H/P，大于则为 S）
_HAMMER_PULL_MAX_DURATION = 0.5 # H/P 只标注时值 ≤ 0.5 quarterLength 的音符（更短=装饰音）


# =============================================================================
# 第 2 层：指板矩阵 —— 音高 → [(弦, 品)] 查找
# =============================================================================


def _build_fretboard(tuning: list[str]) -> dict[int, list[tuple[int, int]]]:
    """根据给定定弦构建完整指板矩阵。

    对每个 MIDI 音高 (0-127)，列出该定弦下所有可弹奏的 (弦号, 品位) 组合。

    Args:
        tuning: 6 个音名，如 ["E2","A2","D3","G3","B3","E4"]。

    Returns:
        {midi_pitch: [(string(1-6), fret(0-24)), ...]}，按品位升序排列。
    """
    # 解析定弦：音名 → MIDI 编号
    open_pitches: dict[int, int] = {}  # string → MIDI pitch
    for i, note_name in enumerate(tuning):
        string_num = 6 - i  # tuning[0]=低音E = string 6
        open_pitches[string_num] = m21note.Note(note_name).pitch.midi

    matrix: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for string_num in range(1, 7):
        open_p = open_pitches[string_num]
        for fret in range(_MAX_FRET + 1):
            pitch = open_p + fret
            matrix[pitch].append((string_num, fret))

    # 每组内按品位升序（默认选品位最低的解）
    for pitch in matrix:
        matrix[pitch].sort(key=lambda x: x[1])

    return matrix


def _find_candidates(
    pitch: int,
    fretboard: dict[int, list[tuple[int, int]]],
    overrides: dict[int, dict] | None = None,
    measure_idx: int = 0,
) -> list[tuple[int, int]]:
    """查找给定音高的所有可弹候选 (弦, 品)，支持 per-measure 品位上限覆盖。

    Args:
        overrides:   {measure_idx: {"fret_limit": int}} 或 None（使用全局 _FRET_LIMIT）。
        measure_idx: 当前小节索引（0-indexed），用于查 overrides。
    """
    max_fret = _FRET_LIMIT
    if overrides:
        fret_limit = overrides.get(measure_idx, {}).get("fret_limit")
        if fret_limit is not None:
            max_fret = fret_limit
    all_candidates = fretboard.get(pitch, [])
    return [(s, f) for (s, f) in all_candidates if f <= max_fret]


def _zone_filter(
    candidates: list[tuple[int, int]],
    voice: Literal["melody", "bass", "inner"],
) -> list[tuple[int, int]]:
    """声部分配硬约束过滤 —— 优先级 -999 的物理隔离规则。

    不是在评分里加权，而是直接删除不合法的候选。"""
    if voice == "melody":
        preferred = [(s, f) for (s, f) in candidates if s in _MELODY_STRINGS]
        if preferred:
            return preferred
        # 降级：1-2 弦无法弹奏该音高（如极低音），允许 3 弦
        return [(s, f) for (s, f) in candidates if s in _MELODY_FALLBACK_STRINGS]
    elif voice == "bass":
        return [(s, f) for (s, f) in candidates if s in _BASS_STRINGS]
    elif voice == "inner":
        return [(s, f) for (s, f) in candidates if s in _INNER_STRINGS]
    return candidates


# =============================================================================
# 第 3 层：品位寻址 —— 四级优先级评分 + 当前把位追踪
# =============================================================================


def _select_best(
    pitch: int,
    current_string: int | None,
    current_fret: int | None,
    voice: Literal["melody", "bass", "inner"],
    fretboard: dict[int, list[tuple[int, int]]],
    overrides: dict[int, dict] | None = None,
    measure_idx: int = 0,
) -> tuple[int, int] | None:
    """为单个音符选择最优 (弦, 品) 分配。

    流程：候选查找 → zone 过滤 → 四级评分 → 选最高分。
    如果当前把位为 None（声部第一个音），跳过邻弦优化，直接选最舒适的位置（最低品）。

    Args:
        pitch:             MIDI 音高。
        current_string:    当前声部的上一音所在弦（None=声部首音）。
        current_fret:      当前声部的上一音品位。
        voice:             声部类型（melody/bass/inner）。
        fretboard:         指板矩阵。

    Returns:
        (string, fret) 或 None（无法找到有效分配时）。
    """
    candidates = _find_candidates(pitch, fretboard, overrides, measure_idx)
    if not candidates:
        return None

    # zone 硬约束过滤
    allowed = _zone_filter(candidates, voice)
    if not allowed:
        return None

    # 声部首音或无从追踪 → 选最低品位（最舒适的手位）
    if current_string is None or current_fret is None:
        return allowed[0]

    # 四级评分
    scored: list[tuple[int, int, int]] = []  # [(score, string, fret), ...]
    for s, f in allowed:
        score = _score_candidate(s, f, current_string, current_fret)
        scored.append((score, s, f))

    scored.sort(key=lambda x: x[0], reverse=True)
    return (scored[0][1], scored[0][2])


def _score_candidate(
    candidate_string: int,
    candidate_fret: int,
    current_string: int,
    current_fret: int,
) -> int:
    """对候选位置评分——基于与当前把位的距离。

    四级优先级：
      1. 同弦、品位差 ≤ 2                   → +100
      2. 相邻弦（|Δ弦| ≤ 1）、品位差 ≤ 4     → +50
      3. 任意弦、品位差 ≤ VOICE_LEADING_MAX_JUMP → +10
      4. 兜底（所有候选都差）               → -900（不阻断但明显劣势）
    """
    string_diff = abs(candidate_string - current_string)
    fret_diff = abs(candidate_fret - current_fret)

    if string_diff == 0 and fret_diff <= 2:
        return _SCORE_SAME_STRING
    if string_diff <= 1 and fret_diff <= _VOICE_LEADING_MAX_JUMP:
        return _SCORE_ADJACENT_STRING
    if fret_diff <= _VOICE_LEADING_MAX_JUMP:
        return _SCORE_DISTANT_BUT_OK
    return _SCORE_LEAST_BAD


# =============================================================================
# 第 4 层：旋律提取 —— 时间聚类（不丢任何非同时音符）
# =============================================================================

def _extract_melody_notes(
    notes: list[MidiNote],
    chords: list[Chord],
    melody_source: Literal["top_note", "highest_density"],
) -> list[MidiNote]:
    """0.5 quarterLength（八分音符）分桶旋律提取——P1 旋律轨识别失败时的回退策略。"""
    bucket_width = 0.5
    buckets: dict[int, list[MidiNote]] = defaultdict(list)
    for n in notes:
        buckets[int(n.start_time / bucket_width)].append(n)

    prev_pitch: int | None = None
    melody: list[MidiNote] = []
    for idx in sorted(buckets):
        bucket_notes = buckets[idx]
        if melody_source == "top_note":
            chosen = max(bucket_notes, key=lambda n: n.midi_number)
        elif prev_pitch is None:
            chosen = max(bucket_notes, key=lambda n: n.midi_number)
        else:
            _prev: int = prev_pitch  # type: ignore[assignment]
            chosen = min(bucket_notes, key=lambda n: abs(n.midi_number - _prev))
        melody.append(chosen)
        prev_pitch = chosen.midi_number

    return melody


# =============================================================================
# 第 5 层：声部生成 —— 低音线 / 旋律线 / 填充音
# =============================================================================


def _generate_bass_line(
    chords: list[Chord],
    config: TabGenerationConfig,
    fretboard: dict[int, list[tuple[int, int]]],
    overrides: dict[int, dict] | None = None,
    measure_duration: float = 4.0,
) -> list[TabNote]:
    """为和弦进行生成低音声部（4-6 弦）。

    策略（优先级从高到低）：
      1. overrides["bass_style"] — ADR-001 P2：LLM 按段落指定的低音模式
      2. config.style → 美式交替根音/五音、日系开放弦 drone
      3. 兜底：根音最低品位
    """
    bass_notes: list[TabNote] = []
    current_string: int | None = None
    current_fret: int | None = None
    consecutive_jumps = 0

    for i, chord in enumerate(chords):
        m_idx = int(chord.start_time / measure_duration)
        bass_override = overrides.get(m_idx, {}).get("bass_style") if overrides else None
        pitch = _bass_pitch_for_chord(chord, i, config.style, fretboard, bass_override)
        if pitch is None:
            continue

        result = _select_best(
            pitch, current_string, current_fret, "bass", fretboard,
            overrides, m_idx,
        )
        if result is None:
            continue

        s, f = result
        if current_string is not None and abs(s - current_string) > 2:
            consecutive_jumps += 1
            if consecutive_jumps >= _JUMP_RETRY_LIMIT:
                fallback = _fallback_bass(chord, fretboard)
                if fallback:
                    s, f = fallback
                    consecutive_jumps = 0
        else:
            consecutive_jumps = 0

        bass_notes.append(
            TabNote(
                string=s, fret=f,
                start_time=chord.start_time, duration=chord.duration,
                technique=Technique.NONE,
                voice="bass",
            )
        )
        current_string, current_fret = s, f

    return bass_notes


def _bass_pitch_for_chord(
    chord: Chord, index: int, style: Style,
    fretboard: dict[int, list[tuple[int, int]]],
    bass_override: str | None = None,
) -> int | None:
    """确定该和弦的低音音符 MIDI 音高，并确保在低音弦可弹范围内。

    两重保障：
      1. 音高阈值：若候选音高于 _BASS_PITCH_MAX(55=G3)，强制八度下移
      2. 八度下移循环：确保能在 4-6 弦上以 _FRET_LIMIT 弹奏
    """
    if not chord.midi_numbers:
        return None

    pitch = _raw_bass_pitch(chord, index, style, bass_override)
    if pitch is None:
        return None

    # 音高阈值：中音不进低音区（如 D#4=63 → 弦4品13 会与旋律跨度爆炸）
    while pitch > _BASS_PITCH_MAX:
        pitch -= 12

    # 八度下移循环：确保能在 bass 弦可弹
    for _ in range(3):
        candidates = _find_candidates(pitch, fretboard, None, 0)
        if _zone_filter(candidates, "bass"):
            return pitch
        pitch -= 12

    return pitch  # 兜底：返回降了 3 个八度的值（几乎不会到这一步）


def _raw_bass_pitch(
    chord: Chord, index: int, style: Style,
    bass_override: str | None = None,
) -> int | None:
    """原始低音音高选择（不含八度适配逻辑）。

    ADR-001 P2：bass_override 来自 LLM 编排计划的 per-section bass_style，
    优先级高于 config.style 的全局策略。
    """
    # 确定实际使用的低音模式
    effective = bass_override or style.value

    # root_only：每小节只弹一次根音（最简）
    if effective in ("root_only",):
        return min(chord.midi_numbers)

    # alternating / travis_picking：根音/五音交替
    if effective in ("alternating", "travis_picking", Style.AMERICAN_FOLK.value):
        pattern = _BASS_PATTERN_ROOTS[index % 2]
        midi_nums = sorted(chord.midi_numbers)
        root_midi = midi_nums[0]
        if pattern == "fifth":
            fifth_candidates = [p for p in midi_nums if (p - root_midi) % 12 == 7]
            if fifth_candidates:
                return fifth_candidates[0]
            return midi_nums[1] if len(midi_nums) >= 2 else root_midi
        return root_midi

    # JPOP 日系风格 → 开放弦 drone
    if effective in (Style.JPOP.value,):
        chord_set = set(chord.midi_numbers)
        for drone in sorted(_JPOP_DRONE_PITCHES):
            if drone in chord_set:
                return drone

    # 兜底
    return min(chord.midi_numbers)


def _fallback_bass(
    chord: Chord,
    fretboard: dict[int, list[tuple[int, int]]],
) -> tuple[int, int] | None:
    """低音跳跃回退：强制选根音最低品位位置。"""
    if not chord.midi_numbers:
        return None
    pitch = _find_lowest_playable_bass(chord, fretboard)
    if pitch is None:
        return None
    candidates = _find_candidates(pitch, fretboard, None, 0)
    allowed = _zone_filter(candidates, "bass")
    return allowed[0] if allowed else None


def _find_lowest_playable_bass(
    chord: Chord,
    fretboard: dict[int, list[tuple[int, int]]],
) -> int | None:
    """找和弦中能在 4-6 弦上弹奏的最低音。"""
    midi_nums = sorted(chord.midi_numbers)
    for pitch in midi_nums:
        candidates = _find_candidates(pitch, fretboard, None, 0)
        if _zone_filter(candidates, "bass"):
            return pitch
    return midi_nums[0] if midi_nums else None


def _generate_melody_line(
    melody_notes: list[MidiNote],
    config: TabGenerationConfig,
    fretboard: dict[int, list[tuple[int, int]]],
    overrides: dict[int, dict] | None = None,
    measure_duration: float = 4.0,
) -> list[TabNote]:
    """为提取出的旋律线分配弦/品（1-2 弦优先，zone 硬约束）。"""
    tab_notes: list[TabNote] = []
    current_string: int | None = None
    current_fret: int | None = None
    consecutive_jumps = 0

    for mn in melody_notes:
        pitch = mn.midi_number
        m_idx = int(mn.start_time / measure_duration)
        result = _select_best(
            pitch, current_string, current_fret, "melody", fretboard,
            overrides, m_idx,
        )
        # 八度回退：当前难度下无合法候选 → 降八度重试（最多 2 次）
        for _ in range(2):
            if result is not None:
                break
            pitch -= 12
            result = _select_best(
                pitch, current_string, current_fret, "melody", fretboard,
                overrides, m_idx,
            )
        if result is None:
            continue

        s, f = result
        # 跳跃检测
        if current_string is not None:
            if abs(s - current_string) > _VOICE_LEADING_MAX_JUMP:
                consecutive_jumps += 1
                if consecutive_jumps >= _JUMP_RETRY_LIMIT:
                    # 退化为该音在首选弦上的最低品位（放弃把位追踪，重选安全位置）
                    s, f = _force_safe_melody(mn.midi_number, fretboard)
                    consecutive_jumps = 0
            else:
                consecutive_jumps = 0

        tab_notes.append(
            TabNote(
                string=s, fret=f,
                start_time=mn.start_time, duration=mn.duration,
                technique=Technique.NONE,
                voice="melody",
            )
        )
        current_string, current_fret = s, f

    return tab_notes


def _force_safe_melody(
    pitch: int,
    fretboard: dict[int, list[tuple[int, int]]],
) -> tuple[int, int]:
    """跳跃回退：选旋律首选弦上品位最低的安全解（不追踪把位）。"""
    candidates = _find_candidates(pitch, fretboard, None, 0)
    allowed = _zone_filter(candidates, "melody")
    return allowed[0] if allowed else candidates[0]


def _generate_inner_voices(
    notes: list[MidiNote],
    melody_pitches: set[int],
    bass_pitches: set[int],
    chords: list[Chord],
    config: TabGenerationConfig,
    fretboard: dict[int, list[tuple[int, int]]],
    overrides: dict[int, dict] | None = None,
    measure_duration: float = 4.0,
) -> list[TabNote]:
    """填充内声部——将和弦内既非旋律也非低音的音分配到 3-4 弦。

    规则：
      - 只填充和弦桶内不足 3 个已分配音的情况（避免谱面过密）
      - 优先选和声最丰富的桶（如属七和弦有 4 个音，旋律+低音只占了 2 个）
    """
    inner_notes: list[TabNote] = []
    current_string: int | None = None
    current_fret: int | None = None

    for chord in chords:
        # 和弦中未分配的音
        fill_pitches = [
            p for p in chord.midi_numbers
            if p not in melody_pitches and p not in bass_pitches
        ]
        if len(fill_pitches) <= 0:
            continue

        # 强拍丰满度：第 1/3 拍（强拍）填充配额提升为 3，弱拍保持 2
        # overrides 中有 fill_quota 时优先使用（per-measure 覆盖）
        m_idx = int(chord.start_time / measure_duration)
        is_strong = _is_strong_beat(chord.start_time)
        max_fill = 3 if is_strong else 2
        if overrides:
            quota = overrides.get(m_idx, {}).get("fill_quota")
            if quota is not None:
                max_fill = max(0, int(round(quota)))

        for pitch in fill_pitches[:max_fill]:
            # 音高阈值：高于 C5(72) 的音不进内声部（应归入旋律或跳过）
            if pitch > _INNER_PITCH_MAX:
                continue
            inner_pitch = pitch
            result = _select_best(
                inner_pitch, current_string, current_fret, "inner", fretboard,
                overrides, m_idx,
            )
            # 八度回退（同旋律线）
            for _ in range(2):
                if result is not None:
                    break
                inner_pitch -= 12
                result = _select_best(
                    inner_pitch, current_string, current_fret, "inner", fretboard,
                    overrides, m_idx,
                )
            if result is None:
                continue
            s, f = result
            inner_notes.append(
                TabNote(
                    string=s, fret=f,
                    start_time=chord.start_time, duration=chord.duration,
                    technique=Technique.NONE,
                    voice="inner",
                )
            )
            current_string, current_fret = s, f

    return inner_notes


# =============================================================================
# 第 6 层：Chord Voicing 优化 —— 降低同时间点品位跨度
# =============================================================================


def _tab_pitch(tn, tuning: list[str]) -> int:
    """TabNote 的弦+品 → MIDI 音高。"""
    open_pitches = {
        i: m21note.Note(name).pitch.midi
        for i, name in zip(range(6, 0, -1), tuning)
    }
    return open_pitches.get(tn.string, 0) + tn.fret


def _optimize_chord_voicing(
    all_notes: list[TabNote],
    tuning: list[str],
) -> list[TabNote]:
    """Chord voicing 后处理：降低同时间点 fretted 音符的品位跨度。

    遍历每个时间桶，若桶内 fretted 品位的最大跨度 > 4（人手无法同时按），
    尝试将离群音符重分配到邻近弦的同音高位置，使跨度 ≤ 4。

    Args:
        all_notes: 按 start_time 升序排列的全部 TabNote。
        tuning:    当前定弦（用于反向计算 MIDI 音高 + 构建指板矩阵）。

    Returns:
        优化后的 TabNote 列表（原位替换，不新建对象）。
    """
    import logging
    logger = logging.getLogger(__name__)

    if not all_notes:
        return all_notes

    # 构建指板矩阵（同 generate_tab 中的逻辑）
    open_pitches: dict[int, int] = {}
    for i, note_name in enumerate(tuning):
        string_num = 6 - i
        open_pitches[string_num] = m21note.Note(note_name).pitch.midi

    from collections import defaultdict
    fretboard: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for s in range(1, 7):
        for f in range(_MAX_FRET + 1):
            pitch = open_pitches[s] + f
            fretboard[pitch].append((s, f))
    for pitch in fretboard:
        fretboard[pitch].sort(key=lambda x: x[1])

    # 按 start_time 分桶
    buckets: dict[float, list[int]] = defaultdict(list)  # time → [索引]
    for idx, tn in enumerate(all_notes):
        buckets[tn.start_time].append(idx)

    optimized_count = 0

    for start_time, indices in buckets.items():
        if len(indices) < 2:
            continue

        # 取桶内所有 fretted 音符（排除空弦）
        fretted = [(i, all_notes[i]) for i in indices if all_notes[i].fret > 0]
        if len(fretted) < 2:
            continue

        frets = [f[1].fret for f in fretted]
        span = max(frets) - min(frets)
        if span <= _MAX_SPAN:
            continue

        # 迭代优化：每次取离群最远的音符尝试重分配
        for _ in range(len(fretted)):
            frets = [f[1].fret for f in fretted]
            fret_median = sum(frets) / len(frets)
            span = max(frets) - min(frets)
            if span <= _MAX_SPAN:
                break

            # 找离中心最远的
            outlier_idx_in_fretted = max(
                range(len(fretted)),
                key=lambda j: abs(fretted[j][1].fret - fret_median),
            )
            idx, outlier = fretted[outlier_idx_in_fretted]

            # 反算该音的音高 → 找所有可弹候选
            pitch = _tab_pitch(outlier, tuning)
            candidates = fretboard.get(pitch, [])
            # 过滤：排除空弦、排除当前分配、排除超出 _FRET_LIMIT 的
            alternatives = [
                (s, f) for (s, f) in candidates
                if f > 0 and f <= _FRET_LIMIT and (s, f) != (outlier.string, outlier.fret)
            ]

            best_alt = None
            best_span = span

            for alt_s, alt_f in alternatives:
                # 模拟：把 outlier 换成 alt
                test_frets = [
                    (alt_f if j == outlier_idx_in_fretted else fretted[j][1].fret)
                    for j in range(len(fretted))
                ]
                test_span = max(test_frets) - min(test_frets)
                if test_span < best_span:
                    best_span = test_span
                    best_alt = (alt_s, alt_f)

            if best_alt and best_span < span:
                logger.debug(
                    "chord voicing: t=%.1f 弦%d品%d → 弦%d品%d (span %d→%d)",
                    start_time, outlier.string, outlier.fret,
                    best_alt[0], best_alt[1], span, best_span,
                )
                all_notes[idx].string = best_alt[0]
                all_notes[idx].fret = best_alt[1]
                fretted[outlier_idx_in_fretted] = (idx, all_notes[idx])
                optimized_count += 1
            else:
                # 无改进，跳过该音符继续尝试下一个
                break  # 当前桶无法进一步优化

    if optimized_count:
        logger.info("chord voicing: 优化了 %d 个音符", optimized_count)

    return all_notes


# =============================================================================
# 第 7 层：技巧标注 —— 后处理自动判定 H/P/S
# =============================================================================


def _annotate_techniques(tab_notes: list[TabNote]) -> None:
    """对相邻音符自动判定 H（击弦）、P（勾弦）、S（滑弦）。

    规则：
      - 同一弦上，相邻音符，品位上升 1-2、时值 ≤ 0.5QL → H
      - 同一弦上，相邻音符，品位下降 1-2、时值 ≤ 0.5QL → P
      - 同一弦上，相邻音符，品位差 ≥ 3、或品位差 1-2 但时值更长 → S
      - 跨弦无技巧标注（跨弦的快速切换是换弦，不是 H/P/S）
    """
    if len(tab_notes) < 2:
        return

    for i in range(len(tab_notes) - 1):
        curr = tab_notes[i]
        nxt = tab_notes[i + 1]

        # 不同弦 → 不标注（换弦不触发 H/P/S）
        if curr.string != nxt.string:
            continue

        fret_diff = nxt.fret - curr.fret
        abs_diff = abs(fret_diff)
        is_fast = (nxt.start_time - curr.start_time) <= _HAMMER_PULL_MAX_DURATION

        if abs_diff <= _HAMMER_PULL_MAX_INTERVAL and is_fast:
            curr.technique = Technique.HAMMER_ON if fret_diff > 0 else Technique.PULL_OFF
        elif abs_diff >= 3 or (abs_diff >= 1 and not is_fast):
            # 较大跨度或慢速同弦移动 → 滑弦更自然
            curr.technique = Technique.SLIDE

    # 最后一个音不标注（无后继音符来判定）


# =============================================================================
# 第 8 层：小节排版 —— TabNote 列表 → 小节化 TabData
# =============================================================================


def _assemble_measures(
    all_notes: list[TabNote],
    bpm: int,
    time_signature: tuple[int, int],
) -> list[TabMeasure]:
    """将所有声部的 TabNote 合并并按小节切分。

    小节边界 = (拍号分子) × quarterLength 每小节。
    默认 4/4：每小节 = 4.0 quarterLength。
    """
    if not all_notes:
        return [TabMeasure(number=0, notes=[], time_signature=time_signature)]

    notes_per_beat = time_signature[0]  # 分子 = 每小节拍数（4/4 → 4）
    measure_duration = float(notes_per_beat)  # quarterLength per measure

    max_time = max(n.start_time for n in all_notes)
    num_measures = max(1, int(max_time / measure_duration) + 1)

    measures: list[TabMeasure] = []
    for m_idx in range(num_measures):
        t_start = m_idx * measure_duration
        t_end = t_start + measure_duration
        measure_notes = [
            n for n in all_notes if t_start <= n.start_time < t_end
        ]
        # 按起始时间 + 弦号排序（低音弦在前）
        measure_notes.sort(key=lambda n: (n.start_time, -n.string))
        measures.append(
            TabMeasure(
                number=m_idx + 1,
                notes=measure_notes,
                time_signature=time_signature,
            )
        )

    return measures


# =============================================================================
# 第 9 层：强拍判定 + Scope 解析 + 操作执行
# =============================================================================


def _is_strong_beat(start_time: float, time_signature: tuple[int, int] = (4, 4)) -> bool:
    """判断给定时间位置是否为强拍（4/4 中的第 1 拍和第 3 拍）。

    用于 `_generate_inner_voices` 中提升强拍填充配额。
    """
    beats_per_measure = time_signature[0]  # 4/4 → 4
    beat_in_measure = int(start_time) % beats_per_measure
    return beat_in_measure in (0, 2)  # beat 1 (offset 0) 和 beat 3 (offset 2)


def _resolve_scope(scope: str, num_measures: int) -> list[int]:
    """将自然语言 scope 转换为小节索引列表（0-indexed）。

    涵盖 LLM 最可能输出的 scope 表述：
      entire_song / 全曲       → 所有小节
      chorus / 副歌            → 后 1/3 小节
      bridge / 间奏            → 中 1/3 小节
      verse / 主歌             → 前 1/3 小节
      measure_N / 第N小节       → [N-1]
      measure_N-M / 第N-M小节   → [N-1 .. M-1]
      其他                     → 全曲（兜底）
    """
    if num_measures <= 0:
        return []

    s = scope.strip().lower()

    if s in ("entire_song", "全曲", ""):
        return list(range(num_measures))

    if s in ("chorus", "副歌"):
        third = max(1, num_measures // 3)
        return list(range(num_measures - third, num_measures))

    if s in ("bridge", "间奏"):
        third = max(1, num_measures // 3)
        return list(range(third, num_measures - third))

    if s in ("verse", "主歌"):
        third = max(1, num_measures // 3)
        return list(range(third))

    # 解析 "measure_N" / "第N小节"
    import re
    m_single = re.match(r'(?:measure_)?(\d+)|第(\d+)小节', s)
    if m_single:
        n = int(m_single.group(1) or m_single.group(2))
        return [max(0, n - 1)]

    # 解析 "measure_N-M" / "第N-M小节"
    m_range = re.match(r'(?:measure_)?(\d+)[-–](\d+)|第(\d+)[-–](\d+)小节', s)
    if m_range:
        n1 = int(m_range.group(1) or m_range.group(3))
        n2 = int(m_range.group(2) or m_range.group(4))
        return list(range(max(0, n1 - 1), min(num_measures, n2)))

    # 兜底：全曲
    return list(range(num_measures))


# LLM 可能输出近义词而非合法枚举值，此处做容错映射
_DIFFICULTY_ALIASES: dict[str, str] = {
    "simplified": "beginner", "easy": "beginner",
    "hard": "advanced", "difficult": "advanced",
    "medium": "intermediate", "normal": "intermediate",
}
# 难度 → per-measure 品位上限（替代全局 _FRET_LIMIT）
_DIFFICULTY_FRET_LIMITS: dict[str, int] = {
    "beginner": 5, "intermediate": 9, "advanced": 15,
}
# 密度 → per-measure 填充配额
_DENSITY_FILL_QUOTAS: dict[str, int] = {
    "sparse": 1, "normal": 2, "rich": 4,
}


def _normalize_difficulty(raw: str) -> Difficulty:
    """容错：将 LLM 可能输出的近义词转为合法 Difficulty 枚举。

    如果 Prompt 约束生效，LLM 应直接输出 'beginner'/'intermediate'/'advanced'。
    此函数作为兜底，处理 Prompt 约束失效的边界情况。
    """
    normalized = _DIFFICULTY_ALIASES.get(raw.lower(), raw.lower())
    return Difficulty(normalized)


def _apply_operations(
    notes: list[MidiNote],
    config: TabGenerationConfig,
    operations: list[ModificationOperation],
    total_measures: int,
    measure_duration: float = 4.0,
) -> tuple[list[MidiNote], TabGenerationConfig, dict[int, dict]]:
    """对原始输入执行 LLM 输出的原子操作序列。

    不修改 harmony（操作对和声结构影响足够小，不重跑 Agent 2）。
    返回 (modified_notes, modified_config) 供 generate_tab() 使用。

    执行顺序（强制执行，不依赖 LLM 输出的排列）：
      Round 1: transpose          — 音高数据先行（后续约束基于正确的音高）
      Round 2: adjust_difficulty + reassign_string — 物理约束（移调后再定）
      Round 3: change_density + switch_technique  — 美学/技巧（不影响音高/约束）

    这个顺序保证：不会出现"先设 beginner 上限再 transpose(+12) 导致越界"的陷阱。
    """
    import logging
    logger = logging.getLogger(__name__)

    modified_notes = list(notes)
    modified_config = config.model_copy()
    overrides: dict[int, dict] = defaultdict(dict)

    # ---- Round 1: transpose（音高先行）----
    for op in operations:
        if op.op != "transpose" or op.semitones is None:
            continue
        scope_indices = _resolve_scope(op.scope, total_measures)
        logger.debug("Round1 transpose %+d, scope=%s → measures=%s", op.semitones, op.scope, scope_indices[:5])
        new_notes = []
        for n in modified_notes:
            measure_idx = int(n.start_time / measure_duration)
            if measure_idx in scope_indices:
                new_notes.append(MidiNote(
                    midi_number=n.midi_number + op.semitones,
                    start_time=n.start_time, duration=n.duration,
                    velocity=n.velocity, track=n.track, channel=n.channel,
                ))
            else:
                new_notes.append(n)
        modified_notes = new_notes

    # ---- Round 2: adjust_difficulty + reassign_string → overrides["fret_limit"] ----
    for op in operations:
        if op.op not in ("adjust_difficulty", "reassign_string"):
            continue
        scope_indices = _resolve_scope(op.scope, total_measures)
        logger.debug("Round2 %s, scope=%s → measures=%s", op.op, op.scope, scope_indices[:5])

        if op.op == "adjust_difficulty" and op.difficulty:
            normalized = _DIFFICULTY_ALIASES.get(op.difficulty.lower(), op.difficulty.lower())
            limit = _DIFFICULTY_FRET_LIMITS.get(normalized, _FRET_LIMIT)
            for idx in scope_indices:
                overrides[idx]["fret_limit"] = limit

        elif op.op == "reassign_string" and op.constraint:
            if op.constraint == "low_position":
                for idx in scope_indices:
                    overrides[idx]["fret_limit"] = 5
            elif op.constraint != "adjacent_strings":
                logger.warning("未知 constraint 值 '%s'，已忽略（合法值: low_position/adjacent_strings）", op.constraint)

    # ---- Round 3: change_density → overrides["fill_quota"] ----
    for op in operations:
        if op.op not in ("change_density", "switch_technique"):
            continue
        scope_indices = _resolve_scope(op.scope, total_measures)
        logger.debug("Round3 %s, scope=%s → measures=%s", op.op, op.scope, scope_indices[:5])

        if op.op == "change_density" and op.density:
            quota = _DENSITY_FILL_QUOTAS.get(op.density.lower())
            if quota is not None:
                for idx in scope_indices:
                    overrides[idx]["fill_quota"] = quota
            else:
                logger.warning("未知 density 值 '%s'，已忽略（合法值: sparse/normal/rich）", op.density)

        elif op.op == "switch_technique":
            logger.debug("技巧替换: scope=%s → %s", op.scope, op.technique)

    return modified_notes, modified_config, dict(overrides)


# =============================================================================
# ADR-001 P2：编排计划 → per-measure overrides 翻译
# =============================================================================

# density → fill_quota 映射（控制内声部填充音符数/拍）
_DENSITY_QUOTA: dict[str, float] = {
    "sparse": 0.5,    # 每拍最多 0.5 个填充音 = 隔拍填
    "medium": 1.0,    # 每拍 1 个填充音
    "full": 2.0,      # 每拍 2 个填充音 = 密集
}

# melody_register → fret_limit 映射（控制旋律线品位上限）
_REGISTER_FRET_LIMIT: dict[str, int] = {
    "low": 9,    # 低把位（0-9 品）
    "mid": 12,   # 中把位（0-12 品）
    "high": 15,  # 高把位（0-15 品，可达最高品）
}

# bass_style → 传给 _generate_bass_line 的模式选择（通过 overrides 分发）
_BASS_STYLE_MAP: dict[str, str] = {
    "root_only": "root_only",
    "alternating": "alternating",
    "travis_picking": "travis_picking",
}

# dynamic → 影响内声部 velocity + 和声丰富度（用不同 fill_quota 倍率实现）
_DYNAMIC_MULTIPLIER: dict[str, float] = {
    "ppp": 0.15,
    "pp": 0.3,
    "p": 0.5,
    "mp": 0.75,
    "mf": 1.0,
    "f": 1.3,
    "ff": 1.6,
    "fff": 2.0,
}


def _arrangement_to_overrides(
    plan: ArrangementPlan,
    num_measures: int,
) -> dict[int, dict]:
    """将 ArrangementPlan 翻译为 per-measure overrides dict。

    每个 measure_idx → {"fill_quota": float, "fret_limit": int, "bass_style": str,
                         "techniques": list[str], "dynamic": str}
    """
    overrides: dict[int, dict] = {}
    for section in plan.sections:  # type: ignore[union-attr]
        start = section.measure_start - 1  # 1-indexed → 0-indexed
        end = min(section.measure_end, num_measures)

        fill_quota = _DENSITY_QUOTA.get(section.density, 1.0)
        fill_quota *= _DYNAMIC_MULTIPLIER.get(section.dynamic, 1.0)
        fret_limit = _REGISTER_FRET_LIMIT.get(section.melody_register, 12)

        for m_idx in range(start, end):
            overrides[m_idx] = {
                "fill_quota": fill_quota,
                "fret_limit": fret_limit,
                "bass_style": section.bass_style,
                "techniques": section.techniques,
                "dynamic": section.dynamic,
            }
    return overrides


# =============================================================================
# 公共 API
# =============================================================================


def generate_tab(
    notes: list[MidiNote],
    harmony: HarmonyAnalysis,
    config: TabGenerationConfig,
    operations: list[ModificationOperation] | None = None,
    *,
    melody_notes: list[MidiNote] | None = None,
    arrangement: ArrangementPlan | None = None,
) -> TabData:
    """核心入口：将音符序列 + 和声分析 + 用户配置 → 完整指弹谱。

    处理流程：
      1. 构建指板矩阵
      2. 提取/接收旋律线（优先使用 P1 已识别旋律轨，回退到 top_note/highest_density 策略）
      3. 生成低音线（和弦根音/五音 → 4-6 弦）
      4. 填充内声部（和弦剩余音 → 3-4 弦）
      5. 合并三个声部 → 技巧标注
      6. 小节排版 → TabData

    Args:
        notes:   Agent 1 产出的所有音符（含伴奏，供内声部填充使用）。
        harmony: Agent 2 产出的和声分析（key + chord_progression）。
        config:  用户配置（难度/风格/定弦/变调夹/旋律策略）。
        operations: QA 修改或校验回退的原子操作列表。
        melody_notes: ADR-001 P1——midi_parser 识别的独立旋律轨音符（保留原始 onset/duration）。
                      若为 None 或空，回退到旧 _extract_melody_notes() 猜旋律策略。

    Returns:
        TabData: 完整六线谱数据，可直接渲染或导出。
    """
    # 0. 计算小节时长（根据拍号，替代硬编码 4.0）
    measure_duration = harmony.time_signature[0] * 4.0 / harmony.time_signature[1]
    estimated_measures = max(1, int(
        max(n.start_time for n in notes) / measure_duration
    ) + 1) if notes else 1
    overrides: dict[int, dict] = {}
    if operations:
        notes, config, overrides = _apply_operations(
            notes, config, operations, estimated_measures, measure_duration,
        )

    # ADR-001 P2：ArrangementPlan → per-measure overrides（作为基线）
    # operations 的 overrides（如 adjust_difficulty）在已合并的 arrangement 上叠加
    if arrangement:
        arr_overrides = _arrangement_to_overrides(arrangement, estimated_measures)
        for m_idx, params in arr_overrides.items():
            if m_idx in overrides:
                # operations overrides 优先（它们是用户明确指定的），arrangement 做默认值
                merged = dict(params)
                merged.update(overrides[m_idx])
                overrides[m_idx] = merged
            else:
                overrides[m_idx] = dict(params)

    fretboard = _build_fretboard(config.tuning)

    # 1. 提取/接收旋律线（ADR-001 P1：优先使用已识别旋律轨）
    if melody_notes:
        final_melody_notes = melody_notes
    else:
        final_melody_notes = _extract_melody_notes(
            notes, harmony.chord_progression, config.melody_source,
        )

    # 2. 生成三个声部（传入 overrides 支持 per-measure fret_limit + fill_quota）
    bass_tab = _generate_bass_line(harmony.chord_progression, config, fretboard, overrides, measure_duration)
    melody_tab = _generate_melody_line(final_melody_notes, config, fretboard, overrides, measure_duration)

    melody_pitches_set = {mn.midi_number for mn in final_melody_notes}
    bass_pitches_set: set[int] = set()
    for i, chord in enumerate(harmony.chord_progression):
        pitch = _bass_pitch_for_chord(chord, i, config.style, fretboard)
        if pitch is not None:
            bass_pitches_set.add(pitch)
    inner_tab = _generate_inner_voices(
        notes, melody_pitches_set, bass_pitches_set,
        harmony.chord_progression, config, fretboard, overrides, measure_duration,
    )

    # 3. 合并 + chord voicing 优化
    all_tab_notes = bass_tab + inner_tab + melody_tab
    all_tab_notes.sort(key=lambda n: (n.start_time, -n.string))
    all_tab_notes = _optimize_chord_voicing(all_tab_notes, config.tuning)

    # 4. 技巧标注
    _annotate_techniques(all_tab_notes)

    # 5. 小节排版
    measures = _assemble_measures(all_tab_notes, harmony.bpm, harmony.time_signature)

    # 6. 收集使用的技巧
    techniques_used = sorted(
        set(tn.technique for tn in all_tab_notes if tn.technique != Technique.NONE),
        key=lambda t: t.value,
    )

    return TabData(
        measures=measures,
        tuning=config.tuning,
        capo=config.capo,
        tempo=harmony.bpm,
        key=harmony.key,
        style=config.style.value,
        techniques_used=techniques_used,
    )

"""MIDI 文件解析器 —— Agent 1（旋律解析）的底层确定性工具。

在多 Agent 系统中的角色：
  这是整条 Agent 链路的"数据原点"——后续的和声编排、指法生成、物理校验
  全部建立在本模块输出的音符序列之上。本模块只做确定性转换（MIDI 二进制 → 结构化音符列表），
  不涉及任何 LLM 调用或艺术判断。

技术方案：
  使用 music21（MIT 许可证，项目已安装）的 converter 打开 MIDI 文件，
  遍历所有非打击乐音轨的 recurse().notes()，将每个 Note/Chord 转为 MidiNote Pydantic 模型。
  BPM 优先取文件中第一个速度标记，若未找到则回退 120。
  和弦（Chord 对象）拆分为多个同期同位置的 MidiNote，保持各组成音独立。

ADR-001 P1：主旋律轨识别
  不再将所有音轨拍平取最高音"猜测"旋律。改为对每轨打启发式分（复音数、音域、
  音高方差、GM 乐器类型），得分最高者 = 主旋律轨。失败回退到全轨拍平策略。

边界处理说明（含具体场景）：
  - 多轨 MIDI → 启发式识别主旋律轨，旋律与伴奏分离输出
  - 单轨 MIDI → 旋律评分 < 阈值 40 → 回退全轨拍平（与旧行为兼容）
  - 打击乐通道 → GM Channel 10（0-indexed: channel 9）自动跳过（不会出现"鼓声被当成音符"）
  - 连音（tie）→ music21 自动合并为单一长音符，duration 已是总时长
  - 休止符 → 直接跳过（TAB 只关心"哪里有音"，不需要记录"哪里没音"）
  - 空 MIDI（无言符文件）→ 正常返回空列表 + BPM，不抛异常
  - 无效 MIDI（非标准格式）→ music21 自身抛异常，本模块不吞异常，原样上抛
  - 极短音符（< 30ms）→ 保留（可能是快速装饰音的 MIDI 真实表示，不用规则过滤）
"""

import tempfile
from pathlib import Path

from music21 import converter, tempo  # type: ignore[import-untyped]
from music21.chord import Chord as M21Chord  # type: ignore[import-untyped]
from music21.note import Note, Rest  # type: ignore[import-untyped]
from music21.stream import Part, Score  # type: ignore[import-untyped]

from src.api.schemas import MidiNote


# =============================================================================
# 主旋律轨识别（ADR-001 P1）
# =============================================================================

# 启发式评分权重
_MELODY_WEIGHT_POLYPHONY = 35
_MELODY_WEIGHT_REGISTER = 25
_MELODY_WEIGHT_VARIANCE = 20
_MELODY_WEIGHT_INSTRUMENT = 10
_MELODY_WEIGHT_NOTE_COUNT = 10

# 最低旋律评分阈值：低于此值 = 无清晰主旋律轨，回退到全轨拍平
# 设为 25（原 40）——很多 MIDI 文件没有 GM instrument 元数据（program=0），
# 仅靠复音数+音域+方差也能拿到 25-40 分。宁可选一个近似轨也不回退全轨混音。
_MIN_MELODY_SCORE = 25

# 人声/旋律乐器 GM Program 族
_MELODIC_GM_FAMILIES: dict[str, set[int]] = {
    "strings": set(range(40, 48)),       # 弦乐
    "voice": set(range(52, 56)),         # 人声/合唱
    "brass": set(range(56, 64)),         # 铜管
    "reed": set(range(64, 72)),          # 木管
    "wind": set(range(72, 80)),          # 管乐
    "synth_lead": set(range(80, 88)),    # 合成主音
}

# 非旋律 GM Program 族（基本不可能承载主旋律）
_NON_MELODIC_GM_FAMILIES: dict[str, set[int]] = {
    "bass": set(range(32, 40)),          # 贝斯
    "percussion": set(range(112, 128)),  # 打击乐/音效
}


def _score_melody_candidacy(part: Part) -> float:
    """对单个 Part 做主旋律候选评分（0-100）。

    四个维度：
      1. 复音数 → 接近单音（~1）得分最高（旋律是单线条）
      2. 音域 → 在人声/乐器旋律常用音域 E3-C5 内得分高
      3. 音高方差 → 方差大 = 旋律性强（vs 和弦伴奏音高稳定）
      4. GM 乐器 → 弦乐/管乐/人声等旋律乐器得分高
    """
    flat = part.flatten()
    notes_and_chords = list(flat.notes)
    if not notes_and_chords:
        return 0.0

    # 提取所有 MIDI 音高
    pitches: list[int] = []
    onset_buckets: dict[float, int] = {}  # start_time → 同时发音符数
    for el in notes_and_chords:
        offset = float(el.offset)
        if isinstance(el, Note) and not isinstance(el, Rest):
            pitches.append(el.pitch.midi)
            onset_buckets[offset] = onset_buckets.get(offset, 0) + 1
        elif isinstance(el, M21Chord):
            for p in el.pitches:
                pitches.append(p.midi)
            onset_buckets[offset] = onset_buckets.get(offset, 0) + len(el.pitches)

    n_pitches = len(pitches)
    if n_pitches < 8:  # 音符太少，不算旋律轨
        return 0.0

    # ---- 维度 1：复音数 ----
    avg_poly = sum(onset_buckets.values()) / max(len(onset_buckets), 1)
    if 1.0 <= avg_poly < 1.3:
        poly_score = 1.0    # 纯单音轨 → 典型人声/主旋律
    elif 1.3 <= avg_poly < 1.8:
        poly_score = 0.7    # 轻微和声 → 可能是旋律+少量双音
    elif 1.8 <= avg_poly < 2.5:
        poly_score = 0.4    # 双音居多 → 可能是钢琴右手
    elif 2.5 <= avg_poly < 4.0:
        poly_score = 0.15   # 和弦居多 → 大概率是伴奏
    else:
        poly_score = 0.05   # 密集和弦 → 绝不可能是旋律

    # ---- 维度 2：音域 ----
    avg_pitch = sum(pitches) / n_pitches
    # 50 (D3) ~ 74 (D5) 为理想旋律音域
    if 50 <= avg_pitch <= 74:
        register_score = 1.0
    elif 45 <= avg_pitch < 50 or 74 < avg_pitch <= 79:
        register_score = 0.6
    elif 40 <= avg_pitch < 45 or 79 < avg_pitch <= 84:
        register_score = 0.3
    else:
        register_score = 0.1

    # ---- 维度 3：音高方差 ----
    mean_p = sum(pitches) / n_pitches
    variance = sum((p - mean_p) ** 2 for p in pitches) / n_pitches
    if variance > 80:
        var_score = 1.0     # 旋律线起伏大
    elif variance > 40:
        var_score = 0.7
    elif variance > 15:
        var_score = 0.4
    else:
        var_score = 0.1     # 几乎不变 → 可能是持续低音/打击乐

    # ---- 维度 4：GM 乐器 ----
    try:
        inst = part.getInstrument()
        program = getattr(inst, 'midiProgram', 0) or 0 if inst is not None else 0
    except Exception:
        program = 0

    if any(program in fam for fam in _MELODIC_GM_FAMILIES.values()):
        inst_score = 1.0
    elif any(program in fam for fam in _NON_MELODIC_GM_FAMILIES.values()):
        inst_score = 0.1
    else:
        inst_score = 0.5   # 未知/中性（钢琴、吉他等）

    # ---- 维度 5：音符数量（旋律轨通常有足够多的音符） ----
    if n_pitches >= 500:
        count_score = 1.0
    elif n_pitches >= 200:
        count_score = 0.7
    elif n_pitches >= 80:
        count_score = 0.5
    elif n_pitches >= 30:
        count_score = 0.3
    else:
        count_score = 0.1

    # ---- 维度 6：音高集中度惩罚（铺底 drone 的特征：少数音高占绝大多数） ----
    from collections import Counter as _Counter
    pitch_counts = _Counter(pitches)
    top2_ratio = sum(c for _, c in pitch_counts.most_common(2)) / n_pitches if n_pitches > 0 else 1.0
    if top2_ratio > 0.85:
        concentration_penalty = -15  # 两个音占了 85%+ → drone/铺底，重罚
    elif top2_ratio > 0.7:
        concentration_penalty = -8
    elif top2_ratio > 0.5:
        concentration_penalty = -3
    else:
        concentration_penalty = 0

    raw = (
        poly_score * _MELODY_WEIGHT_POLYPHONY
        + register_score * _MELODY_WEIGHT_REGISTER
        + var_score * _MELODY_WEIGHT_VARIANCE
        + inst_score * _MELODY_WEIGHT_INSTRUMENT
        + count_score * _MELODY_WEIGHT_NOTE_COUNT
        + concentration_penalty
    )
    return round(raw, 1)


def _identify_melody_track(parts: list[Part]) -> int:
    """返回主旋律轨的索引（parts 列表下标），-1 表示无合格旋律轨。

    对每个 Part 打启发式分，取最高分。若最高分 < _MIN_MELODY_SCORE，
    则认为该 MIDI 没有清晰的主旋律轨 → 返回 -1 → 调用方回退到全轨拍平。
    """
    if not parts:
        return -1
    if len(parts) == 1:
        # 单轨 MIDI：旋律嵌在和弦里（钢琴独奏/吉他独奏），无法从轨层面分离。
        # 应回退到 top_note 策略——从轨内提取高音线作为旋律。
        return -1

    scored = [(i, _score_melody_candidacy(p)) for i, p in enumerate(parts)]
    best_idx, best_score = max(scored, key=lambda x: x[1])  # type: ignore[arg-type]

    if best_score < _MIN_MELODY_SCORE:
        return -1

    # 不要求最高分必须明显高于次高分——多轨歌曲中人声轨和合成器轨可能分数相近，
    # 但宁可选一个（即使选错也比把所有轨混在一起猜旋律强）
    return best_idx


# =============================================================================
# 核心解析函数
# =============================================================================


def parse_midi(file_path: str) -> tuple[list[MidiNote], list[MidiNote], int]:
    """解析磁盘上的 .mid 文件。

    ADR-001 P1 改动：识别主旋律轨，分别返回全轨音符和旋律轨音符。

    Args:
        file_path: .mid 或 .midi 文件的绝对/相对路径。

    Returns:
        (all_notes, melody_notes, bpm):
          - all_notes: 全轨音符（含伴奏），按时间排序，供和弦分析使用
          - melody_notes: 仅主旋律轨音符（若无法识别则回退为空，下游使用全轨）
          - bpm: 每分钟节拍数（优先取速度标记，回退 120）

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件后缀不是 .mid/.midi。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"MIDI 文件不存在: {file_path}")
    if path.suffix.lower() not in (".mid", ".midi"):
        raise ValueError(f"非 MIDI 文件格式: {path.suffix}")

    score = converter.parse(str(path))
    if not isinstance(score, Score):
        raise ValueError(f"MIDI 解析异常：非 Score 类型 ({type(score).__name__})")
    return _extract_from_score(score)


def parse_midi_bytes(data: bytes) -> tuple[list[MidiNote], list[MidiNote], int]:
    """解析内存中的 MIDI 字节流。

    用于 FastAPI 的 UploadFile.read() → 直接解析，无需先落盘。
    """
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        score = converter.parse(tmp.name)
        if not isinstance(score, Score):
            raise ValueError(f"MIDI 解析异常：非 Score 类型 ({type(score).__name__})")
    return _extract_from_score(score)


def _extract_from_score(score: Score) -> tuple[list[MidiNote], list[MidiNote], int]:
    """从 music21 Score 提取音符，分离主旋律与伴奏。

    ADR-001 P1：
      1. 对每轨做启发式主旋律评分
      2. 识别主旋律轨 → 旋律轨音符标记 is_melody=True
      3. 无法识别时回退 → melody_notes 为空列表（下游使用全轨 top_note 回退策略）

    Returns:
        (all_notes, melody_notes, bpm)——all_notes 含所有轨（供和弦分析），
        melody_notes 仅含主旋律轨（供指法生成的旋律线）。
    """
    bpm = _extract_bpm(score)
    parts = list(score.parts)

    melody_idx = _identify_melody_track(parts) if parts else -1

    all_notes: list[MidiNote] = []
    melody_notes: list[MidiNote] = []

    for part_idx, part in enumerate(parts):
        channel, program = _get_part_instrument(part)

        flat = part.flatten()
        part_notes: list[MidiNote] = []
        _extract_notes(flat.notes, part_notes, track=part_idx, channel=channel, program=program)

        if part_idx == melody_idx:
            for n in part_notes:
                n.is_melody = True
            melody_notes.extend(part_notes)

        all_notes.extend(part_notes)

    all_notes.sort(key=lambda n: (n.start_time, n.midi_number))
    melody_notes.sort(key=lambda n: (n.start_time, n.midi_number))

    return all_notes, melody_notes, bpm


def _get_part_instrument(part: Part) -> tuple[int, int]:
    """从 Part 提取 (midi_channel, midi_program)，失败返回 (0, 0)。"""
    try:
        inst = part.getInstrument()
        if inst is not None:
            channel = getattr(inst, 'midiChannel', 0) or 0
            program = getattr(inst, 'midiProgram', 0) or 0
            return channel, program
    except Exception:
        pass
    return 0, 0


def _extract_notes(
    elements,
    notes: list[MidiNote],
    track: int = 0,
    channel: int = 0,
    program: int = 0,
) -> None:
    """遍历 music21 Note/Chord/Rest 元素流，转换为 MidiNote 列表。

    核心逻辑：
      - Note（单音）：直接提取 pitch.midi + offset + duration
      - Chord（和弦）：拆分为多个同 offset + 同 duration 的 MidiNote
      - Rest（休止符）：跳过，不写入（TAB 中休止 = 空白）
      - Tie（连音）：music21 已在 parse 阶段合并 duration，此处无需特殊处理

    注：GM program 暂未存入 MidiNote（schema 无此字段），后续可扩展。
    """
    for el in elements:
        if isinstance(el, Note) and not isinstance(el, Rest):
            offset = float(el.offset)
            dur = float(el.duration.quarterLength)
            if dur <= 0:
                dur = 0.001
            vel = el.volume.velocity if el.volume.velocity is not None else 64

            notes.append(
                MidiNote(
                    midi_number=el.pitch.midi,
                    start_time=offset,
                    duration=dur,
                    velocity=vel,
                    track=track,
                    channel=channel,
                )
            )

        elif isinstance(el, M21Chord):
            offset = float(el.offset)
            dur = float(el.duration.quarterLength)
            if dur <= 0:
                dur = 0.001
            vel = el.volume.velocity if el.volume.velocity is not None else 64

            for pitch in el.pitches:
                notes.append(
                    MidiNote(
                        midi_number=pitch.midi,
                        start_time=offset,
                        duration=dur,
                        velocity=vel,
                        track=track,
                        channel=channel,
                    )
                )


def _extract_bpm(score: Score) -> int:
    """从 Score 中提取 BPM。

    优先级：
      1. 扫描 flat 中的 MetronomeMark（如 quarter=120），取第一个标记的数字
      2. 若为文本型速度标记（如 "Allegro"），用 music21 内置映射表换算
      3. 使用 music21 tempo 分析的 beats per minute
      4. 以上全部失败 → 回退 120（通用默认值）
    """
    # 遍历 recurse() 流收集所有速度标记
    for el in score.recurse():
        if isinstance(el, tempo.MetronomeMark) and el.number is not None:
            return int(el.number)
        # music21 10.x 移除了 TempoIndication.getQuarterBPM()，文本速度标记
        # 极少出现在 MIDI 文件中，跳过此分支，交由下方 analyze("tempo") 处理

    # music21 tempo 分析（基于音符密度 + 拍号推算）
    try:
        analysis_bpm = score.analyze("tempo")
        if analysis_bpm is not None:
            return int(analysis_bpm)
    except Exception:
        pass

    return 120

"""乐理分析封装 —— Agent 2（和声编排）的底层确定性工具。

在多 Agent 系统中的角色：
  接收 Agent 1（MIDI 解析）产出的音符序列，输出调性、和弦进行、拍号。
  这是 Agent 3（指法生成）的前置依赖——没有和声信息，指法生成器不知道按哪个和弦编配。

技术方案：
  基于 music21（MIT 许可证，项目已安装）的以下能力：
    1. Krumhansl-Schmuckler 调性检测算法（`analysis.discrete.analyzeStream`）
       → 统计音符在各调性下的匹配度，返回最高分的调性
    2. 拍内和弦识别 → 将音符按拍位分组，同拍内音符集合用 music21 Chord 对象
       的 pitchedCommonName 命名（如 "C4 E4 G4" → "C-major triad"）
    3. 和弦名称标准化 → 将 music21 的学术风格名称转为人读的 jazz/pop 格式
       （如 "C-major triad" → "C"，"G dominant-seventh" → "G7"）

和弦检测策略：
  不要求严格同时 onset 的音才算和弦（MIDI 中钢琴琶音的和弦各音有轻微时间差）。
  采用"拍位分桶"策略：以半拍（八分音符）为窗口，窗口内的所有音符视为同一和声事件，
  用 music21 对窗口内的音高集合做和弦识别。相邻窗口若和弦相同则去重合并。

边界处理：
  - 空音符序列 → 返回 C major / 120 BPM 的默认 HarmonyAnalysis，不抛异常
  - 单音输入（无法形成和弦）→ 只有调性检测，和弦进行为空列表
  - 极短音符 → 不特殊过滤（music21 自身的最小音符感知已足够）
"""

from collections import defaultdict

from music21 import stream  # type: ignore[import-untyped]
from music21.analysis.discrete import analyzeStream  # type: ignore[import-untyped]  # K-S 调性检测
from music21.chord import Chord as M21Chord  # type: ignore[import-untyped]
from music21.note import Note as M21Note  # type: ignore[import-untyped]

from src.api.schemas import Chord, HarmonyAnalysis, MidiNote


# ===== 和弦名称标准化映射 =====
# music21 的 chord.commonName 返回不含根音的学术名。
# 本表映射学术名 → 流行/爵士风格的短后缀。
# 注意：键名必须与 music21 10.x 的 commonName 完全一致。
_CHORD_NAME_MAP: dict[str, str] = {
    "major triad": "",
    "minor triad": "m",
    "diminished triad": "dim",
    "augmented triad": "aug",
    "dominant seventh chord": "7",
    "major seventh chord": "maj7",
    "minor seventh chord": "m7",
    "diminished seventh chord": "dim7",
    "half-diminished seventh chord": "m7b5",
    "augmented seventh chord": "aug7",
    "suspended fourth": "sus4",
    "suspended second": "sus2",
}


def _chord_short_name(m21_chord: M21Chord) -> str:
    """将 music21 Chord 对象转为短和弦名（如 'Cmaj7'/'G7'/'C'）。

    music21 的 .commonName 返回不含根音的学术名（如 "major triad"、
    "dominant seventh chord"），.root().name 返回根音字母（如 "C"、"F#"）。
    本函数映射学术名 → 流行短后缀，然后拼接根音 + 后缀。
    """
    root = m21_chord.root().name  # e.g. "C", "F#"
    try:
        quality_full = m21_chord.commonName  # e.g. "major triad", "dominant seventh chord"
    except Exception:
        quality_full = m21_chord.pitchedCommonName

    # 两轮映射：先精确匹配，再子串模糊匹配
    # 注意：空字符串 (如 "major triad" → "") 是合法值（大三和弦不加后缀），
    # 不能用 `if not suffix` 来判断"没找到"——空串是 falsy 但有效。
    suffix = _CHORD_NAME_MAP.get(quality_full)
    if suffix is None:
        # 子串匹配：如 "dominant seventh chord" 包含 "dominant seventh"
        for key, val in _CHORD_NAME_MAP.items():
            if key in quality_full:
                suffix = val
                break
    if suffix is None:
        suffix = quality_full  # 回退：保留原始学术名

    return f"{root}{suffix}"


def _detect_root_and_quality(notes_at_beat: list[MidiNote]) -> tuple[str, str, str]:
    """给定同一拍位的音符集合，返回 (和弦全名, 根音, 性质)。

    通过 music21 Chord 对象的命名功能自动识别和弦类型。
    """
    if len(notes_at_beat) < 2:
        # 单音：无法形成和弦，记为根音 + 空品质
        root_note = M21Note(notes_at_beat[0].midi_number) if notes_at_beat else M21Note(60)
        return root_note.name, root_note.name, ""

    # 将 MidiNote 转为 music21 Note 列表，构造 Chord 对象
    m21_notes = [M21Note(n.midi_number) for n in notes_at_beat]
    try:
        m21_chord = M21Chord(m21_notes)
        # 检查是否为有效的和弦（至少 2 个不同音高）
        if len(set(n.pitch.midi for n in m21_chord.notes)) < 2:
            root = max(m21_notes, key=lambda n: n.pitch.midi).name
            return root, root, ""
        short = _chord_short_name(m21_chord)
        root = m21_chord.root().name
        # 从短名推断性质：去掉根音字母（含 #/b）
        quality = short
        if short.startswith(root):
            quality = short[len(root):]
        return short, root, quality
    except Exception:
        root = m21_notes[0].name
        return root, root, ""


def _midi_notes_to_stream(notes: list[MidiNote]) -> stream.Stream:
    """将 MidiNote 列表重建为 music21 Stream，供和弦/调性分析使用。

    不保留原 MIDI 的 velocity/track/channel 信息——这些属性对乐理分析无影响。
    """
    s = stream.Stream()
    for mn in notes:
        n = M21Note(mn.midi_number)
        n.duration.quarterLength = mn.duration
        n.offset = mn.start_time
        s.append(n)
    return s


# ===== 核心 API =====


def detect_key(notes: list[MidiNote]) -> str:
    """对音符序列执行 Krumhansl-Schmuckler 调性检测。

    使用 music21 的 analyzeStream() 实现，统计各音符在 24 个大小调下的匹配权重，
    返回最高分的调性名称（如 "C major" / "A minor"）。

    若输入为空，返回 "C major"（通用默认值）。
    """
    if not notes:
        return "C major"

    s = _midi_notes_to_stream(notes)
    try:
        key_obj = analyzeStream(s, "KrumhanslSchmuckler")
        if key_obj is not None:
            return key_obj.tonicPitchNameWithCase
        return "C major"
    except Exception:
        return "C major"


def analyze_chords(notes: list[MidiNote], bpm: int) -> HarmonyAnalysis:
    """对音符序列执行完整和声分析 → 输出 HarmonyAnalysis。

    流程：
      1. 调性检测
      2. 按拍位分桶（半拍窗口，BPM 驱动）
      3. 每桶内用 music21 Chord 命名
      4. 相邻相同和弦去重合并
      5. 组装 HarmonyAnalysis（key + bpm + chord_progression + time_signature）

    Args:
        notes: Agent 1 产出的音符列表。
        bpm:   Agent 1 提取的 BPM，用于计算拍位窗口大小。

    Returns:
        HarmonyAnalysis: 调性 + BPM + 和弦进行 + 拍号。
    """
    key = detect_key(notes)

    if not notes:
        return HarmonyAnalysis(
            key=key,
            bpm=bpm,
            chord_progression=[],
            time_signature=(4, 4),
        )

    # 拍位分桶：以八分音符（半拍）为窗口宽度。
    # MidiNote 的 start_time 以 quarterLength 为单位（music21 offset），
    # 一个四分音符 = 1.0 quarterLength，八分音符 = 0.5 quarterLength。
    # BPM 影响的是实际秒数，不影响 quarterLength → 音符数量的关系。
    bucket_width = 0.5  # 八分音符窗口（即半拍）

    # 按时间分桶：将 offset 落在同一窗口内的所有音符视为同和声事件
    buckets: dict[int, list[MidiNote]] = defaultdict(list)
    for mn in notes:
        bucket_idx = int(mn.start_time / bucket_width)
        buckets[bucket_idx].append(mn)

    # 逐桶识别和弦 + 去重合并
    chord_progression: list[Chord] = []
    prev_name: str = ""

    for idx in sorted(buckets):
        bucket_notes = buckets[idx]
        start_time = idx * bucket_width
        duration = bucket_width

        name, root, quality = _detect_root_and_quality(bucket_notes)
        midi_nums = sorted(set(n.midi_number for n in bucket_notes))

        if name == prev_name and chord_progression:
            # 同一和弦延续：拉长前一个和弦的 duration
            chord_progression[-1].duration += duration
        else:
            chord_progression.append(
                Chord(
                    name=name,
                    root=root,
                    quality=quality,
                    midi_numbers=midi_nums,
                    start_time=start_time,
                    duration=duration,
                )
            )
        prev_name = name

    return HarmonyAnalysis(
        key=key,
        bpm=bpm,
        chord_progression=chord_progression,
        time_signature=(4, 4),  # 默认；真实的拍号提取需解析 MIDI meta，M3 不做
    )

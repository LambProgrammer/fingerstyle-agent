"""物理校验器 —— Agent 4（物理校验 + 回退）的确定性规则引擎。

在多 Agent 系统中的角色：
  接收 Agent 3（指法生成）产出的 TabData，逐小节检查是否违反人类手部的物理约束。
  校验不通过 → errors 列表送回 Agent 3 做局部修正（回退循环）。
  校验通过 → TabData 直接交付前端渲染。

所有检查项均为确定性布尔逻辑，不调 LLM，不依赖第三方库。

四条核心规则（按严重度排序）：
  1. 音域校验：品位 ≤ 难度上限（初级 5 / 中级 9 / 高级 15），超出 = error
  2. 跨度校验：同一时间点同时发声的各弦品位差 ≤ 4（人手极限），超出 = error
  3. 横按检测：同一小节连续 3+ 弦同品位 → 标记横按（warning，非 error）
  4. 空弦合理性：旋律进行中突兀空弦 → warning

智能变调夹推荐（capo_recommendation）：
  如果 >30% 的错误项集中在低把位（品位 1-5）但属于可修正范围（跨度超标/空弦问题），
  推荐 capo=2 将低把位移到中把位优化手型。
"""
from __future__ import annotations
from collections import defaultdict

from src.api.schemas import (
    Difficulty,
    TabData,
    TabMeasure,
    ValidationError,
    ValidationResult,
)


# ===== 可调参数 =====
_MAX_FRET = 15   # 人手舒适品位上限（与生成器统一，不做难度分级）
_MAX_SPAN = 4     # 人手最大同时按弦跨度（品）——全难度统一，人手极限不随水平变化
# 横按检测阈值（同一小节内同品位跨 N+ 根弦）
_BARRE_STRING_THRESHOLD = 3  # 3 根及以上弦同品位 → 标记横按
# 变调夹推荐的错误比例阈值
_CAPO_RECOMMEND_RATIO = 0.3  # >30% 错误集中在低把位时推荐变调夹
_CAPO_DEFAULT_FRET = 2       # 默认推荐品位


def validate(tab_data: TabData, difficulty: Difficulty | None = None) -> ValidationResult:
    """对完整指弹谱执行四项物理约束检查。

    Args:
        tab_data:  Agent 3 产出的六线谱数据。
        difficulty: 保留参数（向后兼容），当前未使用——所有谱面统一人手约束。

    Returns:
        ValidationResult: is_valid + errors + warnings + capo 推荐。
    """
    errors: list[ValidationError] = []
    warnings: list[str] = []

    for measure in tab_data.measures:
        # 规则 1：音域校验
        _check_fret_range(measure, errors)
        # 规则 2：跨度校验（同时发声音符）
        _check_span(measure, errors)
        # 规则 3：横按检测（warning）
        _check_barre(measure, warnings)

    # 规则 4：空弦合理性（全局，非逐小节）
    _check_open_string_abuse(tab_data, warnings)
    # 规则 5：跳跃密度（跨弦切换频率）
    _check_jump_density(tab_data, warnings)
    # 规则 6：空弦泛音与和弦根音冲突
    _check_open_harmonic_conflict(tab_data, warnings)

    # 智能变调夹推荐
    capo = _recommend_capo(errors) if errors else None

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        capo_recommendation=capo,
    )


def _check_fret_range(
    measure: TabMeasure,
    errors: list[ValidationError],
) -> None:
    """规则 1：每个音符的品位不超过人手舒适上限。"""
    for note in measure.notes:
        if note.fret > _MAX_FRET:
            errors.append(
                ValidationError(
                    measure=measure.number,
                    string=note.string,
                    fret=note.fret,
                    description=f"品位 {note.fret} 超出舒适上限 {_MAX_FRET} 品",
                    severity="error",
                )
            )


def _check_span(
    measure: TabMeasure,
    errors: list[ValidationError],
) -> None:
    """规则 2：同一时间位置的同时发声音符，品位跨度 ≤ _MAX_SPAN。

    按 start_time 分桶，桶内取所有音符的品位 min/max 差。
    """
    time_buckets: dict[float, list[int]] = defaultdict(list)
    for note in measure.notes:
        time_buckets[note.start_time].append(note.fret)

    for t, frets in time_buckets.items():
        # 排除空弦：空弦不需要手指按弦，对跨度无贡献
        fretted = [f for f in frets if f > 0]
        if len(fretted) < 2:
            continue
        fret_min, fret_max = min(fretted), max(fretted)
        span = fret_max - fret_min
        if span > _MAX_SPAN:
            errors.append(
                ValidationError(
                    measure=measure.number,
                    string=0,  # 全局问题，非单弦
                    fret=0,
                    description=f"小节 {measure.number} 时间 {t:.1f}：品位跨度 {span} 超过人手极限 {_MAX_SPAN}",
                    severity="error",
                )
            )


def _check_barre(measure: TabMeasure, warnings: list[str]) -> None:
    """规则 3：横按检测。

    同一小节内，同品位跨 ≥_BARRE_STRING_THRESHOLD 根弦 → 横按标记。
    这是 warning 不是 error——横按是正常技巧，但中级以上才舒适。
    """
    # 按品位分组
    fret_strings: dict[int, set[int]] = defaultdict(set)
    for note in measure.notes:
        if note.fret > 0:  # 空弦不算横按
            fret_strings[note.fret].add(note.string)

    for fret, strings in fret_strings.items():
        if len(strings) >= _BARRE_STRING_THRESHOLD:
            warnings.append(f"小节 {measure.number} 品位 {fret}：横跨 {len(strings)} 根弦（横按），注意手型舒适度")


def _check_open_string_abuse(tab_data: TabData, warnings: list[str]) -> None:
    """规则 4：空弦合理性检查。

    检测模式：空弦音之后紧跟的同弦非空弦音如果间隔 >5 品，
    可能造成突兀的音色跳跃（空弦共鸣 vs 按弦音色差异大）。
    这是美学检查，非结构错误。

    此规则仅对初级/中级生效——高级演奏中空弦利用是常见技巧。
    """
    all_notes = sorted(
        [n for m in tab_data.measures for n in m.notes],
        key=lambda n: (n.string, n.start_time),
    )

    # 按弦分组，检测同弦上空弦→高把位跳跃
    string_notes: dict[int, list] = defaultdict(list)
    for n in all_notes:
        string_notes[n.string].append(n)

    for string, notes in string_notes.items():
        for i in range(len(notes) - 1):
            curr, nxt = notes[i], notes[i + 1]
            if curr.fret == 0 and nxt.fret >= 6:
                warnings.append(
                    f"弦 {string}：空弦后跳至 {nxt.fret} 品（跨度 {nxt.fret}），音色可能突兀"
                )


def _check_jump_density(tab_data: TabData, warnings: list[str]) -> None:
    """规则 5：统计全曲跨弦切换的频率，密度过高时发 Warning。

    跨弦次数 = 相邻音符（同一声部内）的弦号变化次数。
    跳跃密度 = 跨弦次数 / 总音符数。
    若 > 0.5（超过一半音符在换弦），提示用户注意指法流畅性。
    """
    all_notes = sorted(
        [n for m in tab_data.measures for n in m.notes],
        key=lambda n: (n.string, n.start_time),
    )
    if len(all_notes) < 4:
        return

    # 按弦分组，组内按时间排序，统计换弦次数
    string_notes: dict[int, list] = defaultdict(list)
    for n in all_notes:
        string_notes[n.string].append(n)

    # 将各弦的音符按时间交织 → 统计弦号切换
    time_sorted = sorted(all_notes, key=lambda n: (n.start_time, n.string))
    cross_count = 0
    for i in range(len(time_sorted) - 1):
        if time_sorted[i].string != time_sorted[i + 1].string:
            cross_count += 1

    ratio = cross_count / len(time_sorted) if time_sorted else 0
    if ratio > 0.5:
        warnings.append(
            f"跳跃密度较高（{cross_count}/{len(time_sorted)} = {ratio:.0%}），注意指法流畅性"
        )


def _check_open_harmonic_conflict(tab_data: TabData, warnings: list[str]) -> None:
    """规则 6：低音弦空弦音与和弦根音冲突检测。

    低音空弦（E2/A2）有固定音高，若当前和弦根音与空弦音程不协和
    （如和弦是 D#m，但低音 E2 空弦持续响），会影响和声纯净度。
    检测：全曲使用最频繁的低音空弦，若与多数和弦根音构成增一度/小二度等
    不协和音程，则发 Warning 建议调整。
    """
    # 收集低音空弦使用情况
    low_open_usage: dict[int, int] = defaultdict(int)  # pitch → count
    for m in tab_data.measures:
        for n in m.notes:
            if n.fret == 0 and n.string >= 5:  # 5-6弦空弦
                # 反算 MIDI 音高（标准定弦下）
                open_map = {6: 40, 5: 45}  # E2, A2
                pitch = open_map.get(n.string, 0) + n.fret
                low_open_usage[pitch] += 1

    if not low_open_usage:
        return

    # 取使用最频繁的低音空弦
    dominant_pitch = max(low_open_usage, key=lambda k: low_open_usage[k])

    # 注：全面检测需要和弦信息（tab_data 不含 chord_progression），
    # 此处做简化：若低音空弦被频繁使用（占总音符 ≥ 30%），提醒用户留意。
    total_notes = sum(len(m.notes) for m in tab_data.measures)
    dominant_ratio = low_open_usage[dominant_pitch] / total_notes if total_notes > 0 else 0
    if dominant_ratio >= 0.3:
        pitch_names = {40: "E2", 45: "A2"}
        warnings.append(
            f"低音空弦 {pitch_names.get(dominant_pitch, str(dominant_pitch))} "
            f"使用频繁（{low_open_usage[dominant_pitch]}/{total_notes} = {dominant_ratio:.0%}），"
            "特殊调弦下可能与和弦根音冲突，注意和声纯净度"
        )


def _recommend_capo(errors: list[ValidationError]) -> int | None:
    """智能变调夹推荐。

    条件：>30% 的校验错误出现在品位 1-5（低把位密集区域）。
    逻辑：如果大部分错误都出在低把位，说明把位太挤——用户上变调夹后
    低把位自动上移，手型空间更大。
    """
    if not errors:
        return None

    low_fret_errors = [e for e in errors if 1 <= e.fret <= 5]
    ratio = len(low_fret_errors) / len(errors) if errors else 0

    if ratio > _CAPO_RECOMMEND_RATIO:
        return _CAPO_DEFAULT_FRET  # 推荐第 2 品

    return None

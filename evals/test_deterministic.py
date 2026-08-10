"""层 1：确定性指标评估（pytest 形式，无 LLM 成本）。

指标：
  ① 物理校验通过率 — 生成 TAB 经 tab_validator 校验的通过比例
  ② 旋律保真率   — TAB 弦+品反算音高 vs 原 MIDI 旋律音高的覆盖率
  ③ 声部 zone 合规率 — 旋律→1-3弦、低音→4-6弦、内声部→3-4弦 的分配准确率

运行：uv run pytest evals/test_deterministic.py -v
"""

from pathlib import Path

import pytest

from src.api.schemas import (
    MidiNote,
    Style,
    TabData,
    TabGenerationConfig,
)
from src.tools.midi_parser import parse_midi
from src.tools.music21_wrapper import analyze_chords
from src.tools.tab_generator import generate_tab
from src.tools.tab_validator import validate


# =============================================================================
# 辅助函数
# =============================================================================


def _reverse_pitch(string: int, fret: int, tuning: list[str] | None = None) -> int:
    """根据弦+品反算 MIDI 音高（用于旋律保真率计算）。"""
    import music21.note as m21note  # type: ignore[import-untyped]
    if tuning is None:
        tuning = ["E2", "A2", "D3", "G3", "B3", "E4"]
    open_pitches = {6 - i: m21note.Note(name).pitch.midi for i, name in enumerate(tuning)}
    return open_pitches.get(string, 0) + fret


def _midi_notes_at_time(notes: list[MidiNote], t: float, window: float = 0.1) -> list[MidiNote]:
    """查找给定时间窗口内的 MIDI 音符。"""
    return [n for n in notes if abs(n.start_time - t) <= window]


def _tab_notes_at_time(tab_notes: list, t: float, window: float = 0.1) -> list:
    """查找给定时间窗口内的 TabNote。"""
    return [n for n in tab_notes if abs(n.start_time - t) <= window]


# =============================================================================
# 测试：遍历每个黄金 MIDI 跑管线 + 计算指标
# =============================================================================


def _run_pipeline(midi_path: str) -> tuple[TabData, list[MidiNote], list]:
    """跑完整 Agent 1→2→3 管线（覆盖 ADR-001 P1/P2），返回 (tab_data, midi_notes, all_tab_notes)。"""
    all_notes, melody_notes, bpm = parse_midi(midi_path)
    harmony = analyze_chords(all_notes, bpm)
    config = TabGenerationConfig(style=Style.JPOP)
    tab_data = generate_tab(all_notes, harmony, config, melody_notes=melody_notes)
    all_tab_notes = [n for m in tab_data.measures for n in m.notes]
    return tab_data, all_notes, all_tab_notes


@pytest.mark.parametrize("fixture_name,expected_errors_max", [
    ("c_scale_midi", 3),         # 单音音阶：少量 chord voicing span error
    ("c_triad_arpeggio_midi", 0),  # 单音琶音：应零错误
    ("melody_bass_midi", 5),      # 旋律+低音：两声部合奏触发 span 检查
    ("wide_jumps_midi", 5),       # 大跨度跳跃：允许少量span error
    ("dense_chord_midi", 10),     # 密集和弦：chord voicing未完成，允许较多span error
])
def test_validation_pass_rate(fixture_name, expected_errors_max, request):
    """① 物理校验通过率：每个黄金 MIDI 的校验错误数不超过预期上限。"""
    midi_path = request.getfixturevalue(fixture_name)
    tab_data, _, _ = _run_pipeline(midi_path)
    result = validate(tab_data)

    error_count = len(result.errors)
    print(f"\n  {fixture_name}: {error_count} errors (max {expected_errors_max})")
    if result.warnings:
        print(f"  warnings: {len(result.warnings)}")

    assert error_count <= expected_errors_max, (
        f"{fixture_name} 校验错误 {error_count} 超过上限 {expected_errors_max}"
    )


def test_overall_validation_pass_rate(all_golden_midi_paths):
    """① 综合物理校验通过率：所有黄金 MIDI 的平均 error 数。"""
    total_errors = 0
    total_measures = 0
    results = []

    for path in all_golden_midi_paths:
        tab_data, _, _ = _run_pipeline(path)
        result = validate(tab_data)
        errors = len(result.errors)
        measures = len(tab_data.measures)
        total_errors += errors
        total_measures += measures
        results.append((Path(path).name, errors, measures))

    avg_errors = total_errors / len(all_golden_midi_paths) if all_golden_midi_paths else 0
    print(f"\n  综合: {total_errors} errors / {len(all_golden_midi_paths)} files = {avg_errors:.1f} avg")
    for name, err, meas in results:
        print(f"    {name}: {err} err / {meas} meas")

    # 断言：平均错误数不超过每文件 3 个（考虑 chord voicing 未完成）
    assert avg_errors <= 3, f"平均校验错误 {avg_errors:.1f} 超过上限 3"


def test_melody_fidelity(all_golden_midi_paths):
    """② 旋律保真率：TAB 反算音高覆盖原 MIDI 旋律音高的比例。"""
    total_melody_notes = 0
    matched = 0

    for path in all_golden_midi_paths:
        all_notes, melody_notes, actual_bpm = parse_midi(path)
        bpm = actual_bpm or 100
        harmony = analyze_chords(all_notes, bpm)
        config = TabGenerationConfig(style=Style.JPOP)
        tab_data = generate_tab(all_notes, harmony, config, melody_notes=melody_notes)
        all_tab = [n for m in tab_data.measures for n in m.notes]
        midi_notes = all_notes  # 兼容后续变量名引用

        # 只统计可能被生成器用的音符（排除极短/极低 MIDI 音符）
        melody_candidates = [n for n in midi_notes if n.midi_number >= 55]
        total_melody_notes += len(melody_candidates)

        for mn in melody_candidates:
            tab_at_time = _tab_notes_at_time(all_tab, mn.start_time)
            for tn in tab_at_time:
                reversed_pitch = _reverse_pitch(tn.string, tn.fret, tab_data.tuning)
                if reversed_pitch == mn.midi_number:
                    matched += 1
                    break

    fidelity = (matched / total_melody_notes * 100) if total_melody_notes > 0 else 0
    print(f"\n  旋律保真率: {matched}/{total_melody_notes} = {fidelity:.1f}%")

    # 断言：旋律保真率应 ≥ 70%（受八度回退影响）
    assert fidelity >= 70, f"旋律保真率 {fidelity:.1f}% 低于门槛 70%"


def test_zone_compliance(all_golden_midi_paths):
    """③ 声部 zone 合规率——P2 升级：使用 TabNote.voice 字段精确判定。"""
    total_notes = 0
    violations = 0

    for path in all_golden_midi_paths:
        tab_data, _, all_tab_notes = _run_pipeline(path)

        for tn in all_tab_notes:
            total_notes += 1

            if tn.voice == "melody" and tn.string not in {1, 2, 3}:
                # 旋律误入 4-6 弦
                violations += 1
            elif tn.voice == "bass" and tn.string not in {4, 5, 6}:
                # 低音误入 1-3 弦
                violations += 1
            elif tn.voice == "inner" and tn.string not in {3, 4}:
                # 内声部越界
                violations += 1

    zone_rate = (1 - violations / total_notes) * 100 if total_notes > 0 else 100
    print(f"\n  zone 合规率: {total_notes - violations}/{total_notes} = {zone_rate:.1f}%")

    assert zone_rate >= 90, f"zone 合规率 {zone_rate:.1f}% 低于门槛 90%"

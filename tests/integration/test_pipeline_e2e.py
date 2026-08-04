"""集成测试：完整确定性管线（Agent 1→2→3→4），不含 LLM 调用。

使用 evals/datasets/golden_midi/ 中的测试数据，无需外部服务。
"""

from pathlib import Path

from src.api.schemas import Style, TabGenerationConfig
from src.tools.midi_parser import parse_midi
from src.tools.music21_wrapper import analyze_chords
from src.tools.tab_generator import generate_tab
from src.tools.tab_validator import validate

_GOLDEN_DIR = Path(__file__).parent.parent.parent / "evals" / "datasets" / "golden_midi"


class TestPipelineE2E:
    """Agent 1→2→3→4 完整管线，覆盖 ADR-001 P1 旋律轨识别。"""

    def test_c_scale_pipeline_passes_validation(self):
        """C 大调音阶：单音 → 校验应接近零错误。"""
        midi_path = str(_GOLDEN_DIR / "c_scale.mid")
        all_notes, melody_notes, bpm = parse_midi(midi_path)
        harmony = analyze_chords(all_notes, bpm)
        config = TabGenerationConfig(style=Style.JPOP)
        tab = generate_tab(all_notes, harmony, config, melody_notes=melody_notes)
        result = validate(tab)

        # 单音音阶不应该有 span 问题，允许少量 chord voicing 误判
        assert len(result.errors) <= 3, (
            f"单音音阶校验错误过多: {len(result.errors)} errors"
        )
        assert tab.tempo == bpm

    def test_c_triad_arpeggio_zero_errors(self):
        """琶音：单音不应触发 span 检查。"""
        midi_path = str(_GOLDEN_DIR / "c_triad_arpeggio.mid")
        all_notes, melody_notes, bpm = parse_midi(midi_path)
        harmony = analyze_chords(all_notes, bpm)
        config = TabGenerationConfig(style=Style.JPOP)
        tab = generate_tab(all_notes, harmony, config, melody_notes=melody_notes)
        result = validate(tab)

        assert len(result.errors) == 0, (
            f"琶音应零错误，实际 {len(result.errors)}: "
            f"{[e.description for e in result.errors]}"
        )

    def test_melody_bass_produces_two_voices(self):
        """旋律+低音两声部：应产出 melody 和 bass 两个声部的音符。"""
        midi_path = str(_GOLDEN_DIR / "melody_bass.mid")
        all_notes, melody_notes, bpm = parse_midi(midi_path)
        harmony = analyze_chords(all_notes, bpm)
        config = TabGenerationConfig(style=Style.JPOP)
        tab = generate_tab(all_notes, harmony, config, melody_notes=melody_notes)

        voices = {n.voice for m in tab.measures for n in m.notes}
        # P1+P2 后 melody_notes 传入 + inner 填充，应至少包含 melody 和 bass
        assert "melody" in voices, f"缺少旋律声部，voices: {voices}"
        assert "bass" in voices, f"缺少低音声部，voices: {voices}"

    def test_tabdata_measures_have_correct_timing(self):
        """小节编号和拍号正确。"""
        midi_path = str(_GOLDEN_DIR / "melody_bass.mid")
        all_notes, melody_notes, bpm = parse_midi(midi_path)
        harmony = analyze_chords(all_notes, bpm)
        config = TabGenerationConfig(style=Style.JPOP)
        tab = generate_tab(all_notes, harmony, config, melody_notes=melody_notes)

        # 小节编号从 1 开始递增
        numbers = [m.number for m in tab.measures]
        assert numbers == list(range(1, len(numbers) + 1)), f"小节编号不连续: {numbers}"
        # 每小节有拍号
        for m in tab.measures:
            assert len(m.time_signature) == 2

    def test_wide_jumps_generates_without_crash(self):
        """大跨度跳跃 MIDI：不崩溃，正常输出。"""
        midi_path = str(_GOLDEN_DIR / "wide_jumps.mid")
        all_notes, melody_notes, bpm = parse_midi(midi_path)
        harmony = analyze_chords(all_notes, bpm)
        config = TabGenerationConfig(style=Style.JPOP)
        tab = generate_tab(all_notes, harmony, config, melody_notes=melody_notes)
        result = validate(tab)

        # 大跨度可能触发 span error，允许最多 5 个
        assert len(result.errors) <= 5, (
            f"大跨度校验错误过多: {len(result.errors)}"
        )

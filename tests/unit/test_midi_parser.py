"""单元测试：midi_parser MIDI 解析 + 主旋律轨识别。

使用 evals/datasets/golden_midi/ 中的 5 个 MIDI 文件，无需联网。
"""

from pathlib import Path

from src.tools.midi_parser import parse_midi

_GOLDEN_DIR = Path(__file__).parent.parent.parent / "evals" / "datasets" / "golden_midi"


def _path(name: str) -> str:
    return str(_GOLDEN_DIR / name)


class TestParseMidi:
    """MIDI 解析：音符提取 + BPM。"""

    def test_c_scale_parses_notes(self):
        """C 大调音阶 MIDI 至少包含 8 个音符。"""
        all_notes, melody_notes, bpm = parse_midi(_path("c_scale.mid"))
        assert len(all_notes) >= 8, f"C 大调音阶应有 ≥8 个音符，实际 {len(all_notes)}"
        assert bpm > 0, f"BPM 应为正数: {bpm}"

    def test_c_triad_arpeggio_no_negative_times(self):
        """所有音符 start_time ≥ 0。"""
        all_notes, _, _ = parse_midi(_path("c_triad_arpeggio.mid"))
        assert all(n.start_time >= 0 for n in all_notes)

    def test_midi_notes_have_required_fields(self):
        """解析后的 MidiNote 包含必要字段。"""
        all_notes, _, _ = parse_midi(_path("c_scale.mid"))
        if all_notes:
            n = all_notes[0]
            assert 0 <= n.midi_number <= 127
            assert n.start_time >= 0
            assert n.duration > 0
            assert n.velocity > 0

    def test_dense_chord_parses_multiple_notes(self):
        """密集和弦 MIDI 包含多个音符。"""
        all_notes, _, _ = parse_midi(_path("dense_chord.mid"))
        assert len(all_notes) >= 3, f"密集和弦应有 ≥3 个音符: {len(all_notes)}"

    def test_all_golden_midis_parse(self):
        """5 个黄金 MIDI 全部可以正常解析。"""
        for name in ["c_scale.mid", "c_triad_arpeggio.mid", "melody_bass.mid",
                      "wide_jumps.mid", "dense_chord.mid"]:
            all_notes, melody_notes, bpm = parse_midi(_path(name))
            assert len(all_notes) > 0, f"{name} 解析出 0 个音符"
            assert bpm > 0, f"{name} BPM = {bpm}"

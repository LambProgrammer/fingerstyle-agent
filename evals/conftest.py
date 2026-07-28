"""Pytest fixtures —— 从 evals/datasets/golden_midi/ 加载黄金 MIDI 测试文件。

5 个短 MIDI（音阶/琶音/两声部/大跳/密集和弦），体积 ~600 字节，
可直接提交到 Git 仓库。无需动态生成。
"""

from pathlib import Path

import pytest

_GOLDEN_DIR = Path(__file__).parent / "datasets" / "golden_midi"


def _path(name: str) -> str:
    return str(_GOLDEN_DIR / name)


@pytest.fixture
def c_scale_midi() -> str:
    """C 大调上行音阶：C4→C5，单音无和弦。"""
    return _path("c_scale.mid")


@pytest.fixture
def c_triad_arpeggio_midi() -> str:
    """C 大三和弦琶音 (C4 E4 G4)，重复 2 遍。"""
    return _path("c_triad_arpeggio.mid")


@pytest.fixture
def melody_bass_midi() -> str:
    """旋律 + 低音两声部：高音 C5→D5→E5→F5，低音 C3↔G2。"""
    return _path("melody_bass.mid")


@pytest.fixture
def wide_jumps_midi() -> str:
    """大跨度跳跃：C4→C5→C3→C5，触发跨度校验。"""
    return _path("wide_jumps.mid")


@pytest.fixture
def dense_chord_midi() -> str:
    """密集和弦：Cmaj7 (C4 E4 G4 B4) 和 Dm7 同时发声。"""
    return _path("dense_chord.mid")


@pytest.fixture
def all_golden_midi_paths(request) -> list[str]:
    """收集所有黄金 MIDI 文件路径。"""
    return [
        request.getfixturevalue(name)
        for name in [
            "c_scale_midi", "c_triad_arpeggio_midi", "melody_bass_midi",
            "wide_jumps_midi", "dense_chord_midi",
        ]
    ]

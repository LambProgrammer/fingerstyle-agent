"""单元测试：tab_generator 确定性规则（zone filter / fretboard / candidates）。

不调 LLM，不依赖外部服务。所有函数均为纯 Python 规则。
"""

from src.tools.tab_generator import _build_fretboard, _find_candidates, _zone_filter


class TestZoneFilter:
    """声部分配硬约束——物理隔离规则。"""

    def test_melody_stays_on_high_strings(self):
        """旋律只能分配在 1-2 弦（fallback 3 弦）。"""
        candidates = [(1, 3), (2, 5), (4, 2), (5, 0)]
        result = _zone_filter(candidates, "melody")
        strings = {s for s, _ in result}
        assert strings <= {1, 2, 3}, f"旋律误入低音弦: {strings}"
        assert len(result) >= 1

    def test_bass_stays_on_low_strings(self):
        """低音只能分配在 4-6 弦。"""
        candidates = [(1, 3), (3, 0), (4, 2), (5, 5), (6, 0)]
        result = _zone_filter(candidates, "bass")
        strings = {s for s, _ in result}
        assert strings <= {4, 5, 6}, f"低音误入高音弦: {strings}"

    def test_inner_stays_on_middle_strings(self):
        """内声部只能分配在 3-4 弦。"""
        candidates = [(1, 3), (2, 5), (3, 0), (4, 2), (5, 5)]
        result = _zone_filter(candidates, "inner")
        strings = {s for s, _ in result}
        assert strings <= {3, 4}, f"内声部越界: {strings}"

    def test_melody_fallback_to_string_3(self):
        """1-2 弦无候选时，旋律降级到 3 弦。"""
        candidates = [(3, 5), (4, 2)]  # 只有 3-4 弦
        result = _zone_filter(candidates, "melody")
        strings = {s for s, _ in result}
        assert strings == {3}, f"旋律未降级到 3 弦: {strings}"


class TestFretboard:
    """指板矩阵构建。"""

    def test_standard_tuning_has_6_strings(self):
        """标准定弦 EADGBE 生成 6 根弦的矩阵。"""
        tuning = ["E2", "A2", "D3", "G3", "B3", "E4"]
        fb = _build_fretboard(tuning)
        strings_in_matrix = {s for candidates in fb.values() for s, _ in candidates}
        assert strings_in_matrix == {1, 2, 3, 4, 5, 6}

    def test_middle_c_has_multiple_positions(self):
        """中央 C (MIDI 60) 在标准定弦下至少有 3 个可弹位置。"""
        tuning = ["E2", "A2", "D3", "G3", "B3", "E4"]
        fb = _build_fretboard(tuning)
        candidates = fb.get(60, [])
        assert len(candidates) >= 3, f"MIDI 60 只有 {len(candidates)} 个候选位置"


class TestFindCandidates:
    """候选查找 + 品位上限。"""

    def test_respects_fret_limit(self):
        """候选不超过全局品位上限。"""
        tuning = ["E2", "A2", "D3", "G3", "B3", "E4"]
        fb = _build_fretboard(tuning)
        # MIDI 64 (E4) 在标准定弦下最低是弦 1 空弦(0品)
        candidates = _find_candidates(64, fb)
        frets = [f for _, f in candidates]
        assert all(f <= 15 for f in frets)

    def test_per_measure_override_fret_limit(self):
        """per-measure overrides 可降低品位上限。"""
        tuning = ["E2", "A2", "D3", "G3", "B3", "E4"]
        fb = _build_fretboard(tuning)
        overrides = {0: {"fret_limit": 5}}
        candidates = _find_candidates(64, fb, overrides=overrides, measure_idx=0)
        frets = [f for _, f in candidates]
        assert frets, "应该至少有一个候选"
        assert all(f <= 5 for f in frets), f"override fret_limit=5 但候选: {frets}"

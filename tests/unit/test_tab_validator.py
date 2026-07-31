"""单元测试：tab_validator 确定性校验规则（fret_range / span / barre）。

不调 LLM，不依赖外部服务。所有检查项为确定性布尔逻辑。
"""

from src.api.schemas import (
    TabMeasure,
    TabNote,
    Technique,
    ValidationError,
)
from src.tools.tab_validator import _check_fret_range, _check_span, _check_barre


def _make_note(string=1, fret=0, start_time=0.0, duration=1.0, voice=""):
    return TabNote(string=string, fret=fret, start_time=start_time, duration=duration,
                   technique=Technique.NONE, voice=voice)


class TestFretRange:
    """规则 1：品位不超过人手舒适上限（15 品）。"""

    def test_all_notes_in_range_passes(self):
        """所有音符在 0-15 品内，无误。"""
        measure = TabMeasure(number=1, notes=[
            _make_note(string=1, fret=3),
            _make_note(string=2, fret=12),
            _make_note(string=3, fret=0),
        ], time_signature=(4, 4))
        errors: list[ValidationError] = []
        _check_fret_range(measure, errors)
        assert len(errors) == 0

    def test_fret_exceeds_limit_errors(self):
        """品位 > 15 应报 error。"""
        measure = TabMeasure(number=1, notes=[
            _make_note(string=1, fret=17),
        ], time_signature=(4, 4))
        errors: list[ValidationError] = []
        _check_fret_range(measure, errors)
        assert len(errors) == 1
        assert "17" in errors[0].description
        assert errors[0].severity == "error"


class TestSpan:
    """规则 2：同时发声音符的品位跨度 ≤ 4 品（人手极限）。"""

    def test_small_span_passes(self):
        """同时间点的品位差 ≤ 4，无误。"""
        measure = TabMeasure(number=1, notes=[
            _make_note(string=1, fret=3, start_time=0.0),
            _make_note(string=2, fret=5, start_time=0.0),  # span = 2
            _make_note(string=3, fret=4, start_time=0.0),
        ], time_signature=(4, 4))
        errors: list[ValidationError] = []
        _check_span(measure, errors)
        assert len(errors) == 0

    def test_large_span_errors(self):
        """同时间点的品位差 > 4，应报 error。"""
        measure = TabMeasure(number=1, notes=[
            _make_note(string=1, fret=1, start_time=0.0),
            _make_note(string=4, fret=10, start_time=0.0),  # span = 9
        ], time_signature=(4, 4))
        errors: list[ValidationError] = []
        _check_span(measure, errors)
        assert len(errors) == 1
        assert "跨度" in errors[0].description

    def test_open_strings_excluded_from_span(self):
        """空弦不算入跨度：空弦 + 高品位不应误报。"""
        measure = TabMeasure(number=1, notes=[
            _make_note(string=1, fret=0, start_time=0.0),   # 空弦
            _make_note(string=2, fret=10, start_time=0.0),  # 高品
        ], time_signature=(4, 4))
        errors: list[ValidationError] = []
        _check_span(measure, errors)
        # 空弦不计入 fretted span，只有 1 个 fretted 音符，不够 2 个不触发
        assert len(errors) == 0

    def test_different_times_no_conflict(self):
        """不同时间点的音符不触发 span 检查。"""
        measure = TabMeasure(number=1, notes=[
            _make_note(string=1, fret=1, start_time=0.0),
            _make_note(string=2, fret=10, start_time=2.0),  # 不同时间
        ], time_signature=(4, 4))
        errors: list[ValidationError] = []
        _check_span(measure, errors)
        assert len(errors) == 0


class TestBarre:
    """规则 3：横按检测（warning，非 error）。"""

    def test_three_strings_same_fret_warns(self):
        """3 根弦同品位 → 横按 warning。"""
        measure = TabMeasure(number=1, notes=[
            _make_note(string=1, fret=5),
            _make_note(string=2, fret=5),
            _make_note(string=3, fret=5),
        ], time_signature=(4, 4))
        warnings: list[str] = []
        _check_barre(measure, warnings)
        assert len(warnings) == 1
        assert "横按" in warnings[0]

    def test_two_strings_same_fret_no_warning(self):
        """2 根弦同品位 → 不触发横按。"""
        measure = TabMeasure(number=1, notes=[
            _make_note(string=1, fret=5),
            _make_note(string=2, fret=5),
        ], time_signature=(4, 4))
        warnings: list[str] = []
        _check_barre(measure, warnings)
        assert len(warnings) == 0

    def test_open_strings_not_counted_in_barre(self):
        """空弦不计入横按统计。"""
        measure = TabMeasure(number=1, notes=[
            _make_note(string=1, fret=0),  # 空弦
            _make_note(string=2, fret=0),  # 空弦
            _make_note(string=3, fret=0),  # 空弦
        ], time_signature=(4, 4))
        warnings: list[str] = []
        _check_barre(measure, warnings)
        assert len(warnings) == 0

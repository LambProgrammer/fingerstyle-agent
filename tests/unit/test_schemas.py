"""单元测试：Pydantic Schema 校验——数据契约不能破。

覆盖 TabNote / ArrangementPlan / SectionPlan 的核心约束。
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.api.schemas import (
    ArrangementPlan,
    SectionPlan,
    TabNote,
    Technique,
)


class TestTabNote:
    """TabNote：voice 字段向后兼容 + 弦/品范围约束。"""

    def test_voice_defaults_to_empty(self):
        """新字段 voice 默认为空字符串，兼容旧数据。"""
        n = TabNote(string=1, fret=3, start_time=0.0, duration=0.5)
        assert n.voice == ""

    def test_voice_accepts_valid_values(self):
        """voice 接受 melody / inner / bass。"""
        for v in ["melody", "inner", "bass"]:
            n = TabNote(string=1, fret=3, start_time=0.0, duration=0.5, voice=v)
            assert n.voice == v

    def test_old_data_without_voice_deserializes(self):
        """旧数据（无 voice 字段）反序列化后 voice 为空。"""
        old = {"string": 1, "fret": 3, "start_time": 0.0, "duration": 0.5}
        n = TabNote.model_validate(old)
        assert n.voice == ""

    def test_string_out_of_range_rejected(self):
        """弦号必须 1-6。"""
        with pytest.raises(PydanticValidationError):
            TabNote(string=7, fret=3, start_time=0.0, duration=0.5)

    def test_fret_out_of_range_rejected(self):
        """品位必须 0-24。"""
        with pytest.raises(PydanticValidationError):
            TabNote(string=1, fret=25, start_time=0.0, duration=0.5)

    def test_technique_defaults_to_none(self):
        """技巧默认为 NONE。"""
        n = TabNote(string=1, fret=3, start_time=0.0, duration=0.5)
        assert n.technique == Technique.NONE


class TestSectionPlan:
    """SectionPlan：段落参数约束。"""

    def test_valid_section_plan(self):
        """合法参数可以正常创建。"""
        sec = SectionPlan(
            measure_start=1, measure_end=8, label="intro",
            density="sparse", bass_style="root_only",
            melody_register="low", techniques=[],
            dynamic="p",
        )
        assert sec.measure_start == 1
        assert sec.measure_end == 8

    def test_invalid_density_rejected(self):
        """非法 density 值被拒绝。"""
        with pytest.raises(PydanticValidationError):
            SectionPlan(
                measure_start=1, measure_end=8, label="verse",
                density="extreme",  # 非法值
                bass_style="alternating", melody_register="mid",
            )

    def test_invalid_bass_style_rejected(self):
        """非法 bass_style 值被拒绝。"""
        with pytest.raises(PydanticValidationError):
            SectionPlan(
                measure_start=1, measure_end=8, label="verse",
                density="medium", bass_style="slap",  # 非法值
                melody_register="mid",
            )


class TestArrangementPlan:
    """ArrangementPlan：完整编排计划校验。"""

    def test_empty_sections_rejected(self):
        """sections 不能为空。"""
        with pytest.raises(PydanticValidationError):
            ArrangementPlan(sections=[], summary="empty")

    def test_valid_arrangement(self):
        """合法的编排计划可正常创建。"""
        plan = ArrangementPlan(
            sections=[
                SectionPlan(
                    measure_start=1, measure_end=16, label="verse",
                    density="medium", bass_style="alternating",
                    melody_register="mid",
                ),
            ],
            summary="测试编排",
        )
        assert len(plan.sections) == 1
        assert plan.summary == "测试编排"

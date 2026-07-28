"""LangGraph 工作流定义 —— 构建 6 Agent 节点 + 条件路由 + 回退循环。

图形结构（MIDI 管线 — ADR-001 P2 升级）：
  START → route_entry → agent_1 → agent_2 → agent_2_5(LLM编曲) → agent_3 → agent_4
                                                                          ↓
                                                                [should_retry?]
                                                                ↙              ↘
                                                           agent_3 (回退)      END

图形结构（修改管线）：
  START → route_entry → agent_5 → agent_3 → agent_4 → END

编译时注入：
  - M8 接入 PostgreSQL Checkpointer（短期记忆）
  - M8 接入 Redis Store（长期记忆——API 层 raw redis-py 实现）
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.nodes import (
    agent_1_midi_parse,
    agent_2_5_arrangement,
    agent_2_harmony_analysis,
    agent_3_tab_generate,
    agent_4_validate,
    agent_5_modify_understand,
    route_entry,
    route_entry_decision,
    should_retry,
)
from src.agents.state import AgentState
from src.memory.checkpointer import get_checkpointer


def build_graph() -> CompiledStateGraph:
    """构建并返回编译后的 StateGraph（注入 Checkpointer）。

    调用方通过 graph.invoke(state, config={"configurable": {"thread_id": ...}})
    执行完整链路。
    """
    builder = StateGraph(AgentState)

    # ---- 注册节点 ----
    builder.add_node("route_entry", route_entry)
    builder.add_node("agent_1", agent_1_midi_parse)
    builder.add_node("agent_2", agent_2_harmony_analysis)
    builder.add_node("agent_2_5", agent_2_5_arrangement)
    builder.add_node("agent_3", agent_3_tab_generate)
    builder.add_node("agent_4", agent_4_validate)
    builder.add_node("agent_5", agent_5_modify_understand)

    # ---- 入口：根据 modify_instruction 分流 ----
    builder.set_entry_point("route_entry")
    builder.add_conditional_edges(
        "route_entry",
        route_entry_decision,
        {
            "agent_1": "agent_1",
            "agent_5": "agent_5",
        },
    )

    # ---- MIDI 管线：1 → 2 → 2.5(LLM编曲) → 3 → 4 ----
    builder.add_edge("agent_1", "agent_2")
    builder.add_edge("agent_2", "agent_2_5")
    builder.add_edge("agent_2_5", "agent_3")
    builder.add_edge("agent_3", "agent_4")

    # ---- 校验后条件路由：通过 → END，不通过 → agent_3 ----
    builder.add_conditional_edges(
        "agent_4",
        should_retry,
        {
            "agent_3": "agent_3",
            "end": END,
        },
    )

    # ---- 修改管线：5 → 3 → 4（3→4 已在 MIDI 管线中定义，共享）----
    builder.add_edge("agent_5", "agent_3")

    # ---- 编译（注入 M8 Checkpointer；Store 在 API 层用 raw Redis 实现）----
    graph = builder.compile(checkpointer=get_checkpointer())
    return graph


# 模块级全局实例（懒编译，首次调用 build_graph() 时初始化）
_compiled_graph: CompiledStateGraph | None = None


def get_graph() -> CompiledStateGraph:
    """获取编译后的图实例（单例）。"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph

"""PostgreSQL Checkpointer —— LangGraph 短期记忆（会话状态持久化）。

在多 Agent 系统中的角色：
  LangGraph 框架在每次 node 执行后自动调用 Checkpointer.put(state)，
  将完整的 AgentState 存入 PostgreSQL。后续调用 graph.invoke() 时若
  传入相同的 thread_id，框架自动从 checkpoint 恢复上次的状态——
  用户刷新页面后状态不丢失。

连接方式：
  LangGraph 1.2.x 中 PostgresSaver.from_conn_string() 返回的是 context manager，
  需直接传入 psycopg connection 才能获取长期存活的 saver 实例。
"""

import logging

from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

from src.config import settings

logger = logging.getLogger(__name__)

# 模块级单例（复用连接）
_checkpointer: PostgresSaver | None = None


def get_checkpointer() -> PostgresSaver:
    """获取或创建 PostgreSQL Checkpointer 连接。

    首次调用时建立连接并自动初始化表结构。
    后续调用返回同一实例。
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    conn_string = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )
    logger.info("正在连接 PostgreSQL Checkpointer: %s:%s/%s",
                settings.postgres_host, settings.postgres_port, settings.postgres_db)

    conn = psycopg.connect(conn_string, autocommit=True)
    _checkpointer = PostgresSaver(conn)  # type: ignore[arg-type]
    # 首次运行时自动创建 checkpoint 相关表（幂等操作）
    _checkpointer.setup()
    logger.info("PostgreSQL Checkpointer 已就绪")
    return _checkpointer

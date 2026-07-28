"""LangSmith 可观测性配置。

里程碑 1 已写好基础设施代码，但开发期默认关闭追踪以免浪费免费额度，
里程碑 4（Agent 节点开发）起将 .env 中 LANGSMITH_TRACING=true 即可启用。
"""

import os
import logging

from src.config import settings

logger = logging.getLogger(__name__)


def configure_tracing() -> None:
    """根据 settings.langsmith_tracing 决定是否启用 LangSmith 追踪。

    当 LANGSMITH_TRACING=true 时：校验 API Key → 设置环境变量（LangChain 自动检测）→ 连通性检查。
    当 LANGSMITH_TRACING=false 时：仅打一行日志，不做任何网络调用。
    """
    if not settings.langsmith_tracing:
        logger.info("LangSmith 追踪已关闭（LANGSMITH_TRACING=false），进入 M4 后可改为 true")
        return

    api_key = settings.langsmith_api_key
    if not api_key:
        logger.warning("LANGSMITH_TRACING=true 但 LANGSMITH_API_KEY 未设置，追踪不会生效")
        return

    # langchain / langgraph 检测到环境变量即自动上报，无需额外代码
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    # 轻量连通性校验
    try:
        from langsmith import Client  # noqa: F811

        client = Client()
        # list_projects 是便宜的只读接口（返回生成器，需消费才真正发请求）；
        # 网络不通 / api_key 无效时会抛异常
        next(client.list_projects(limit=1), None)
        logger.info("LangSmith 追踪已启用，project=%s", settings.langsmith_project)
    except Exception as exc:
        logger.warning("LangSmith 连通性检查失败（追踪可能不生效）: %s", exc)


def is_tracing_enabled() -> bool:
    """供 M9 评估脚本调用，判断当前是否处于追踪就绪状态。"""
    return settings.langsmith_tracing and bool(settings.langsmith_api_key)

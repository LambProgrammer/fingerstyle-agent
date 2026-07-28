"""统一日志配置。

全项目（API 层、Agent 节点、工具层）共用一套格式，
以便在控制台串起一次请求经过多个 Agent 的完整轨迹。
用法：应用启动时调用一次 setup_logging()，各模块通过 get_logger(__name__) 取 logger。
"""

import logging
from logging.config import dictConfig

from src.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging() -> None:
    """配置根 logger：控制台输出，级别读自 .env 的 LOG_LEVEL。"""
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,  # 保留 uvicorn 等第三方 logger
            "formatters": {"default": {"format": _LOG_FORMAT}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": settings.log_level, "handlers": ["console"]},
        }
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

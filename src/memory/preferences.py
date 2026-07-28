"""Redis 用户偏好存储 —— M8 长期记忆（跨会话持久化）。

在多 Agent 系统中的角色：
  存储用户的风格偏好和定弦偏好，跨会话持久化。与 Checkpointer 不同：
  - Checkpointer 存的是 Agent 执行状态（自动，同 thread_id 内有效）
  - 本模块存的是业务层偏好（API 层手动读写，同 user_id 跨所有会话有效）

为什么不用 LangGraph RedisStore：
  LangGraph 的 RedisStore 底层依赖 RediSearch 模块（FT.SEARCH 命令），
  需要 Redis Stack 镜像。我们的 redis:7-alpine 不含此模块。
  偏好存取不经过 Agent 节点，用原始 redis-py 直接读写更简单可靠。

数据模型（Redis Hash）：
  Key:   fs:prefs:{user_id}
  Value: JSON {"style": "jpop", "tuning": "E2,A2,D3,G3,B3,E4"}
"""

import json
import logging

import redis

from src.config import settings

logger = logging.getLogger(__name__)

# 模块级单例
_redis_client: redis.Redis | None = None

# Redis key 前缀
_KEY_PREFIX = "fs:prefs:"


def _get_redis() -> redis.Redis:
    """获取或创建 Redis 连接。"""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    _redis_client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True,
    )
    try:
        _redis_client.ping()
        logger.info("Redis 已就绪: %s:%s", settings.redis_host, settings.redis_port)
    except Exception as exc:
        logger.error("Redis 连接失败: %s", exc)
    return _redis_client


def load_preferences(user_id: str) -> dict:
    """读取用户偏好（style + tuning）。

    Args:
        user_id: 前端 localStorage 中的 UUID。

    Returns:
        dict: {"style": "jpop", "tuning": "E2,A2,D3,G3,B3,E4"} 或空 dict。
    """
    try:
        r = _get_redis()
        data: str | None = r.get(_KEY_PREFIX + user_id)  # type: ignore[assignment]
        if data:
            return json.loads(data)
    except Exception as exc:
        logger.warning("读取用户偏好失败: %s", exc)
    return {}


def save_preferences(user_id: str, style: str, tuning: str) -> None:
    """保存用户偏好到 Redis。

    Args:
        user_id: 前端 localStorage 中的 UUID。
        style:    风格偏好（"jpop" / "american_folk" / "pop_adaptation"）。
        tuning:   定弦偏好（如 "E2,A2,D3,G3,B3,E4"）。
    """
    try:
        r = _get_redis()
        r.set(
            _KEY_PREFIX + user_id,
            json.dumps({"style": style, "tuning": tuning}, ensure_ascii=False),
        )
        logger.info("用户偏好已保存: user=%s style=%s", user_id[:8], style)
    except Exception as exc:
        logger.warning("保存用户偏好失败: %s", exc)

"""Chroma 向量数据库客户端封装。

在多 Agent 系统中的角色：
  为 RAG 曲谱库提供持久化的向量存储与相似度检索能力。
  Agent 链路不直接调用本模块——只在预处理入库（indexer.py）和检索（retriever.py）中使用。

技术方案：
  使用 chromadb 的 HttpClient 连接到 Docker 容器中的 Chroma 服务器。
  Collection 名称：song_tab_collection。
  embedding 由 sentence-transformers（all-MiniLM-L6-v2）在外部生成后传入，
  本模块只负责存储和检索，不负责生成向量。
"""

import logging
from typing import Any

import chromadb  # type: ignore[import-untyped]

from src.config import settings

logger = logging.getLogger(__name__)

# Chroma Collection 名称（全项目唯一）
COLLECTION_NAME = "song_tab_collection"

# 全局客户端实例（懒加载，复用连接）
# 注：chromadb.HttpClient 是工厂函数非类型，不做类型标注
_client: Any = None
_collection: Any = None


def _get_client():
    """获取或创建 Chroma HttpClient 连接。

    主机名和端口从 .env / settings 读取：
      - 容器内：chromadb:8000
      - 宿主机调试：localhost:8001（docker-compose 端口映射）
    """
    global _client
    if _client is None:
        host = settings.chroma_host
        port = settings.chroma_port
        logger.info("连接 Chroma 服务器: %s:%d", host, port)
        _client = chromadb.HttpClient(host=host, port=port)
        # 心跳检测
        try:
            _client.heartbeat()
            logger.info("Chroma 连接成功，心跳正常")
        except Exception as exc:
            logger.warning("Chroma 心跳检测失败（服务器可能未启动）: %s", exc)
    return _client


def get_collection() -> Any:
    """获取或创建 song_tab_collection。

    Chroma 的 Collection 是持久化的——首次创建后，后续调用直接返回已存在的。
    数据存储在 Docker 容器的 /data 目录，对应宿主机的 ./data/rag/。
    """
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "指弹吉他谱 RAG 曲谱库——歌名检索 + 类型分类"},
        )
        logger.info("Chroma Collection '%s' 已就绪", COLLECTION_NAME)
    return _collection


def add_songs(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> None:
    """批量添加歌曲文档到 Chroma。

    Args:
        ids:        唯一标识符（建议用文件名 MD5 哈希，防重）。
        documents:  文本描述（用于生成 embedding 的原始文本，如"歌名 - 艺术家 - 类型"）。
        metadatas:  元数据字典列表（title / artist / type / confidence / key / bpm / style 等）。
        embeddings: sentence-transformers 生成的 384 维向量。
    """
    if not ids:
        return
    collection = get_collection()
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    logger.info("Chroma: 已添加 %d 首歌曲", len(ids))


def query_songs(
    query_embedding: list[float],
    n_results: int = 10,
) -> dict[str, Any]:
    """向量相似度检索。

    Args:
        query_embedding: 查询文本的 embedding 向量（由 sentence-transformers 生成）。
        n_results:       返回的最大结果数。

    Returns:
        Chroma 查询结果字典，包含 ids / documents / metadatas / distances。
    """
    collection = get_collection()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    logger.debug("Chroma 检索返回 %d 条结果", len(results.get("ids", [[]])[0]))
    return results


def count_songs() -> int:
    """返回知识库中的歌曲总数。"""
    return get_collection().count()


def reset_collection() -> None:
    """⚠️ 删除整个 Collection（不可逆操作）。

    仅用于开发阶段重建知识库。生产环境不应调用。
    """
    global _collection
    client = _get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.warning("Chroma Collection '%s' 已删除", COLLECTION_NAME)
    except Exception:
        logger.debug("Collection 不存在，无需删除")
    _collection = None
    # 重建空 collection
    get_collection()

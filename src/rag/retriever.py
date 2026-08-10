"""RAG 检索器 —— 歌名搜索。

在多 Agent 系统中的角色：
  供 API 路由层调用：用户输入歌名 → 向量检索 → 返回最佳命中。
  路由层拿到结果后，统一走 Agent 完整链路生成指弹谱——无论 type 是什么。
  type 字段当前仅作为元数据标签（标记源素材的指弹适配度），不影响路由行为。
  后续版本（v2.0）经典指弹谱充足后，可启用 type 路由：
    full_tab（真·指弹成品谱）→ 跳过 Agent 直接渲染
    chord_only（仅有和弦素材）→ 走 Agent 完整链路

检索策略：
  1. sentence-transformers 将 query 转为 embedding
  2. Chroma 向量相似度检索 top_k=10
  3. 取前 3 个候选做 keyword match rerank：
     a. 分词查询 → 任意 token 匹配 title → 立即返回
     b. token 匹配 artist → 暂存作备选
     c. 全都不匹配 → 返回 None（拒识）

使用方式：
  from src.rag.retriever import retrieve_by_song_name
  result = retrieve_by_song_name("Yesterday")
  if result:
      print(result["type"], result["title"], result["artist"])
"""

import logging

from src.rag.chroma_client import query_songs
from src.rag.indexer import _get_embedder

logger = logging.getLogger("retriever")

# 检索兜底阈值：confidence 低于此值的 full_tab 降级为 chord_only
_CONFIDENCE_DOWNGRADE_THRESHOLD = 0.75
# keyword match 搜索的候选数
_KEYWORD_MATCH_TOP_K = 3


def _keyword_rerank(query: str, ids: list[str], metas: list[dict]) -> tuple[str, dict] | None:
    """top-K keyword match rerank——歌名优先，艺术家其次，全不匹配返回 None。"""
    tokens = [query.strip().lower()]
    for part in query.split():
        t = part.strip().lower()
        if t:
            tokens.append(t)

    artist_fallback: tuple[str, dict] | None = None
    limit = min(_KEYWORD_MATCH_TOP_K, len(metas))

    for i in range(limit):
        meta = metas[i]
        t = meta.get("title", "").lower()
        a = meta.get("artist", "").lower()
        tid = ids[i] if i < len(ids) else ""

        if any(tok in t for tok in tokens):
            logger.info("检索命中(title): '%s' → %s (rank=%d)", query, meta["title"], i + 1)
            return tid, meta

        if artist_fallback is None and any(tok in a for tok in tokens):
            artist_fallback = (tid, meta)

    if artist_fallback:
        meta = artist_fallback[1]
        logger.info("检索命中(artist): '%s' → %s / %s", query, meta["title"], meta["artist"])
        return artist_fallback

    logger.info("检索拒识: '%s' — top-%d keyword match 全失败", query, limit)
    return None


def _build_rag_result(best_id: str, best_meta: dict) -> dict:
    """构建返回的 dict——抽取公共字段。"""
    tag = best_meta.get("type", "chord_only")
    confidence = best_meta.get("confidence", 0)
    if tag == "full_tab" and confidence < _CONFIDENCE_DOWNGRADE_THRESHOLD:
        tag = "chord_only"

    return {
        "id": best_id,
        "title": best_meta.get("title", ""),
        "artist": best_meta.get("artist", ""),
        "type": tag,
        "confidence": confidence,
        "score": best_meta.get("fingerstyle_score", 0),
        "bpm": best_meta.get("bpm", 0),
        "key": best_meta.get("key", ""),
        "style": best_meta.get("style", ""),
        "file_path": best_meta.get("file_path", ""),
        "curated": best_meta.get("curated", False),
    }


def retrieve_by_song_name(query: str, top_k: int = 10) -> dict | None:
    """根据歌名搜索最佳匹配。

    Args:
        query:  用户输入的歌名（支持中/英/日文，支持部分匹配）。
        top_k:  Chroma 检索的最大候选数。

    Returns:
        dict: {"id", "title", "artist", "type", "confidence", "score", "bpm",
               "key", "style", "file_path", "curated"}
        或 None（无任何命中）。
    """
    if not query.strip():
        return None

    # 构造查询 document（与入库时同样的 _build_document 格式，确保语义空间一致）
    query_doc = f"歌名: {query.strip()}"
    embedder = _get_embedder()
    query_emb = embedder.encode(query_doc, show_progress_bar=False).tolist()

    raw = query_songs(query_emb, n_results=top_k)
    ids_list = raw.get("ids", [[]])
    metas_list = raw.get("metadatas", [[]])

    if not ids_list or not ids_list[0]:
        logger.info("检索无结果: '%s'", query)
        return None

    ids = ids_list[0]
    metas = metas_list[0]

    # top-3 keyword match rerank：歌名优先，艺术家其次，全不匹配 = 拒识
    best_meta = _keyword_rerank(query, ids, metas)

    if best_meta is None:
        return None

    result = _build_rag_result(best_meta[0], best_meta[1])
    logger.info("检索命中: '%s' → %s (type=%s)", query, result["title"], result["type"])
    return result



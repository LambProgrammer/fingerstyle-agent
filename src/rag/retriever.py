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
  3. 信任 Chroma 余弦距离排序，不做 type/confidence 二次重排
     （二次重排会破坏语义相似度的自然顺序，导致错配）
  4. 当前所有入库标签统一为 chord_only（LMD 素材均为和弦级，评分配方待 v2.0 重新校准）

使用方式：
  from src.rag.retriever import retrieve_by_song_name
  result = retrieve_by_song_name("Yesterday")
  if result:
      print(result["type"], result["title"], result["artist"])
"""

import logging

from src.rag.chroma_client import query_songs
from src.rag.indexer import _build_document, _get_embedder

logger = logging.getLogger("retriever")

# 检索兜底阈值：confidence 低于此值的 full_tab 降级为 chord_only
_CONFIDENCE_DOWNGRADE_THRESHOLD = 0.75


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
    docs_list = raw.get("documents", [[]])
    metas_list = raw.get("metadatas", [[]])

    if not ids_list or not ids_list[0]:
        logger.info("检索无结果: '%s'", query)
        return None

    ids = ids_list[0]
    metas = metas_list[0]

    # Chroma 已按余弦距离排序（最近的在前），直接取第一个作为最佳命中。
    # 不做 type/confidence 的二次重排——那会破坏语义相似度的自然顺序，
    # 导致"搜'黄昏'命中一个 full_tab 英文歌"的错配。
    best_meta = metas[0]

    # 检索兜底：confidence < 阈值 的 full_tab → 降级为 chord_only
    tag = best_meta.get("type", "chord_only")
    confidence = best_meta.get("confidence", 0)
    if tag == "full_tab" and confidence < _CONFIDENCE_DOWNGRADE_THRESHOLD:
        logger.info(
            "full_tab 置信度 %.2f < %.2f，降级为 chord_only（避免低质量直出）",
            confidence, _CONFIDENCE_DOWNGRADE_THRESHOLD,
        )
        tag = "chord_only"

    result = {
        "id": ids[0] if ids else "",
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
    logger.info("检索命中: '%s' → %s (type=%s, conf=%.2f)", query, result["title"], tag, confidence)
    return result



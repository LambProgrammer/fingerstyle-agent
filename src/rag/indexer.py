"""曲谱向量化入库器 —— 批量生成 embedding 并写入 Chroma。

在多 Agent 系统中的角色：
  消费 seed_rag.py 产出的 JSON 报告 → sentence-transformers 生成文本嵌入
  → 写入 Chroma song_tab_collection。本模块不直接面向最终用户，由 CLI 调用。

技术方案：
  sentence-transformers all-MiniLM-L6-v2（384 维，MIT 许可，已安装）
  每首歌构造一段自然语言描述文本作为 embedding 输入。
  批量写入 Chroma（每批 128 首，平衡内存与网络效率）。
"""

import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

from src.rag.chroma_client import add_songs, count_songs, reset_collection

logger = logging.getLogger("indexer")

# Embedding 模型（懒加载）
_embedder: SentenceTransformer | None = None
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_BATCH_SIZE = 128


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("加载 sentence-transformers 模型: %s", _MODEL_NAME)
        _embedder = SentenceTransformer(_MODEL_NAME, local_files_only=True)
    return _embedder


def _build_document(report: dict) -> str:
    """将 report 转换为 embedding 输入文本。

    格式决定检索质量——包含歌名、艺术家、类型、风格、调性、BPM。
    歌名权重最高（放在最前面），因为用户搜索的是歌名。
    """
    title = report.get("title", "未知")
    artist = report.get("artist", "")
    tag = report.get("type", "chord_only")
    style = report.get("style", "pop_adaptation")
    key = report.get("key", "")
    bpm = report.get("bpm", 0)

    parts = [f"歌名: {title}"]
    if artist:
        parts.append(f"艺术家: {artist}")
    parts.append(f"类型: {tag}")
    parts.append(f"风格: {style}")
    if key:
        parts.append(f"调性: {key}")
    if bpm:
        parts.append(f"BPM: {bpm}")

    return " - ".join(parts)


def index_reports(reports_dir: str | Path, reset: bool = False) -> dict:
    """批量向量化入库。

    Args:
        reports_dir: seed_rag.py 产出的 reports 目录。
        reset:       是否先清空 Collection（重建知识库时用）。

    Returns:
        统计 dict: {total, indexed, skipped, errors}
    """
    reports_path = Path(reports_dir)
    if not reports_path.exists():
        raise FileNotFoundError(f"Reports 目录不存在: {reports_dir}")

    if reset:
        logger.warning("重置 Chroma Collection...")
        reset_collection()

    report_files = sorted(reports_path.glob("*.json"))
    total = len(report_files)
    logger.info("开始向量化入库: %d 个报告", total)

    embedder = _get_embedder()
    stats = {"total": total, "indexed": 0, "skipped": 0, "errors": 0}

    ids_batch: list[str] = []
    docs_batch: list[str] = []
    metas_batch: list[dict] = []
    emb_batch: list[list[float]] = []

    for i, rf in enumerate(report_files):
        try:
            report = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:
            stats["errors"] += 1
            continue

        doc_text = _build_document(report)
        embedding = embedder.encode(doc_text, show_progress_bar=False).tolist()
        if i == 0:
            logger.debug("first encode OK, emb_len=%d", len(embedding))

        ids_batch.append(report["id"])
        docs_batch.append(doc_text)
        emb_batch.append(embedding)
        metas_batch.append({
            "title": report.get("title", ""),
            "artist": report.get("artist", ""),
            "type": report.get("type", ""),
            "confidence": report.get("confidence", 0),
            "fingerstyle_score": report.get("fingerstyle_score", 0),
            "key": report.get("key", ""),
            "bpm": report.get("bpm", 0),
            "style": report.get("style", ""),
            "file_path": report.get("file_path", ""),
            "curated": report.get("curated", False),
        })

        if len(ids_batch) >= _BATCH_SIZE:
            logger.debug("flushing batch of %d", len(ids_batch))
            add_songs(ids_batch, docs_batch, metas_batch, emb_batch)
            stats["indexed"] += len(ids_batch)
            ids_batch, docs_batch, metas_batch, emb_batch = [], [], [], []

    # 最后一批（防御：确保四列表长度一致）
    logger.debug("final batch: ids=%d embs=%d", len(ids_batch), len(emb_batch))
    if ids_batch:
        n = len(ids_batch)
        if len(emb_batch) != n:
            logger.error("批次长度不一致: ids=%d docs=%d metas=%d embs=%d，跳过最后一批",
                         n, len(docs_batch), len(metas_batch), len(emb_batch))
        else:
            add_songs(ids_batch, docs_batch, metas_batch, emb_batch)
            stats["indexed"] += n

    logger.info("入库完成: %d/%d 已索引 (跳过 %d, 错误 %d)",
                stats["indexed"], stats["total"], stats["skipped"], stats["errors"])
    logger.info("Chroma 当前文档数: %d", count_songs())
    return stats


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse
    parser = argparse.ArgumentParser(description="向量化入库 Chroma")
    parser.add_argument("--reports-dir", default="data/processed/reports")
    parser.add_argument("--reset", action="store_true", help="重建知识库（清空现有数据）")
    args = parser.parse_args()
    index_reports(args.reports_dir, reset=args.reset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    main()

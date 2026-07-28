#!/usr/bin/env python3
"""RAG 曲谱库预处理流水线。

遍历 MIDI 文件 → 解析 → 特征提取 → 打分 → 打标签 → 输出 JSON 报告。

用法：
  # 子集测试（随机 100 个文件）
  uv run python scripts/seed_rag.py --subset 100

  # 全量运行（默认 3 workers）
  uv run python scripts/seed_rag.py

  # 自定义 workers
  uv run python scripts/seed_rag.py --workers 2

输出：
  data/processed/reports/*.json  —— 每个 MIDI 文件的分析报告
  data/processed/summary.json    —— 汇总统计

数据流（在 Indexer 阶段消费）：
  reports/*.json → indexer.py → Chroma song_tab_collection
"""

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.schemas import MidiNote  # noqa: E402
from src.tools.midi_parser import parse_midi  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("seed_rag")

# =============================================================================
# 可调参数
# =============================================================================

_MIN_NOTE_COUNT = 50
_MIN_DURATION_QL = 30.0  # quarterLength，约 15 秒（120BPM 下 1QL≈0.5s）
_MAX_WORKERS = 3        # 多进程 worker 数（物理核数 - 1 = 4 - 1）

# 打分权重
_W_POLYPHONY = 0.35
_W_DENSITY = 0.25
_W_GUITAR = 0.25
_W_RHYTHM = 0.15

# 分类阈值
_SCORE_FULL_TAB = 60
_SCORE_CHORD_ONLY = 30

# GM 乐器分组
_GM_GUITAR = set(range(24, 32))
_GM_PIANO = set(range(0, 8))


# =============================================================================
# 特征提取
# =============================================================================


def extract_features(notes: list[MidiNote]) -> dict:
    """从音符序列中提取特征值。"""
    if not notes:
        return {"note_count": 0}

    note_count = len(notes)
    total_duration = max(n.start_time + n.duration for n in notes)
    if total_duration <= 0:
        total_duration = 1.0

    note_density = note_count / total_duration

    # 复音数：按 0.25QL 窗口统计同时发声音符数
    polyphony_buckets: dict[int, int] = defaultdict(int)
    for n in notes:
        bucket = int(n.start_time / 0.25)
        polyphony_buckets[bucket] += 1
    poly_values = list(polyphony_buckets.values())
    polyphony_avg = sum(poly_values) / len(poly_values) if poly_values else 0
    polyphony_max = max(poly_values) if poly_values else 0

    # GM 乐器：当前 midi_parser 未保留 channel，留空（M10 增强）
    gm_instruments: list[int] = []

    # 节奏规律性
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    iois = [
        sorted_notes[i].start_time - sorted_notes[i - 1].start_time
        for i in range(1, len(sorted_notes))
        if sorted_notes[i].start_time > sorted_notes[i - 1].start_time
    ]
    if iois:
        mean_ioi = sum(iois) / len(iois)
        variance = sum((g - mean_ioi) ** 2 for g in iois) / len(iois)
        rhythm_regularity = 1.0 / (1.0 + variance)
    else:
        rhythm_regularity = 0.0

    return {
        "note_count": note_count,
        "total_duration": round(total_duration, 2),
        "note_density": round(note_density, 3),
        "polyphony_avg": round(polyphony_avg, 2),
        "polyphony_max": polyphony_max,
        "gm_instruments": gm_instruments,
        "rhythm_regularity": round(rhythm_regularity, 3),
    }


# =============================================================================
# 打分与分类
# =============================================================================


def compute_score(features: dict) -> float:
    """fingerstyle_score（0-100）。"""
    if features.get("note_count", 0) == 0:
        return 0.0

    poly = features["polyphony_avg"]
    density = features["note_density"]
    rhythm = features["rhythm_regularity"]
    gm_list = features.get("gm_instruments", [])

    # 复音数：2-4 为指弹黄金区间
    if 2.0 <= poly <= 4.0:
        polyphony_score = 1.0
    elif 1.5 <= poly < 2.0 or 4.0 < poly <= 5.5:
        polyphony_score = 0.6
    elif 1.0 <= poly < 1.5 or 5.5 < poly <= 7.0:
        polyphony_score = 0.3
    else:
        polyphony_score = 0.1

    # 密度：1-3 音/QL 适中
    if 1.0 <= density <= 3.0:
        density_score = 1.0
    elif 0.5 <= density < 1.0:
        density_score = 0.6
    elif 3.0 < density <= 6.0:
        density_score = 0.5
    else:
        density_score = 0.2

    # 吉他偏向
    has_guitar = any(g in _GM_GUITAR for g in gm_list)
    has_piano = any(g in _GM_PIANO for g in gm_list)
    guitar_bias = 1.0 if has_guitar else (0.5 if has_piano else 0.3)

    # 节奏
    if rhythm >= 0.7:
        rhythm_score = 1.0
    elif rhythm >= 0.4:
        rhythm_score = 0.6
    else:
        rhythm_score = 0.3

    raw = (
        polyphony_score * _W_POLYPHONY
        + density_score * _W_DENSITY
        + guitar_bias * _W_GUITAR
        + rhythm_score * _W_RHYTHM
    )
    return round(raw * 100, 1)


def classify(score: float) -> tuple[str, float]:
    """分数 → (type, confidence)。"""
    if score >= _SCORE_FULL_TAB:
        return "full_tab", round(score / 100, 3)
    if score >= _SCORE_CHORD_ONLY:
        return "chord_only", round(score / 100, 3)
    return "unknown", 0.0


# =============================================================================
# 文件遍历
# =============================================================================


def iter_midi_files(root: Path) -> list[Path]:
    """递归收集 .mid/.midi 文件。"""
    files = list(root.rglob("*.mid")) + list(root.rglob("*.midi"))
    logger.info("扫描 %s: 找到 %d 个 MIDI 文件", root, len(files))
    return files


def _load_metadata(metadata_path: Path | None = None) -> dict[str, dict[str, str]]:
    """加载 MSD track metadata（MSD ID → {title, artist}）。"""
    if metadata_path is None:
        metadata_path = Path("data/lmd_metadata.jsonl")
    if not metadata_path.exists():
        logger.warning("metadata 文件不存在: %s，歌名将使用文件名", metadata_path)
        return {}
    meta = {}
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tid = rec.get("track_id", "")
            if tid:
                meta[tid] = {"title": rec.get("title", ""), "artist": rec.get("artist", "")}
    logger.info("已加载 %d 首歌曲元数据", len(meta))
    return meta


# 模块级缓存（首次调用时加载）
_METADATA_CACHE: dict[str, dict[str, str]] | None = None


def _extract_metadata(midi_path: Path) -> tuple[str, str]:
    """文件路径 → (title, artist)。

    优先级：
      1. MSD ID 查表 → 真实歌名+歌手（lmd_metadata.jsonl）
      2. 文件名含 " - " → 拆分为 (艺术家, 歌名)
      3. 父目录是 MSD ID 但不在元数据中 → (hash, "LMD")
      4. 其他 → (文件名, "")
    """
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        _METADATA_CACHE = _load_metadata()

    stem = midi_path.stem

    # 1. 有真实元数据
    msd_id = midi_path.parent.name
    if msd_id in _METADATA_CACHE:
        m = _METADATA_CACHE[msd_id]
        return m["title"], m["artist"]

    # 2. 文件名含 " - "
    if " - " in stem:
        parts = stem.split(" - ", 1)
        return parts[1].strip(), parts[0].strip()

    # 3. MSD ID 但无元数据
    if len(msd_id) >= 18 and msd_id.startswith("TR"):
        return stem, "LMD"

    # 4. 其他
    return stem, ""


def _guess_style(midi_path: Path, features: dict) -> str:
    """文件名 + 特征启发 → 风格标签。"""
    stem = midi_path.stem.lower()
    if any(kw in stem for kw in ("jpop", "anime", "japanese", "j-pop")):
        return "jpop"
    if any(kw in stem for kw in ("folk", "country", "fingerpick", "fingerstyle")):
        return "american_folk"
    return "pop_adaptation"


def _file_id(midi_path: Path) -> str:
    return hashlib.md5(str(midi_path).encode()).hexdigest()[:12]


# =============================================================================
# 单文件处理
# =============================================================================


def process_file(midi_path: Path) -> dict | None:
    """解析 → 特征 → 打分 → 标签。返回 report dict 或 None（被过滤）。"""
    try:
        notes, _melody, bpm = parse_midi(str(midi_path))
    except Exception as exc:
        logger.debug("解析失败 %s: %s", midi_path.name, exc)
        return None

    if len(notes) < _MIN_NOTE_COUNT:
        return None

    features = extract_features(notes)
    if features.get("note_count", 0) == 0:
        return None

    if features.get("total_duration", 0) < _MIN_DURATION_QL:
        return None

    score = compute_score(features)
    tag, confidence = classify(score)
    if tag == "unknown":
        return None

    title, artist = _extract_metadata(midi_path)

    return {
        "id": _file_id(midi_path),
        "file_path": midi_path.as_posix(),
        "title": title,
        "artist": artist,
        "type": tag,
        "confidence": confidence,
        "fingerstyle_score": score,
        "bpm": bpm,
        "key": "",
        "style": _guess_style(midi_path, features),
        "features": features,
    }


# =============================================================================
# 流水线主入口
# =============================================================================


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    subset: int | None = None,
    curated_dir: Path | None = None,
    workers: int = _MAX_WORKERS,
) -> dict:
    """主流水线。"""
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. LMD 文件
    all_files = iter_midi_files(input_dir)
    if subset and subset < len(all_files):
        random.seed(42)
        all_files = random.sample(all_files, subset)
        logger.info("子集模式：随机抽取 %d 个文件", len(all_files))

    stats: dict[str, float] = {"total_scanned": len(all_files), "parse_failed": 0,
             "filtered_out": 0, "tagged_full_tab": 0, "tagged_chord_only": 0, "curated": 0}
    processed = 0
    t_start = time.time()

    # 多进程并行处理（3 workers, 4 核物理机）
    logger.info("使用 %d workers 并行处理 %d 个文件", workers, len(all_files))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, p): p for p in all_files}
        for future in as_completed(futures):
            midi_path = futures[future]
            try:
                report = future.result()
            except Exception as exc:
                logger.debug("Worker 异常 %s: %s", midi_path.name, exc)
                stats["filtered_out"] += 1
                continue

            if report is None:
                stats["filtered_out"] += 1
                continue

            _write_report(reports_dir, report)
            if report["type"] == "full_tab":
                stats["tagged_full_tab"] += 1
            else:
                stats["tagged_chord_only"] += 1
            processed += 1
            if processed % 500 == 0:
                elapsed = time.time() - t_start
                logger.info("进度: %d/%d (%.1f files/s)", processed, len(all_files),
                            processed / elapsed if elapsed > 0 else 0)

    # 2. 核心曲库
    if curated_dir and curated_dir.exists():
        curated_files = iter_midi_files(curated_dir)
        logger.info("核心曲库: %d 个文件（强制 chord_only, confidence=1.0）", len(curated_files))
        for midi_path in curated_files:
            try:
                notes, _melody, bpm = parse_midi(str(midi_path))
            except Exception as exc:
                logger.warning("核心曲库解析失败 %s: %s", midi_path.name, exc)
                continue
            features = extract_features(notes)
            title, artist = _extract_metadata(midi_path)
            report = {
                "id": _file_id(midi_path), "file_path": midi_path.as_posix(),
                "title": title, "artist": artist,
                "type": "chord_only", "confidence": 1.0, "fingerstyle_score": 100.0,
                "bpm": bpm, "key": "", "style": _guess_style(midi_path, features),
                "features": features, "curated": True,
            }
            _write_report(reports_dir, report)
            stats["curated"] += 1
            stats["tagged_chord_only"] += 1

    # 3. 汇总
    elapsed = time.time() - t_start
    stats["total_processed"] = processed + stats["curated"]
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["files_per_second"] = round(stats["total_processed"] / elapsed, 1) if elapsed > 0 else 0

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("汇总: %s", summary_path)
    logger.info("=" * 50)
    logger.info("完成: %d files → %d入库 (%d full_tab, %d chord_only, %d curated)",
                stats["total_scanned"], stats["total_processed"],
                stats["tagged_full_tab"], stats["tagged_chord_only"] - stats["curated"], stats["curated"])
    logger.info("过滤: %d | 耗时: %.1fs", stats["filtered_out"], elapsed)
    logger.info("=" * 50)
    return stats


def _write_report(reports_dir: Path, report: dict) -> None:
    (reports_dir / f"{report['id']}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="RAG 曲谱库预处理流水线")
    parser.add_argument("--input-dir", default="data/raw_midi/lmd_matched")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--subset", type=int, default=0,
                        help="测试模式：只处理 N 个随机文件（0=全量）")
    parser.add_argument("--workers", type=int, default=_MAX_WORKERS,
                        help=f"并行进程数（默认 {_MAX_WORKERS}）")
    parser.add_argument("--curated-dir", default="data/curated_fingerstyle")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error("输入目录不存在: %s", input_dir)
        sys.exit(1)

    workers = args.workers

    subset = args.subset if args.subset > 0 else None
    curated_dir = Path(args.curated_dir) if Path(args.curated_dir).exists() else None

    run_pipeline(input_dir, Path(args.output_dir), subset, curated_dir, workers)


if __name__ == "__main__":
    main()

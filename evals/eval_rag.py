"""层 3：RAG 检索评估（手动触发，不涉及 LLM 成本）。

评估指标：
  - hit@1：应命中的查询——最近邻是否正确返回
  - 正确拒识率：不应命中的查询——是否正确返回"未找到"

当前 retriever 只取最近邻（不返回 top-k），故不评估 hit@3 和
优先级排序（M5 的二次排序 _rank() 已删除，信任 Chroma 余弦距离）。

运行：uv run python evals/eval_rag.py
输出：终端打印两个指标。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.retriever import retrieve_by_song_name

DATASET_PATH = Path(__file__).parent / "datasets" / "rag_queries.jsonl"


def _is_hit(result: dict | None) -> bool:
    return result is not None and result.get("title") is not None


def evaluate() -> dict:
    with open(DATASET_PATH, encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f if line.strip()]

    print(f"RAG 检索评估：{len(test_cases)} 条查询")
    print("-" * 50)

    total_hit = 0
    hit_count = 0
    total_miss = 0
    correct_miss = 0

    for tc in test_cases:
        query = tc["query"]
        should_hit = tc["should_hit"]
        result = retrieve_by_song_name(query)
        is_hit = _is_hit(result)

        status = "OK" if (is_hit == should_hit) else "XX"
        info = f"-> {result['title']}" if result else "-> (无结果)"
        print(f"  [{status}] '{query}' {info}")

        if should_hit:
            total_hit += 1
            if is_hit:
                hit_count += 1
        else:
            total_miss += 1
            if not is_hit:
                correct_miss += 1

    hit1 = (hit_count / total_hit * 100) if total_hit > 0 else 0
    miss_rate = (correct_miss / total_miss * 100) if total_miss > 0 else 0

    print("-" * 50)
    print(f"应命中 {total_hit} 条, 实际命中 {hit_count} 条")
    print(f"应未命中 {total_miss} 条, 实际正确拒识 {correct_miss} 条")
    print(f"hit@1: {hit1:.1f}%")
    print(f"正确拒识率: {miss_rate:.1f}%")

    return {
        "total": len(test_cases),
        "hit1": round(hit1, 1),
        "correct_miss_rate": round(miss_rate, 1),
    }


if __name__ == "__main__":
    evaluate()

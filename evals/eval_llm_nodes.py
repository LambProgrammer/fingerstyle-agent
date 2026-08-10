"""层 2：Agent 5 指令解析准确率评估（手动触发，涉及 LLM 调用成本）。

评估指标：
  - op 类型准确率：LLM 输出的 operations[0].op 与期望是否一致
  - scope 类别准确率：LLM 输出的 scope 归入哪个大类（verse/chorus/bridge/entire）

运行：uv run python evals/eval_llm_nodes.py
输出：终端打印准确率，若 LangSmith 已配置则自动上传 Experiment。
"""

import json
import logging
import sys
from pathlib import Path

# 项目路径（从 evals/ 目录跑时需要）
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.nodes import _OPERATIONS_SYSTEM_PROMPT, _parse_modification_plan, _get_llm
from src.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ---- 数据集路径 ----
DATASET_PATH = Path(__file__).parent / "datasets" / "instructions.jsonl"


def _classify_scope(scope: str) -> str:
    """将 LLM 输出的自然语言 scope 归类到大类。"""
    s = scope.strip().lower()
    if s in ("entire_song", "全曲", ""):
        return "entire_song"
    if s in ("chorus", "副歌"):
        return "chorus"
    if s in ("bridge", "间奏", "interlude"):
        return "bridge"
    if s in ("verse", "主歌", "intro", "outro", "前奏", "尾奏"):
        return "verse"
    if "measure" in s or "小节" in s or "第" in s:
        return "measure_range"
    return "other"


def evaluate() -> dict:
    """跑评估：逐条调 Agent 5 LLM → 对比期望 → 算准确率。"""
    if not settings.deepseek_api_key:
        print("[SKIP] 未配置 DEEPSEEK_API_KEY，跳过 LLM 评估")
        return {}

    # 加载数据集
    with open(DATASET_PATH, encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f if line.strip()]

    print(f"Agent 5 指令解析评估：{len(test_cases)} 条测试")
    print("LLM 模型：deepseek-v4-pro")
    print("-" * 50)

    llm = _get_llm()
    op_correct = 0
    scope_correct = 0
    total = 0
    results = []

    for i, tc in enumerate(test_cases):
        instruction = tc["instruction"]
        expected_op = tc["expected_op"]
        expected_scope = tc["expected_scope"]

        # 调用 Agent 5 同款 LLM 逻辑
        response = llm.invoke([
            {"role": "system", "content": _OPERATIONS_SYSTEM_PROMPT},
            {"role": "user", "content": f"用户修改指令：{instruction}\n当前谱面：5 小节，调性=C major，速度=100BPM，风格=jpop\n请输出修改 operations JSON："},
        ])
        raw = response.content if hasattr(response, "content") else str(response)
        plan_text: str = raw if isinstance(raw, str) else str(raw)
        plan = _parse_modification_plan(plan_text)
        total += 1

        # 判断 op 准确率
        actual_ops = plan.operations
        actual_op = actual_ops[0].op if actual_ops else "none"
        actual_scope = actual_ops[0].scope if actual_ops else "entire_song"
        actual_scope_cat = _classify_scope(actual_scope)

        op_match = actual_op == expected_op
        scope_match = actual_scope_cat == expected_scope

        if op_match:
            op_correct += 1
        if scope_match:
            scope_correct += 1

        status = "OK" if op_match else "XX"
        print(f"  [{status}] '{instruction}'")
        print(f"       期望 op={expected_op} scope={expected_scope}")
        print(f"       实际 op={actual_op} scope={actual_scope} → {actual_scope_cat}")

        results.append({
            "instruction": instruction,
            "expected_op": expected_op,
            "actual_op": actual_op,
            "expected_scope": expected_scope,
            "actual_scope": actual_scope_cat,
            "op_match": op_match,
            "scope_match": scope_match,
        })

    op_rate = (op_correct / total * 100) if total > 0 else 0
    scope_rate = (scope_correct / total * 100) if total > 0 else 0

    print("-" * 50)
    print(f"op 类型准确率: {op_correct}/{total} = {op_rate:.1f}%")
    print(f"scope 类别准确率: {scope_correct}/{total} = {scope_rate:.1f}%")

    return {
        "total": total,
        "op_accuracy": round(op_rate, 1),
        "scope_accuracy": round(scope_rate, 1),
        "results": results,
    }


if __name__ == "__main__":
    evaluate()

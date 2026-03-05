"""Run cortex against GAIA benchmark and produce evaluation JSONL.

Usage:
    # From cortex worktree root:
    uv run python scripts/eval/gaia/run_gaia.py \
        --input /home/theo/projects/LLM-planning/data/GAIA/gaia.infer/gaia.infer.jsonl \
        --output scripts/eval/gaia/cortex_gaia_results.jsonl \
        --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import sys
import time
from pathlib import Path

# Ensure cortex root is on sys.path
CORTEX_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CORTEX_ROOT))

from app.config import CortexConfig  # noqa: E402
from app.task.task_manager import TaskManager  # noqa: E402
from cortex import Cortex  # noqa: E402
from scripts.eval.gaia.converter import convert_plan_to_pred  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --- Monkey-patch to capture Plan before removal ---
_captured_plans: dict[str, object] = {}
_original_remove = TaskManager.remove_plan


@classmethod  # type: ignore[misc]
def _capturing_remove(cls: type, plan_id: str) -> None:
    plan = TaskManager.get_plan(plan_id)
    if plan is not None:
        _captured_plans[plan_id] = copy.deepcopy(plan)
    _original_remove(plan_id)


TaskManager.remove_plan = _capturing_remove  # type: ignore[assignment]


def _pop_latest_plan() -> object | None:
    """Pop the most recently captured plan."""
    if not _captured_plans:
        return None
    latest_key = max(_captured_plans.keys())
    return _captured_plans.pop(latest_key)


def load_completed_ids(output_path: Path) -> set[str]:
    """Load sample IDs already in the output file for resume support."""
    ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    sample_id = record.get("meta", {}).get("id", "")
                    if sample_id:
                        ids.add(sample_id)
                except json.JSONDecodeError:
                    continue
    return ids


async def run_single(cortex_instance: Cortex, query: str) -> tuple[object | None, str]:
    """Run cortex on a single query, return (plan, result_text)."""
    result_text = await cortex_instance.execute(query)
    plan = _pop_latest_plan()
    return plan, result_text


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run cortex on GAIA benchmark")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/theo/projects/LLM-planning/data/GAIA/gaia.infer/gaia.infer.jsonl"),
        help="Input GAIA JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(CORTEX_ROOT / "scripts/eval/gaia/cortex_gaia_results.jsonl"),
        help="Output JSONL with pred fields",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max samples to run (0=all)")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed samples")
    args = parser.parse_args()

    # Load input
    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d GAIA records from %s", len(records), args.input)

    if args.limit > 0:
        records = records[: args.limit]
        logger.info("Limited to %d records", len(records))

    # Resume support
    completed_ids: set[str] = set()
    if args.resume:
        completed_ids = load_completed_ids(args.output)
        logger.info("Resuming: %d samples already completed", len(completed_ids))

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Init cortex
    config = CortexConfig()  # type: ignore[call-arg]
    cortex_instance = Cortex(config)

    # Run each sample
    for i, record in enumerate(records):
        _captured_plans.clear()  # Reset to prevent cross-sample plan leakage
        sample_id = record.get("meta", {}).get("id", f"unknown_{i}")

        if sample_id in completed_ids:
            logger.info("[%d/%d] Skipping %s (already done)", i + 1, len(records), sample_id)
            continue

        query = record["query"]["user_query"]
        extra = record["query"].get("extra_instruction", "")
        if extra:
            query = f"{query}\n\n{extra}"

        # Append instruction to use submit_final_answer
        query_with_instruction = (
            f"{query}\n\n"
            "Important: When you have the final answer, you MUST call the "
            "submit_final_answer tool with a concise answer string."
        )

        logger.info("[%d/%d] Running sample %s", i + 1, len(records), sample_id)
        t0 = time.time()

        try:
            plan, result_text = await run_single(cortex_instance, query_with_instruction)

            if plan is not None:
                pred = convert_plan_to_pred(plan, result_text)
            else:
                logger.warning("No plan captured for sample %s", sample_id)
                pred = {
                    "plan_dag": {"nodes": [], "edges": []},
                    "tool_calls": [],
                    "final_answer": {"answer_type": "string", "answer": ""},
                    "_parse_error": True,
                }

            record["pred"] = pred
            elapsed = time.time() - t0
            logger.info(
                "[%d/%d] Done %s in %.1fs — %d nodes, %d tool calls, answer=%s",
                i + 1,
                len(records),
                sample_id,
                elapsed,
                len(pred["plan_dag"]["nodes"]),
                len(pred["tool_calls"]),
                repr(pred["final_answer"].get("answer", "")[:50]),
            )

        except Exception:
            logger.exception("[%d/%d] FAILED sample %s", i + 1, len(records), sample_id)
            record["pred"] = {
                "plan_dag": {"nodes": [], "edges": []},
                "tool_calls": [],
                "final_answer": {"answer_type": "string", "answer": ""},
                "_parse_error": True,
            }

        # Append to output (incremental write)
        with open(args.output, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(main())

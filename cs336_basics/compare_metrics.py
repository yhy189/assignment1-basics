from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid metrics payload: {path}")
    if "history" not in payload or not isinstance(payload["history"], list):
        raise ValueError(f"Missing history in metrics payload: {path}")
    return payload


def _last_value(history: list[dict[str, Any]], key: str) -> tuple[float | None, int | None]:
    for row in reversed(history):
        if key in row:
            return float(row[key]), int(row["step"])
    return None, None


def _best_value(history: list[dict[str, Any]], key: str) -> tuple[float | None, int | None]:
    best_val = None
    best_step = None
    for row in history:
        if key not in row:
            continue
        value = float(row[key])
        if best_val is None or value < best_val:
            best_val = value
            best_step = int(row["step"])
    return best_val, best_step


def _mean_tail(history: list[dict[str, Any]], key: str, tail_size: int) -> float | None:
    vals = [float(row[key]) for row in history if key in row]
    if not vals:
        return None
    tail = vals[-tail_size:]
    return float(sum(tail) / len(tail))


def _max_value(history: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in history if key in row]
    if not vals:
        return None
    return float(max(vals))


def _sum_value(history: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row[key]) for row in history if key in row))


def _exp_safe(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 80:
        return float("inf")
    return float(math.exp(value))


def _summarize_run(name: str, payload: dict[str, Any], tail_size: int) -> dict[str, Any]:
    history = payload["history"]
    status = payload.get("status", "unknown")

    final_train_loss, final_train_step = _last_value(history, "train/loss")
    final_valid_loss, final_valid_step = _last_value(history, "valid/loss")
    best_valid_loss, best_valid_step = _best_value(history, "valid/loss")

    step_time_sum = _sum_value(history, "perf/step_time_s")
    summary = {
        "name": name,
        "status": status,
        "steps_recorded": len(history),
        "final_train_loss": final_train_loss,
        "final_train_step": final_train_step,
        "final_valid_loss": final_valid_loss,
        "final_valid_step": final_valid_step,
        "final_valid_ppl": _exp_safe(final_valid_loss),
        "best_valid_loss": best_valid_loss,
        "best_valid_step": best_valid_step,
        "best_valid_ppl": _exp_safe(best_valid_loss),
        "avg_tokens_per_sec_tail": _mean_tail(history, "perf/tokens_per_sec", tail_size),
        "avg_step_time_s_tail": _mean_tail(history, "perf/step_time_s", tail_size),
        "max_memory_gb": _max_value(history, "perf/max_mem_gb"),
        "train_time_s": step_time_sum,
        "train_time_h": step_time_sum / 3600.0,
    }
    return summary


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator / denominator)


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate - baseline)


def _pct_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return float((candidate - baseline) / baseline * 100.0)


def build_report(
    baseline_name: str,
    baseline_payload: dict[str, Any],
    candidate_name: str,
    candidate_payload: dict[str, Any],
    tail_size: int,
) -> dict[str, Any]:
    baseline = _summarize_run(baseline_name, baseline_payload, tail_size=tail_size)
    candidate = _summarize_run(candidate_name, candidate_payload, tail_size=tail_size)

    comparison = {
        "loss_delta_final_valid": _delta(candidate["final_valid_loss"], baseline["final_valid_loss"]),
        "loss_delta_best_valid": _delta(candidate["best_valid_loss"], baseline["best_valid_loss"]),
        "loss_pct_delta_final_valid": _pct_delta(candidate["final_valid_loss"], baseline["final_valid_loss"]),
        "ppl_delta_final_valid": _delta(candidate["final_valid_ppl"], baseline["final_valid_ppl"]),
        "speed_ratio_tokens_per_sec_tail": _ratio(
            candidate["avg_tokens_per_sec_tail"], baseline["avg_tokens_per_sec_tail"]
        ),
        "time_ratio_train_time": _ratio(candidate["train_time_s"], baseline["train_time_s"]),
        "memory_ratio_max_gb": _ratio(candidate["max_memory_gb"], baseline["max_memory_gb"]),
    }

    return {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
    }


def _to_text(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    candidate = report["candidate"]
    comp = report["comparison"]

    lines = [
        f"Baseline: {baseline['name']} (status={baseline['status']})",
        f"  final_valid_loss={baseline['final_valid_loss']}",
        f"  best_valid_loss={baseline['best_valid_loss']} @ step={baseline['best_valid_step']}",
        f"  avg_tok_s_tail={baseline['avg_tokens_per_sec_tail']}",
        f"  max_mem_gb={baseline['max_memory_gb']}",
        f"  train_time_h={baseline['train_time_h']}",
        "",
        f"Candidate: {candidate['name']} (status={candidate['status']})",
        f"  final_valid_loss={candidate['final_valid_loss']}",
        f"  best_valid_loss={candidate['best_valid_loss']} @ step={candidate['best_valid_step']}",
        f"  avg_tok_s_tail={candidate['avg_tokens_per_sec_tail']}",
        f"  max_mem_gb={candidate['max_memory_gb']}",
        f"  train_time_h={candidate['train_time_h']}",
        "",
        "Comparison:",
        f"  loss_delta_final_valid={comp['loss_delta_final_valid']}",
        f"  loss_delta_best_valid={comp['loss_delta_best_valid']}",
        f"  loss_pct_delta_final_valid={comp['loss_pct_delta_final_valid']}",
        f"  ppl_delta_final_valid={comp['ppl_delta_final_valid']}",
        f"  speed_ratio_tokens_per_sec_tail={comp['speed_ratio_tokens_per_sec_tail']}",
        f"  time_ratio_train_time={comp['time_ratio_train_time']}",
        f"  memory_ratio_max_gb={comp['memory_ratio_max_gb']}",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two training metrics JSON files")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-txt", type=Path, default=None)
    parser.add_argument("--baseline-name", type=str, default="dense_baseline")
    parser.add_argument("--candidate-name", type=str, default="moe_candidate")
    parser.add_argument("--tail-size", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    baseline_payload = _read_metrics(args.baseline)
    candidate_payload = _read_metrics(args.candidate)
    report = build_report(
        baseline_name=args.baseline_name,
        baseline_payload=baseline_payload,
        candidate_name=args.candidate_name,
        candidate_payload=candidate_payload,
        tail_size=args.tail_size,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if args.output_txt is not None:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        args.output_txt.write_text(_to_text(report), encoding="utf-8")

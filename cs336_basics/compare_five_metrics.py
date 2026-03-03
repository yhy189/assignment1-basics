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


def _summarize_run(name: str, payload: dict[str, Any], tail_size: int) -> dict[str, Any]:
    history = payload["history"]
    status = payload.get("status", "unknown")

    final_train_loss, final_train_step = _last_value(history, "train/loss")
    final_valid_loss, final_valid_step = _last_value(history, "valid/loss")
    best_valid_loss, best_valid_step = _best_value(history, "valid/loss")

    step_time_sum = _sum_value(history, "perf/step_time_s")
    return {
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


def _pairwise(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
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


def _safe_key(v: float | None) -> float:
    if v is None:
        return float("inf")
    return float(v)


def build_report(
    dense_payload: dict[str, Any],
    moe_payload: dict[str, Any],
    muon_payload: dict[str, Any],
    mla_payload: dict[str, Any],
    mla_moe_payload: dict[str, Any],
    tail_size: int,
) -> dict[str, Any]:
    dense = _summarize_run("dense", dense_payload, tail_size=tail_size)
    moe = _summarize_run("moe", moe_payload, tail_size=tail_size)
    muon = _summarize_run("muon", muon_payload, tail_size=tail_size)
    mla = _summarize_run("mla", mla_payload, tail_size=tail_size)
    mla_moe = _summarize_run("mla_moe", mla_moe_payload, tail_size=tail_size)

    runs = [dense, moe, muon, mla, mla_moe]
    ranking_valid = [r["name"] for r in sorted(runs, key=lambda x: _safe_key(x["final_valid_loss"]))]
    ranking_speed = [r["name"] for r in sorted(runs, key=lambda x: -float(x["avg_tokens_per_sec_tail"] or 0.0))]
    ranking_memory = [r["name"] for r in sorted(runs, key=lambda x: _safe_key(x["max_memory_gb"]))]

    return {
        "runs": {
            "dense": dense,
            "moe": moe,
            "muon": muon,
            "mla": mla,
            "mla_moe": mla_moe,
        },
        "pairwise": {
            "moe_vs_dense": _pairwise(moe, dense),
            "muon_vs_dense": _pairwise(muon, dense),
            "mla_vs_dense": _pairwise(mla, dense),
            "mla_moe_vs_dense": _pairwise(mla_moe, dense),
            "mla_moe_vs_moe": _pairwise(mla_moe, moe),
            "mla_moe_vs_mla": _pairwise(mla_moe, mla),
            "mla_moe_vs_muon": _pairwise(mla_moe, muon),
        },
        "ranking": {
            "final_valid_loss_asc": ranking_valid,
            "avg_tokens_per_sec_desc": ranking_speed,
            "max_memory_gb_asc": ranking_memory,
        },
    }


def _to_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    runs = report["runs"]
    for name in ["dense", "moe", "muon", "mla", "mla_moe"]:
        run = runs[name]
        lines.extend(
            [
                f"{name.upper()} (status={run['status']})",
                f"  final_valid_loss={run['final_valid_loss']}",
                f"  best_valid_loss={run['best_valid_loss']} @ step={run['best_valid_step']}",
                f"  avg_tok_s_tail={run['avg_tokens_per_sec_tail']}",
                f"  max_mem_gb={run['max_memory_gb']}",
                f"  train_time_h={run['train_time_h']}",
                "",
            ]
        )

    lines.append("Pairwise:")
    for name, pair in report["pairwise"].items():
        lines.append(f"  {name}:")
        lines.append(f"    loss_delta_final_valid={pair['loss_delta_final_valid']}")
        lines.append(f"    loss_pct_delta_final_valid={pair['loss_pct_delta_final_valid']}")
        lines.append(f"    speed_ratio_tokens_per_sec_tail={pair['speed_ratio_tokens_per_sec_tail']}")
        lines.append(f"    memory_ratio_max_gb={pair['memory_ratio_max_gb']}")
        lines.append(f"    time_ratio_train_time={pair['time_ratio_train_time']}")
    lines.append("")

    rank = report["ranking"]
    lines.append(f"Ranking(final_valid_loss asc): {rank['final_valid_loss_asc']}")
    lines.append(f"Ranking(avg_tokens_per_sec desc): {rank['avg_tokens_per_sec_desc']}")
    lines.append(f"Ranking(max_memory_gb asc): {rank['max_memory_gb_asc']}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dense/moe/muon/mla/mla_moe metrics JSON files")
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--moe", type=Path, required=True)
    parser.add_argument("--muon", type=Path, required=True)
    parser.add_argument("--mla", type=Path, required=True)
    parser.add_argument("--mla-moe", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-txt", type=Path, default=None)
    parser.add_argument("--tail-size", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dense_payload = _read_metrics(args.dense)
    moe_payload = _read_metrics(args.moe)
    muon_payload = _read_metrics(args.muon)
    mla_payload = _read_metrics(args.mla)
    mla_moe_payload = _read_metrics(args.mla_moe)
    report = build_report(
        dense_payload=dense_payload,
        moe_payload=moe_payload,
        muon_payload=muon_payload,
        mla_payload=mla_payload,
        mla_moe_payload=mla_moe_payload,
        tail_size=args.tail_size,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if args.output_txt is not None:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        args.output_txt.write_text(_to_text(report), encoding="utf-8")

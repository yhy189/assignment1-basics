from __future__ import annotations

import argparse
import json
import os
import time
import typing

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from jaxtyping import Int

try:
    import wandb
except Exception:
    wandb = None

try:
    from .module import TransformerLM
    from .moe import MoETransformerLM
    from .optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
except ImportError:
    from module import TransformerLM
    from moe import MoETransformerLM
    from optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping


def get_batch(
    x: Int[npt.NDArray, "length"],
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Int[torch.Tensor, "batch_size context_length"], Int[torch.Tensor, "batch_size context_length"]]:
    """Sample next-token prediction batches from a 1D token-id array."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if x.shape[0] <= context_length:
        raise ValueError("dataset is too small for context_length")

    indices = np.random.randint(0, x.shape[0] - context_length, size=(batch_size,))
    inputs = np.stack([x[i : i + context_length] for i in indices])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in indices])

    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs_tensor, targets_tensor


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


class TrainConfig(typing.TypedDict):
    device: torch.device
    dtype: torch.dtype

    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float

    lr: float
    lr_min: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    max_grad_norm: float

    token_ids_path: str | os.PathLike
    checkpoint_dir: str | os.PathLike
    val_token_ids_path: typing.NotRequired[str | os.PathLike | None]
    metrics_json_path: typing.NotRequired[str | os.PathLike | None]

    batch_size: int
    micro_batch_size: typing.NotRequired[int]
    total_tokens: int
    validation_interval: int
    checkpoint_interval: int
    eval_batches: typing.NotRequired[int]

    use_wandb: typing.NotRequired[bool]
    wandb_project: str
    wandb_name: str

    model_type: typing.NotRequired[str]
    num_experts: typing.NotRequired[int]
    moe_top_k: typing.NotRequired[int]
    moe_fixed_expert_idx: typing.NotRequired[int]
    moe_expert_d_ffs: typing.NotRequired[list[int]]
    moe_aux_loss_coef: typing.NotRequired[float]
    moe_z_loss_coef: typing.NotRequired[float]
    resume_checkpoint: typing.NotRequired[str | os.PathLike | None]


TinyStoriesConfig = TrainConfig(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float32,
    vocab_size=10000,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    rope_theta=10000,
    lr=3e-4,
    lr_min=3e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    max_grad_norm=1.0,
    token_ids_path="../data/TinyStoriesV2-GPT4-train/token_ids.npy",
    checkpoint_dir="../data/checkpoints/tiny_stories",
    val_token_ids_path=None,
    metrics_json_path=None,
    batch_size=128,
    micro_batch_size=128,
    total_tokens=327_680_000,
    validation_interval=10,
    checkpoint_interval=1000,
    eval_batches=16,
    use_wandb=False,
    wandb_project="cs336",
    wandb_name="tiny_stories_dense_baseline",
)


OpenWebTextBaselineConfig = TrainConfig(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float32,
    vocab_size=50257,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=8,
    d_ff=2048,
    rope_theta=10000,
    lr=3e-4,
    lr_min=3e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    max_grad_norm=1.0,
    token_ids_path="/root/autodl-tmp/data/openwebtext/openwebtext_train_tokens.npy",
    checkpoint_dir="/root/autodl-tmp/data/openwebtext/checkpoints/dense_baseline",
    val_token_ids_path="/root/autodl-tmp/data/openwebtext/openwebtext_valid_tokens.npy",
    metrics_json_path="/root/autodl-tmp/data/openwebtext/metrics/dense_baseline_metrics.json",
    batch_size=32,
    micro_batch_size=32,
    total_tokens=100_000_000,
    validation_interval=20,
    checkpoint_interval=500,
    eval_batches=16,
    use_wandb=False,
    wandb_project="cs336-baseline",
    wandb_name="openwebtext_dense_baseline",
    model_type="dense",
)


OpenWebTextMoEConfig = TrainConfig(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float32,
    vocab_size=50257,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=8,
    d_ff=2048,
    rope_theta=10000,
    lr=3e-4,
    lr_min=3e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    max_grad_norm=1.0,
    token_ids_path="/root/autodl-tmp/data/openwebtext/openwebtext_train_tokens.npy",
    checkpoint_dir="/root/autodl-tmp/data/openwebtext/checkpoints/moe_baseline",
    val_token_ids_path="/root/autodl-tmp/data/openwebtext/openwebtext_valid_tokens.npy",
    metrics_json_path="/root/autodl-tmp/data/openwebtext/metrics/moe_baseline_metrics.json",
    batch_size=32,
    micro_batch_size=16,
    total_tokens=100_000_000,
    validation_interval=20,
    checkpoint_interval=500,
    eval_batches=16,
    use_wandb=False,
    wandb_project="cs336-moe",
    wandb_name="openwebtext_moe_baseline",
    model_type="moe",
    num_experts=5,
    moe_top_k=2,
    moe_fixed_expert_idx=0,
    moe_expert_d_ffs=[2048, 1024, 1024, 1024, 1024],
    moe_aux_loss_coef=1e-2,
    moe_z_loss_coef=1e-4,
)


def _evaluate_loss(
    lm: torch.nn.Module,
    token_ids: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    eval_batches: int,
) -> float:
    lm.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(eval_batches):
            inputs, targets = get_batch(
                token_ids,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            logits = lm(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            losses.append(loss.item())
    lm.train()
    return float(np.mean(losses))


def _jsonify_config(config: TrainConfig) -> dict[str, typing.Any]:
    out: dict[str, typing.Any] = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (tuple, list)):
            out[key] = [str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v for v in value]
        else:
            out[key] = str(value)
    return out


def _dump_metrics_json(
    metrics_json_path: str | os.PathLike,
    run_meta: dict[str, typing.Any],
    history: list[dict[str, float | int]],
    status: str,
) -> None:
    payload = {
        "run_meta": run_meta,
        "status": status,
        "history": history,
    }

    dirpath = os.path.dirname(os.fspath(metrics_json_path))
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp_path = f"{metrics_json_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, metrics_json_path)


def train(config: TrainConfig) -> None:
    model_type = str(config.get("model_type", "dense"))
    if model_type == "dense":
        lm = TransformerLM(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=config["rope_theta"],
            device=config["device"],
            dtype=config["dtype"],
        )
    elif model_type == "moe":
        lm = MoETransformerLM(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=config["rope_theta"],
            num_experts=int(config.get("num_experts", 5)),
            moe_top_k=int(config.get("moe_top_k", 1)),
            moe_fixed_expert_idx=int(config.get("moe_fixed_expert_idx", 0)),
            moe_expert_d_ffs=typing.cast(list[int] | None, config.get("moe_expert_d_ffs")),
            moe_aux_loss_coef=float(config.get("moe_aux_loss_coef", 1e-2)),
            moe_z_loss_coef=float(config.get("moe_z_loss_coef", 1e-4)),
            device=config["device"],
            dtype=config["dtype"],
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    adamw = AdamW(
        lm.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=config["betas"],
        eps=config["eps"],
    )

    token_ids = np.load(config["token_ids_path"], mmap_mode="r")

    val_token_ids = None
    if config.get("val_token_ids_path"):
        val_token_ids = np.load(config["val_token_ids_path"], mmap_mode="r")

    micro_batch_size = int(config.get("micro_batch_size", config["batch_size"]))
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if config["batch_size"] % micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    grad_accum_steps = config["batch_size"] // micro_batch_size

    total_steps = config["total_tokens"] // config["batch_size"] // config["context_length"]
    start_step = 0
    resume_checkpoint = config.get("resume_checkpoint")
    if resume_checkpoint:
        start_step = load_checkpoint(resume_checkpoint, lm, adamw)
        print(f"Resumed checkpoint: {resume_checkpoint} (step={start_step})")
        if start_step >= total_steps:
            raise ValueError(
                f"resume step ({start_step}) >= total_steps ({total_steps}); increase total_tokens"
            )

    print(f"Training tokens: {config['total_tokens']:,}")
    print(f"Total steps: {total_steps:,}")
    print(f"Start step: {start_step:,}")
    print(f"Train token file: {config['token_ids_path']}")
    print(f"Batch size: {config['batch_size']} (micro={micro_batch_size}, accum={grad_accum_steps})")
    if val_token_ids is not None:
        print(f"Valid token file: {config['val_token_ids_path']}")

    use_wandb = bool(config.get("use_wandb", False))
    metrics_json_path = config.get("metrics_json_path")
    wandb_run = None
    if use_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed/importable but use_wandb=True")
        wandb_run = wandb.init(
            project=config["wandb_project"],
            name=config["wandb_name"],
            config={**config, "total_steps": total_steps},
        )

    eval_batches = int(config.get("eval_batches", 16))
    metrics_history: list[dict[str, float | int]] = []
    if metrics_json_path and start_step > 0 and os.path.exists(metrics_json_path):
        try:
            with open(metrics_json_path, "r", encoding="utf-8") as f:
                old_payload = json.load(f)
            old_history = old_payload.get("history", [])
            if isinstance(old_history, list):
                metrics_history = [
                    row for row in old_history if isinstance(row, dict) and int(row.get("step", 0)) <= start_step
                ]
                print(f"Loaded {len(metrics_history)} existing metric rows from {metrics_json_path}")
        except Exception as ex:
            print(f"Warning: failed to load previous metrics history: {ex}")
    run_meta = {
        "started_unix": time.time(),
        "total_steps": int(total_steps),
        "start_step": int(start_step),
        "resume_checkpoint": os.fspath(resume_checkpoint) if resume_checkpoint else None,
        "config": _jsonify_config(config),
    }
    if metrics_json_path:
        print(f"Metrics JSON file: {metrics_json_path}")

    for step in range(start_step, total_steps):
        if isinstance(config["device"], torch.device) and config["device"].type == "cuda":
            torch.cuda.reset_peak_memory_stats(config["device"])

        step_start = time.perf_counter()
        adamw.zero_grad()
        step_main_loss = 0.0
        step_moe_aux_loss = 0.0

        for _ in range(grad_accum_steps):
            inputs, targets = get_batch(
                token_ids,
                batch_size=micro_batch_size,
                context_length=config["context_length"],
                device=config["device"],
            )

            logits = lm(inputs)
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            moe_aux_loss = torch.zeros((), dtype=main_loss.dtype, device=main_loss.device)
            if hasattr(lm, "get_aux_loss"):
                aux_val = lm.get_aux_loss()  # type: ignore[attr-defined]
                if isinstance(aux_val, torch.Tensor):
                    moe_aux_loss = aux_val.to(main_loss.dtype)

            micro_loss = main_loss + moe_aux_loss
            (micro_loss / grad_accum_steps).backward()

            step_main_loss += float(main_loss.item())
            step_moe_aux_loss += float(moe_aux_loss.item())

            del inputs, targets, logits, main_loss, moe_aux_loss, micro_loss

        gradient_clipping(lm.parameters(), config["max_grad_norm"])

        lr = get_lr_cosine_schedule(
            step,
            lr_max=config["lr"],
            lr_min=config["lr_min"],
            T_w=max(1, total_steps // 10),
            T_c=max(1, total_steps),
        )
        for param_group in adamw.param_groups:
            param_group["lr"] = lr

        adamw.step()

        step_time_s = time.perf_counter() - step_start
        tokens_per_step = config["batch_size"] * config["context_length"]
        tokens_per_sec = tokens_per_step / max(step_time_s, 1e-9)

        step_main_loss /= grad_accum_steps
        step_moe_aux_loss /= grad_accum_steps
        step_total_loss = step_main_loss + step_moe_aux_loss

        max_mem_gb = 0.0
        if isinstance(config["device"], torch.device) and config["device"].type == "cuda":
            max_mem_gb = torch.cuda.max_memory_allocated(config["device"]) / (1024**3)

        metrics = {
            "train/loss": step_total_loss,
            "train/main_loss": step_main_loss,
            "train/moe_aux_loss": step_moe_aux_loss,
            "train/lr": lr,
            "perf/step_time_s": step_time_s,
            "perf/tokens_per_sec": tokens_per_sec,
            "perf/max_mem_gb": max_mem_gb,
        }

        do_validate = (step + 1) % config["validation_interval"] == 0
        if do_validate and val_token_ids is not None:
            metrics["valid/loss"] = _evaluate_loss(
                lm,
                val_token_ids,
                batch_size=micro_batch_size,
                context_length=config["context_length"],
                device=config["device"],
                eval_batches=eval_batches,
            )

        if do_validate:
            valid_str = f", val_loss={metrics['valid/loss']:.4f}" if "valid/loss" in metrics else ""
            aux_str = f", aux_loss={metrics['train/moe_aux_loss']:.4f}"
            print(
                f"step={step+1}/{total_steps} "
                f"loss={metrics['train/loss']:.4f}{aux_str}{valid_str} "
                f"lr={metrics['train/lr']:.3e} "
                f"tok/s={metrics['perf/tokens_per_sec']:.1f} "
                f"max_mem={metrics['perf/max_mem_gb']:.2f}GB"
            )

        metrics_record: dict[str, float | int] = {"step": int(step + 1)}
        metrics_record.update({k: float(v) for k, v in metrics.items()})
        metrics_history.append(metrics_record)

        if use_wandb and wandb_run is not None:
            wandb_run.log(metrics, step=step)

        if metrics_json_path and (do_validate or ((step + 1) % config["checkpoint_interval"] == 0)):
            _dump_metrics_json(
                metrics_json_path=metrics_json_path,
                run_meta=run_meta,
                history=metrics_history,
                status="running",
            )

        if (step + 1) % config["checkpoint_interval"] == 0:
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
            checkpoint_path = os.path.join(config["checkpoint_dir"], f"checkpoint_step_{step + 1}.pt")
            save_checkpoint(lm, adamw, step + 1, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            if isinstance(config["device"], torch.device) and config["device"].type == "cuda":
                torch.cuda.empty_cache()

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["checkpoint_dir"], "checkpoint_final.pt")
    save_checkpoint(lm, adamw, total_steps, checkpoint_path)
    print(f"Saved final checkpoint to {checkpoint_path}")

    if metrics_json_path:
        _dump_metrics_json(
            metrics_json_path=metrics_json_path,
            run_meta=run_meta,
            history=metrics_history,
            status="completed",
        )
        print(f"Saved metrics JSON to {metrics_json_path}")

    if use_wandb and wandb_run is not None:
        wandb_run.finish()


def _build_config(name: str) -> TrainConfig:
    if name == "tinystories":
        return typing.cast(TrainConfig, dict(TinyStoriesConfig))
    if name == "owt":
        return typing.cast(TrainConfig, dict(OpenWebTextBaselineConfig))
    if name == "owt_moe":
        return typing.cast(TrainConfig, dict(OpenWebTextMoEConfig))
    raise ValueError(f"Unknown config: {name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer (dense or MoE)")
    parser.add_argument("--config", choices=["tinystories", "owt", "owt_moe"], default="owt")
    parser.add_argument("--token-ids-path", type=str, default=None)
    parser.add_argument("--val-token-ids-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--total-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--moe-top-k", type=int, default=None)
    parser.add_argument("--moe-fixed-expert-idx", type=int, default=None)
    parser.add_argument(
        "--moe-expert-d-ffs",
        type=str,
        default=None,
        help="Comma-separated expert d_ff values, e.g. 2048,1024,1024,1024,1024",
    )
    parser.add_argument("--moe-aux-loss-coef", type=float, default=None)
    parser.add_argument("--moe-z-loss-coef", type=float, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="e.g., cuda, cuda:0, cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = _build_config(args.config)

    if args.token_ids_path:
        config["token_ids_path"] = args.token_ids_path
    if args.val_token_ids_path is not None:
        config["val_token_ids_path"] = args.val_token_ids_path
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.total_tokens is not None:
        config["total_tokens"] = args.total_tokens
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.micro_batch_size is not None:
        config["micro_batch_size"] = args.micro_batch_size
    if args.context_length is not None:
        config["context_length"] = args.context_length
    if args.eval_batches is not None:
        config["eval_batches"] = args.eval_batches
    if args.metrics_json is not None:
        config["metrics_json_path"] = args.metrics_json
    if args.num_experts is not None:
        config["num_experts"] = args.num_experts
    if args.moe_top_k is not None:
        config["moe_top_k"] = args.moe_top_k
    if args.moe_fixed_expert_idx is not None:
        config["moe_fixed_expert_idx"] = args.moe_fixed_expert_idx
    if args.moe_expert_d_ffs is not None:
        parsed = [int(x.strip()) for x in args.moe_expert_d_ffs.split(",") if x.strip()]
        config["moe_expert_d_ffs"] = parsed
    if args.moe_aux_loss_coef is not None:
        config["moe_aux_loss_coef"] = args.moe_aux_loss_coef
    if args.moe_z_loss_coef is not None:
        config["moe_z_loss_coef"] = args.moe_z_loss_coef
    if args.resume_checkpoint is not None:
        config["resume_checkpoint"] = args.resume_checkpoint

    if args.device:
        config["device"] = torch.device(args.device)

    config["use_wandb"] = bool(args.wandb)

    train(config)

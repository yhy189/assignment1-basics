from __future__ import annotations

import argparse
import os
import typing

import tiktoken
import torch

try:
    from .mla import MLATransformerLM
    from .mla_moe import MLAMoETransformerLM
    from .module import TransformerLM
    from .moe import MoETransformerLM
    from .train import (
        OpenWebTextBaselineConfig,
        OpenWebTextMLAConfig,
        OpenWebTextMLAMoEConfig,
        OpenWebTextMoEConfig,
        OpenWebTextMuonConfig,
    )
except ImportError:
    from mla import MLATransformerLM
    from mla_moe import MLAMoETransformerLM
    from module import TransformerLM
    from moe import MoETransformerLM
    from train import (
        OpenWebTextBaselineConfig,
        OpenWebTextMLAConfig,
        OpenWebTextMLAMoEConfig,
        OpenWebTextMoEConfig,
        OpenWebTextMuonConfig,
    )


def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = probs.sort(descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    cutoff = cumulative_probs > top_p
    cutoff[..., 0] = False
    sorted_probs[cutoff] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_idx)


def _build_model(model_name: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, typing.Any]]:
    if model_name == "dense":
        cfg = dict(OpenWebTextBaselineConfig)
        cfg["device"] = device
        model = TransformerLM(
            vocab_size=cfg["vocab_size"],
            context_length=cfg["context_length"],
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            rope_theta=cfg["rope_theta"],
            activation_checkpointing=False,
            device=device,
            dtype=torch.float32,
        )
        return model, cfg

    if model_name == "muon":
        cfg = dict(OpenWebTextMuonConfig)
        cfg["device"] = device
        model = TransformerLM(
            vocab_size=cfg["vocab_size"],
            context_length=cfg["context_length"],
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            rope_theta=cfg["rope_theta"],
            activation_checkpointing=False,
            device=device,
            dtype=torch.float32,
        )
        return model, cfg

    if model_name == "moe":
        cfg = dict(OpenWebTextMoEConfig)
        cfg["device"] = device
        model = MoETransformerLM(
            vocab_size=cfg["vocab_size"],
            context_length=cfg["context_length"],
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            rope_theta=cfg["rope_theta"],
            num_experts=int(cfg.get("num_experts", 5)),
            moe_top_k=int(cfg.get("moe_top_k", 2)),
            moe_fixed_expert_idx=int(cfg.get("moe_fixed_expert_idx", 0)),
            moe_expert_d_ffs=typing.cast(list[int] | None, cfg.get("moe_expert_d_ffs")),
            moe_aux_loss_coef=float(cfg.get("moe_aux_loss_coef", 1e-2)),
            moe_z_loss_coef=float(cfg.get("moe_z_loss_coef", 1e-4)),
            device=device,
            dtype=torch.float32,
        )
        return model, cfg

    if model_name == "mla":
        cfg = dict(OpenWebTextMLAConfig)
        cfg["device"] = device
        model = MLATransformerLM(
            vocab_size=cfg["vocab_size"],
            context_length=cfg["context_length"],
            d_model=cfg["d_model"],
            latent_d_model=int(cfg.get("mla_d_model", cfg["d_model"] // 2)),
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            rope_theta=cfg["rope_theta"],
            activation_checkpointing=False,
            device=device,
            dtype=torch.float32,
        )
        return model, cfg

    if model_name == "mla_moe":
        cfg = dict(OpenWebTextMLAMoEConfig)
        cfg["device"] = device
        model = MLAMoETransformerLM(
            vocab_size=cfg["vocab_size"],
            context_length=cfg["context_length"],
            d_model=cfg["d_model"],
            latent_d_model=int(cfg.get("mla_d_model", cfg["d_model"] // 2)),
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            rope_theta=cfg["rope_theta"],
            num_experts=int(cfg.get("num_experts", 5)),
            moe_top_k=int(cfg.get("moe_top_k", 2)),
            moe_fixed_expert_idx=int(cfg.get("moe_fixed_expert_idx", 0)),
            moe_expert_d_ffs=typing.cast(list[int] | None, cfg.get("moe_expert_d_ffs")),
            moe_aux_loss_coef=float(cfg.get("moe_aux_loss_coef", 1e-2)),
            moe_z_loss_coef=float(cfg.get("moe_z_loss_coef", 1e-4)),
            activation_checkpointing=False,
            device=device,
            dtype=torch.float32,
        )
        return model, cfg

    raise ValueError(f"Unsupported model name: {model_name}")


def generate(
    model: torch.nn.Module,
    context_length: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    device: torch.device,
    stop_token: str | None = None,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not (0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    enc = tiktoken.get_encoding("gpt2")
    input_ids = enc.encode_ordinary(prompt)
    if not input_ids:
        raise ValueError("Prompt is empty after tokenization")

    tokens = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    stop_token_id = None
    if stop_token:
        stop_ids = enc.encode_ordinary(stop_token)
        if len(stop_ids) == 1:
            stop_token_id = int(stop_ids[0])

    with torch.no_grad():
        for _ in range(max_new_tokens):
            model_input = tokens[:, -context_length:]
            logits = model(model_input)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            if top_p < 1.0:
                next_token = _sample_top_p(probs, top_p)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            if stop_token_id is not None and int(next_token.item()) == stop_token_id:
                break
            tokens = torch.cat([tokens, next_token], dim=1)

    return enc.decode(tokens[0].tolist())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenWebText model inference from checkpoint")
    parser.add_argument("--model", choices=["dense", "moe", "muon", "mla", "mla_moe"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="long long ago")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-token", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/cuda:0; default auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = _build_model(args.model, device=device)
    checkpoint_path = os.fspath(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    output = generate(
        model=model,
        context_length=int(cfg["context_length"]),
        prompt=args.prompt,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        seed=int(args.seed),
        device=device,
        stop_token=args.stop_token,
    )

    print(f"model={args.model}")
    print(f"checkpoint={checkpoint_path}")
    print(f"prompt={args.prompt!r}")
    print(f"seed={args.seed}, temperature={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens}")
    print("----- output -----")
    print(output)


if __name__ == "__main__":
    main()

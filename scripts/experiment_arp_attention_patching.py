"""Patch-aware attention extraction for AR(p) UVModel checkpoints.

This script supports patch_size > 1.
It reports attention at two levels:

1. Patch level: each context token is a patch covering a range of lags.
2. Lag-distributed level: a patch's attention is distributed uniformly over
   the real, non-padding lags inside that patch. 
   
Typical usage:
python scripts\\experiment_arp_attention_patching.py ^
  --checkpoint outputs_patching\\ar3\\patch4_stride4\\sanity\\ar3_patch4_stride4_rope_seed7_best_model.pt ^
  --eval-n 4000 ^
  --device cpu ^
  --output-dir outputs_patching\\ar3\\patch4_stride4\\attention ^
  --run-name ar3_patch4_stride4_attention
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from playground.data.ar import ar_batch_to_uvmodel, generate_arp_batch
from playground.model.registry.uv import UVModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch-aware horizon-token attention for AR(p) UVModel checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-n", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-ar-order", type=int, default=None, help="Override/evaluate on a specific AR order; required for mixed checkpoints")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-lag-plot", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--run-name", type=str, default="ar_attention_patching")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(device_arg)


def as_namespace(d: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**d)


def get_arg(train_args: SimpleNamespace, name: str, default: Any) -> Any:
    return getattr(train_args, name, default)


def train_order_label(train_args: SimpleNamespace) -> str:
    return str(get_arg(train_args, "ar_order", 1))


def resolve_test_order(args: argparse.Namespace, train_args: SimpleNamespace) -> int:
    if args.test_ar_order is not None:
        return int(args.test_ar_order)
    label = train_order_label(train_args).lower()
    if label == "mixed" or label.startswith("mixed"):
        raise ValueError("--test-ar-order is required when checkpoint was trained with mixed orders")
    return int(label)


def build_model(train_args: SimpleNamespace) -> UVModel:
    return UVModel(
        d_model=train_args.d_model,
        d_ff=train_args.d_ff,
        d_kv=train_args.d_kv,
        n_heads=train_args.n_heads,
        dropout=train_args.dropout,
        activation_fn=train_args.activation,
        n_quantiles=9,
        n_encoder_layers=train_args.n_layers,
        pred_length=1,
        use_arcsinh=train_args.use_arcsinh,
        use_rope=train_args.use_rope,
        context_length=train_args.context_length,
        patch_size=train_args.patch_size,
        patch_stride=train_args.patch_stride,
    )


def attach_attention_capture(model: UVModel) -> list[tuple[int, torch.Tensor]]:
    captured: list[tuple[int, torch.Tensor]] = []
    for layer_idx, layer in enumerate(model.encoder):
        mha = layer.mha

        def wrapped_attention(q, k, v, mask=None, *, layer_idx=layer_idx):
            scores = torch.matmul(q, k.transpose(-2, -1))
            if mask is not None:
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, float("-inf"))
                else:
                    scores = scores + mask
            weights = torch.softmax(scores, dim=-1)
            captured.append((layer_idx, weights.detach().cpu()))
            return torch.matmul(weights, v)

        mha._attention = wrapped_attention
    return captured


@torch.no_grad()
def median_forecast(model: UVModel, context: torch.Tensor, true_horizon: torch.Tensor) -> torch.Tensor:
    model.eval()
    _, quantile_forecast = model(context=context, n_horizon=1, true_horizon=true_horizon)
    return quantile_forecast[:, 4, 0]


def context_patch_lag_ranges(context_length: int, patch_size: int, patch_stride: int) -> list[list[int]]:
    """Return lags covered by each context patch token, in token order oldest->newest."""
    pad = (patch_size - (context_length % patch_size)) % patch_size
    padded_len = context_length + pad
    ranges: list[list[int]] = []
    for start in range(0, padded_len - patch_size + 1, patch_stride):
        lags: list[int] = []
        for padded_pos in range(start, start + patch_size):
            orig_pos = padded_pos - pad
            if 0 <= orig_pos < context_length:
                lag = context_length - orig_pos
                lags.append(lag)
        ranges.append(sorted(lags))
    return ranges


def patch_label(lags: list[int]) -> str:
    if not lags:
        return "padding"
    if min(lags) == max(lags):
        return f"lag {min(lags)}"
    return f"lags {min(lags)}-{max(lags)}"


def top_p_recall_by_lag(attn_by_lag: torch.Tensor, ar_order: int) -> float:
    top = torch.argsort(attn_by_lag, descending=True)[:ar_order]
    hits = int((top < ar_order).sum().item())
    return hits / ar_order


@torch.no_grad()
def evaluate_attention(
    *,
    model: UVModel,
    train_args: SimpleNamespace,
    test_order: int,
    eval_n: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    captured: list[tuple[int, torch.Tensor]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    pacf_low = float(get_arg(train_args, "pacf_low", -0.9))
    pacf_high = float(get_arg(train_args, "pacf_high", 0.9))

    batch = generate_arp_batch(
        batch_size=eval_n,
        context_length=train_args.context_length,
        ar_order=test_order,
        pacf_low=pacf_low,
        pacf_high=pacf_high,
        noise_std=train_args.noise_std,
        burn_in=train_args.burn_in,
        device=device,
    )
    context, true_horizon, conditional_mean_uv = ar_batch_to_uvmodel(batch)
    conditional_mean = conditional_mean_uv[:, 0, 0]

    patch_ranges = context_patch_lag_ranges(train_args.context_length, train_args.patch_size, train_args.patch_stride)
    n_ctx_patches = len(patch_ranges)
    n_layers = train_args.n_layers

    norm_sums = torch.zeros(n_layers, n_ctx_patches)
    raw_sums = torch.zeros(n_layers, n_ctx_patches)
    context_mass_sums = torch.zeros(n_layers)
    self_mass_sums = torch.zeros(n_layers)
    counts = torch.zeros(n_layers)

    all_preds: list[torch.Tensor] = []
    for start in range(0, eval_n, batch_size):
        end = min(start + batch_size, eval_n)
        captured.clear()
        pred = median_forecast(model, context[start:end], true_horizon[start:end])
        all_preds.append(pred.detach().cpu())

        if len(captured) != n_layers:
            raise RuntimeError(f"Expected {n_layers} captured attention tensors, got {len(captured)}")

        for layer_idx, weights in captured:
            horizon_to_context = weights[:, :, -1, :n_ctx_patches]
            context_mass = horizon_to_context.sum(dim=-1)
            context_norm = horizon_to_context / context_mass.unsqueeze(-1).clamp_min(1e-8)
            raw_sums[layer_idx] += horizon_to_context.sum(dim=(0, 1))
            norm_sums[layer_idx] += context_norm.sum(dim=(0, 1))
            context_mass_sums[layer_idx] += context_mass.sum()
            self_mass_sums[layer_idx] += weights[:, :, -1, -1].sum()
            counts[layer_idx] += horizon_to_context.shape[0] * horizon_to_context.shape[1]

    base_pred = torch.cat(all_preds).to(conditional_mean.device)
    base_mse = float((base_pred - conditional_mean).square().mean().cpu())

    patch_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    per_layer: list[dict[str, Any]] = []

    for layer_idx in range(n_layers):
        mean_norm_by_patch = norm_sums[layer_idx] / counts[layer_idx].clamp_min(1)
        mean_raw_by_patch = raw_sums[layer_idx] / counts[layer_idx].clamp_min(1)

        lag_distributed = torch.zeros(train_args.context_length)
        lag_raw_distributed = torch.zeros(train_args.context_length)
        for patch_idx, lags in enumerate(patch_ranges):
            if not lags:
                continue
            share = mean_norm_by_patch[patch_idx] / len(lags)
            raw_share = mean_raw_by_patch[patch_idx] / len(lags)
            for lag in lags:
                lag_distributed[lag - 1] += share
                lag_raw_distributed[lag - 1] += raw_share

        if lag_distributed.sum() > 0:
            lag_distributed = lag_distributed / lag_distributed.sum()
        if lag_raw_distributed.sum() > 0:
            lag_raw_distributed = lag_raw_distributed / lag_raw_distributed.sum()

        top_lag = int(torch.argmax(lag_distributed).item() + 1)
        sorted_lag_idx = torch.argsort(lag_distributed, descending=True)
        lag1_rank = int((sorted_lag_idx == 0).nonzero(as_tuple=False)[0].item() + 1)
        first_p_lag_mass = float(lag_distributed[:test_order].sum().item())
        top_p_lag_recall = top_p_recall_by_lag(lag_distributed, test_order)

        top_patch_idx = int(torch.argmax(mean_norm_by_patch).item())
        top_patch_lags = patch_ranges[top_patch_idx]
        per_layer.append(
            {
                "layer": layer_idx,
                "top_lag_by_distributed_attention": top_lag,
                "lag1_rank_by_distributed_attention": lag1_rank,
                "first_p_distributed_attention_mass": first_p_lag_mass,
                "top_p_recall_by_distributed_attention": top_p_lag_recall,
                "top_patch_index_oldest_to_newest": top_patch_idx + 1,
                "top_patch_label": patch_label(top_patch_lags),
                "top_patch_min_lag": min(top_patch_lags) if top_patch_lags else None,
                "top_patch_max_lag": max(top_patch_lags) if top_patch_lags else None,
                "mean_context_attention_mass": float((context_mass_sums[layer_idx] / counts[layer_idx].clamp_min(1)).item()),
                "mean_horizon_self_attention_mass": float((self_mass_sums[layer_idx] / counts[layer_idx].clamp_min(1)).item()),
            }
        )

        for patch_idx, lags in enumerate(patch_ranges):
            overlaps_true_order = any(lag <= test_order for lag in lags)
            patch_rows.append(
                {
                    "test_ar_order": test_order,
                    "layer": layer_idx,
                    "patch_index_oldest_to_newest": patch_idx + 1,
                    "patch_label": patch_label(lags),
                    "patch_min_lag": min(lags) if lags else "",
                    "patch_max_lag": max(lags) if lags else "",
                    "patch_overlaps_ar_order": int(overlaps_true_order),
                    "context_normalized_attention": float(mean_norm_by_patch[patch_idx].item()),
                    "raw_attention": float(mean_raw_by_patch[patch_idx].item()),
                }
            )

        for lag in range(1, train_args.context_length + 1):
            lag_rows.append(
                {
                    "test_ar_order": test_order,
                    "layer": layer_idx,
                    "lag": lag,
                    "is_within_ar_order": int(lag <= test_order),
                    "distributed_context_normalized_attention": float(lag_distributed[lag - 1].item()),
                    "distributed_raw_attention": float(lag_raw_distributed[lag - 1].item()),
                }
            )

    last = per_layer[-1]
    summary = {
        "test_ar_order": test_order,
        "attention_base_mse_to_conditional_mean": base_mse,
        "last_layer_top_lag_by_distributed_attention": last["top_lag_by_distributed_attention"],
        "last_layer_lag1_rank_by_distributed_attention": last["lag1_rank_by_distributed_attention"],
        "last_layer_first_p_distributed_attention_mass": last["first_p_distributed_attention_mass"],
        "last_layer_top_p_recall_by_distributed_attention": last["top_p_recall_by_distributed_attention"],
        "last_layer_top_patch_label": last["top_patch_label"],
        "last_layer_top_patch_min_lag": last["top_patch_min_lag"],
        "last_layer_top_patch_max_lag": last["top_patch_max_lag"],
        "per_layer_summary": per_layer,
    }
    return summary, patch_rows, lag_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_patch_attention(rows: list[dict[str, Any]], path: Path, layer_idx: int, ar_order: int) -> None:
    d = [r for r in rows if int(r["layer"]) == layer_idx]
    d = sorted(d, key=lambda r: int(r["patch_index_oldest_to_newest"]))
    labels = [str(r["patch_label"]) for r in d]
    vals = [float(r["context_normalized_attention"]) for r in d]
    colors = ["#d62728" if int(r["patch_overlaps_ar_order"]) else "#4c78a8" for r in d]

    plt.figure(figsize=(max(8, 0.7 * len(d)), 5))
    plt.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
    plt.title(f"AR({ar_order}) patch-level attention, layer {layer_idx}", fontsize=15, fontweight="bold")
    plt.xlabel("Context patch lag range")
    plt.ylabel("Context-normalized attention")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_lag_attention(rows: list[dict[str, Any]], path: Path, layer_idx: int, ar_order: int, max_lag: int) -> None:
    d = [r for r in rows if int(r["layer"]) == layer_idx and int(r["lag"]) <= max_lag]
    d = sorted(d, key=lambda r: int(r["lag"]))
    lags = [int(r["lag"]) for r in d]
    vals = [float(r["distributed_context_normalized_attention"]) for r in d]
    colors = ["#d62728" if lag <= ar_order else "#4c78a8" for lag in lags]

    plt.figure(figsize=(10, 5))
    plt.bar(lags, vals, color=colors, edgecolor="black", linewidth=0.8)
    plt.axvline(ar_order + 0.5, color="black", linestyle="--", linewidth=1.2, label=f"true p={ar_order}")
    plt.title(f"AR({ar_order}) lag-distributed attention, layer {layer_idx}", fontsize=15, fontweight="bold")
    plt.xlabel("Lag (1 = latest context value). Dashed line = true p")
    plt.ylabel("Distributed attention")
    plt.xticks(lags)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = as_namespace(checkpoint["args"])
    test_order = resolve_test_order(args, train_args)

    if not getattr(train_args, "use_rope", False):
        print("WARNING: checkpoint was trained with use_rope=False; lag/patch interpretation may be weak.")

    model = build_model(train_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    captured = attach_attention_capture(model)

    summary, patch_rows, lag_rows = evaluate_attention(
        model=model,
        train_args=train_args,
        test_order=test_order,
        eval_n=args.eval_n,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        captured=captured,
    )

    output = {
        "checkpoint": str(args.checkpoint),
        "test_ar_order": test_order,
        "device": str(device),
        "eval_n": args.eval_n,
        "context_length": train_args.context_length,
        "patch_size": train_args.patch_size,
        "patch_stride": train_args.patch_stride,
        "summary": summary,
        "attention_patch_rows": patch_rows,
        "attention_lag_distributed_rows": lag_rows,
    }

    json_path = args.output_dir / f"{args.run_name}.json"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.output_dir / f"{args.run_name}_patch_attention.csv", patch_rows)
    write_csv(args.output_dir / f"{args.run_name}_lag_distributed_attention.csv", lag_rows)

    for layer_idx in range(int(train_args.n_layers)):
        patch_plot_path = args.output_dir / f"{args.run_name}_patch_attention_layer{layer_idx}.png"
        lag_plot_path = args.output_dir / f"{args.run_name}_lag_distributed_attention_layer{layer_idx}.png"

        plot_patch_attention(
            patch_rows,
            patch_plot_path,
            layer_idx=layer_idx,
            ar_order=test_order,
        )
        plot_lag_attention(
            lag_rows,
            lag_plot_path,
            layer_idx=layer_idx,
            ar_order=test_order,
            max_lag=min(args.max_lag_plot, train_args.context_length),
        )

    last_layer = int(train_args.n_layers) - 1
    last_patch_plot_path = args.output_dir / f"{args.run_name}_patch_attention_last_layer.png"
    last_lag_plot_path = args.output_dir / f"{args.run_name}_lag_distributed_attention_last_layer.png"
    plot_patch_attention(
        patch_rows,
        last_patch_plot_path,
        layer_idx=last_layer,
        ar_order=test_order,
    )
    plot_lag_attention(
        lag_rows,
        last_lag_plot_path,
        layer_idx=last_layer,
        ar_order=test_order,
        max_lag=min(args.max_lag_plot, train_args.context_length),
    )

    print("Patch-aware attention summary")
    for k, v in summary.items():
        if k != "per_layer_summary":
            print(f"{k}: {v}")
    print(f"Wrote {json_path}")
    for layer_idx in range(int(train_args.n_layers)):
        print(args.output_dir / f"{args.run_name}_patch_attention_layer{layer_idx}.png")
        print(args.output_dir / f"{args.run_name}_lag_distributed_attention_layer{layer_idx}.png")
    print(last_patch_plot_path)
    print(last_lag_plot_path)


if __name__ == "__main__":
    main()

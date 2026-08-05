"""Attention extraction for AR(p) UVModel checkpoints.

For patch_size=1, the forecast horizon token can attend to each context token.
This script extracts that horizon-to-context attention and reports which lags
get the largest attention mass.
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
    parser = argparse.ArgumentParser(description="Extract horizon-token attention by lag from a trained AR(p) UVModel.")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "outputs" / "ar_sanity_uvmodel_best_model.pt")
    parser.add_argument("--eval-n", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--test-ar-order",
        type=int,
        default=None,
        help="Override/evaluate on a specific AR order; required for mixed checkpoints",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--run-name", type=str, default="ar_attention")
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
    """Monkey-patch each MHA._attention to save attention weights.

    The skeleton uses torch.nn.functional.scaled_dot_product_attention, which does
    not return weights. Here we reproduce its eval-time computation with scale=1.0,
    matching the skeleton's MHA implementation.
    """
    captured: list[tuple[int, torch.Tensor]] = []

    for layer_idx, layer in enumerate(model.encoder):
        mha = layer.mha

        def wrapped_attention(q, k, v, mask=None, *, layer_idx=layer_idx):
            scores = torch.matmul(q, k.transpose(-2, -1))  # skeleton uses scale=1.0
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


def save_plot(rows: list[dict[str, float]], path: Path, layer_idx: int, ar_order: int) -> None:
    layer_rows = [r for r in rows if int(r["layer"]) == layer_idx]
    lags = [int(r["lag"]) for r in layer_rows]
    attn = [r["context_normalized_attention"] for r in layer_rows]

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(lags, attn, marker="o")
    if ar_order > 1:
        plt.axvline(ar_order, linestyle="--", linewidth=1.0, label=f"AR order p={ar_order}")
        plt.legend()
    plt.xlabel("Lag: 1 means latest context value")
    plt.ylabel("Context-normalized attention")
    plt.title(f"AR({ar_order}) horizon-token attention by lag, layer {layer_idx}")
    plt.xticks(lags)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def top_p_recall_by_attention(attn_by_lag: torch.Tensor, ar_order: int) -> float:
    top = torch.argsort(attn_by_lag, descending=True)[:ar_order]
    hits = int((top < ar_order).sum().item())
    return hits / ar_order


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = as_namespace(checkpoint["args"])
    train_ar_order = train_order_label(train_args)
    ar_order = resolve_test_order(args, train_args)
    pacf_low = float(get_arg(train_args, "pacf_low", -0.9))
    pacf_high = float(get_arg(train_args, "pacf_high", 0.9))

    if not getattr(train_args, "use_rope", False):
        print("WARNING: checkpoint was trained with use_rope=False; lag interpretation may be weak.")
    if train_args.patch_size != 1 or train_args.patch_stride != 1:
        raise ValueError("This script expects patch_size=1 and patch_stride=1 for direct lag-level attention.")

    model = build_model(train_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    captured = attach_attention_capture(model)

    torch.manual_seed(args.seed)
    batch = generate_arp_batch(
        batch_size=args.eval_n,
        context_length=train_args.context_length,
        ar_order=ar_order,
        pacf_low=pacf_low,
        pacf_high=pacf_high,
        noise_std=train_args.noise_std,
        burn_in=train_args.burn_in,
        device=device,
    )
    context, true_horizon, conditional_mean_uv = ar_batch_to_uvmodel(batch)
    conditional_mean = conditional_mean_uv[:, 0, 0]

    n_layers = train_args.n_layers
    context_length = train_args.context_length
    norm_sums = torch.zeros(n_layers, context_length)
    raw_sums = torch.zeros(n_layers, context_length)
    context_mass_sums = torch.zeros(n_layers)
    self_mass_sums = torch.zeros(n_layers)
    counts = torch.zeros(n_layers)

    all_preds: list[torch.Tensor] = []
    for start in range(0, args.eval_n, args.batch_size):
        end = min(start + args.batch_size, args.eval_n)
        captured.clear()
        pred = median_forecast(model, context[start:end], true_horizon[start:end])
        all_preds.append(pred.detach().cpu())

        if len(captured) != n_layers:
            raise RuntimeError(f"Expected {n_layers} captured attention tensors, got {len(captured)}")

        for layer_idx, weights in captured:
            # weights: (batch, heads, seq, seq). Last token is the one-step horizon token.
            horizon_to_context = weights[:, :, -1, :context_length]
            context_mass = horizon_to_context.sum(dim=-1)
            context_norm = horizon_to_context / context_mass.unsqueeze(-1).clamp_min(1e-8)

            raw_sums[layer_idx] += horizon_to_context.sum(dim=(0, 1))
            norm_sums[layer_idx] += context_norm.sum(dim=(0, 1))
            context_mass_sums[layer_idx] += context_mass.sum()
            self_mass_sums[layer_idx] += weights[:, :, -1, -1].sum()
            counts[layer_idx] += horizon_to_context.shape[0] * horizon_to_context.shape[1]

    base_pred = torch.cat(all_preds).to(conditional_mean.device)
    base_mse = float((base_pred - conditional_mean).square().mean().cpu())

    rows: list[dict[str, float]] = []
    per_layer_summary = []
    for layer_idx in range(n_layers):
        mean_norm_by_pos = norm_sums[layer_idx] / counts[layer_idx].clamp_min(1)
        mean_raw_by_pos = raw_sums[layer_idx] / counts[layer_idx].clamp_min(1)
        mean_norm_by_lag = torch.flip(mean_norm_by_pos, dims=[0])
        mean_raw_by_lag = torch.flip(mean_raw_by_pos, dims=[0])

        top_lag = int(torch.argmax(mean_norm_by_lag).item() + 1)
        sorted_idx = torch.argsort(mean_norm_by_lag, descending=True)
        lag1_rank = int((sorted_idx == 0).nonzero(as_tuple=False)[0].item() + 1)
        first_p_attention = float(mean_norm_by_lag[:ar_order].sum().item())
        top_p_recall = top_p_recall_by_attention(mean_norm_by_lag, ar_order)

        per_layer_summary.append(
            {
                "layer": layer_idx,
                "top_lag_by_context_normalized_attention": top_lag,
                "lag1_rank": lag1_rank,
                "lag1_context_normalized_attention": float(mean_norm_by_lag[0].item()),
                "first_p_context_normalized_attention_mass": first_p_attention,
                "top_p_recall_by_context_normalized_attention": top_p_recall,
                "mean_context_attention_mass": float((context_mass_sums[layer_idx] / counts[layer_idx].clamp_min(1)).item()),
                "mean_horizon_self_attention_mass": float((self_mass_sums[layer_idx] / counts[layer_idx].clamp_min(1)).item()),
            }
        )

        for lag in range(1, context_length + 1):
            rows.append(
                {
                    "layer": float(layer_idx),
                    "lag": float(lag),
                    "is_within_ar_order": float(lag <= ar_order),
                    "context_normalized_attention": float(mean_norm_by_lag[lag - 1].item()),
                    "raw_attention": float(mean_raw_by_lag[lag - 1].item()),
                }
            )

    last = per_layer_summary[-1]
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "seed": args.seed,
        "eval_n": args.eval_n,
        "train_ar_order": train_ar_order,
        "test_ar_order": ar_order,
        "ar_order": ar_order,
        "pacf_low": pacf_low,
        "pacf_high": pacf_high,
        "context_length": context_length,
        "patch_size": train_args.patch_size,
        "patch_stride": train_args.patch_stride,
        "use_rope": train_args.use_rope,
        "use_arcsinh": train_args.use_arcsinh,
        "base_median_mse_to_conditional_mean": base_mse,
        "last_layer_top_lag": last["top_lag_by_context_normalized_attention"],
        "last_layer_lag1_rank": last["lag1_rank"],
        "last_layer_first_p_attention_mass": last["first_p_context_normalized_attention_mass"],
        "last_layer_top_p_recall": last["top_p_recall_by_context_normalized_attention"],
        "per_layer_summary": per_layer_summary,
        "rows": rows,
    }

    if ar_order == 1:
        summary["base_median_mse_to_phi_xt"] = base_mse

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.run_name}.json"
    csv_path = args.output_dir / f"{args.run_name}.csv"
    last_layer_plot_path = args.output_dir / f"{args.run_name}_last_layer.png"

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    layer_plot_paths: list[Path] = []
    for layer_idx in range(n_layers):
        layer_plot_path = args.output_dir / f"{args.run_name}_layer{layer_idx}.png"
        save_plot(rows, layer_plot_path, layer_idx=layer_idx, ar_order=ar_order)
        layer_plot_paths.append(layer_plot_path)

    save_plot(rows, last_layer_plot_path, layer_idx=n_layers - 1, ar_order=ar_order)

    print("Attention summary")
    print(f"checkpoint: {args.checkpoint}")
    print(f"train_ar_order: {train_ar_order}")
    print(f"test_ar_order: {ar_order}")
    print(f"base_median_mse_to_conditional_mean: {base_mse:.6f}")
    for item in per_layer_summary:
        print(
            f"layer {item['layer']} | top_lag={item['top_lag_by_context_normalized_attention']} | "
            f"lag1_rank={item['lag1_rank']} | lag1_attn={item['lag1_context_normalized_attention']:.6f} | "
            f"first_p_mass={item['first_p_context_normalized_attention_mass']:.6f} | "
            f"top_p_recall={item['top_p_recall_by_context_normalized_attention']:.3f} | "
            f"context_mass={item['mean_context_attention_mass']:.6f} | "
            f"self_mass={item['mean_horizon_self_attention_mass']:.6f}"
        )

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    for layer_plot_path in layer_plot_paths:
        print(f"Wrote {layer_plot_path}")
    print(f"Wrote {last_layer_plot_path}")


if __name__ == "__main__":
    main()

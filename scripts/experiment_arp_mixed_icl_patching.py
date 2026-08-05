"""Task 4b evaluation for a patched mixed-order AR checkpoint.

Adds the same forecast and perturbation metrics as the scalar version, while
reporting both true patch-level attention and lag-distributed attention for
every encoder layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from arp_task4_common import (
    as_namespace,
    build_model,
    evaluate_gaussian_strengths_by_lag,
    evaluate_permutation_by_lag,
    first_p_positive_share,
    forecast_metrics,
    gaussian_strength_plot,
    generate_fixed_order_data,
    get_arg,
    median_forecast,
    resolve_device,
    simple_order_bar,
    top_p_recall,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 4b mixed-order patch-aware evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-orders", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--eval-n", type=int, default=4000)
    parser.add_argument("--attention-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seasonality", type=int, default=1)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0, 0.25, 0.5, 1, 2])
    parser.add_argument("--max-lag-plot", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="mixed_ar1_5_patching_task4b")
    return parser.parse_args()


def attach_attention_capture(model):
    captured: list[tuple[int, torch.Tensor]] = []
    for layer_idx, layer in enumerate(model.encoder):
        mha = layer.mha

        def wrapped(q, k, v, mask=None, *, layer_idx=layer_idx):
            scores = torch.matmul(q, k.transpose(-2, -1))
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
            weights = torch.softmax(scores, dim=-1)
            captured.append((layer_idx, weights.detach().cpu()))
            return torch.matmul(weights, v)

        mha._attention = wrapped
    return captured


def patch_lag_ranges(context_length: int, patch_size: int, patch_stride: int) -> list[list[int]]:
    pad = (patch_size - (context_length % patch_size)) % patch_size
    padded_len = context_length + pad
    ranges: list[list[int]] = []
    for start in range(0, padded_len - patch_size + 1, patch_stride):
        lags = []
        for padded_pos in range(start, start + patch_size):
            original_pos = padded_pos - pad
            if 0 <= original_pos < context_length:
                lags.append(context_length - original_pos)
        ranges.append(sorted(lags))
    return ranges


def patch_label(lags: list[int]) -> str:
    if not lags:
        return "padding"
    if min(lags) == max(lags):
        return f"lag {min(lags)}"
    return f"lags {min(lags)}-{max(lags)}"


@torch.no_grad()
def attention_all_layers(model, train_args, context, true_horizon, batch_size, captured, test_order):
    ranges = patch_lag_ranges(train_args.context_length, train_args.patch_size, train_args.patch_stride)
    n_patches = len(ranges)
    n_layers = int(train_args.n_layers)
    patch_sums = torch.zeros(n_layers, n_patches)
    counts = torch.zeros(n_layers)

    for start in range(0, context.shape[0], batch_size):
        end = min(start + batch_size, context.shape[0])
        captured.clear()
        _ = median_forecast(model, context[start:end], true_horizon[start:end])
        if len(captured) != n_layers:
            raise RuntimeError(f"Expected {n_layers} captured tensors, got {len(captured)}")
        for layer, weights in captured:
            h2c = weights[:, :, -1, :n_patches]
            norm = h2c / h2c.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            patch_sums[layer] += norm.sum(dim=(0, 1))
            counts[layer] += h2c.shape[0] * h2c.shape[1]

    patch_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for layer in range(n_layers):
        patch_attention = patch_sums[layer] / counts[layer].clamp_min(1)
        distributed = torch.zeros(train_args.context_length)
        for patch_idx, lags in enumerate(ranges):
            if not lags:
                continue
            share = patch_attention[patch_idx] / len(lags)
            for lag in lags:
                distributed[lag - 1] += share
        distributed = distributed / distributed.sum().clamp_min(1e-8)

        top = torch.argsort(distributed, descending=True)[:test_order]
        summaries.append(
            {
                "test_ar_order": test_order,
                "layer": layer,
                "top_lag_by_distributed_attention": int(torch.argmax(distributed).item() + 1),
                "first_p_distributed_attention_mass": float(distributed[:test_order].sum().item()),
                "top_p_recall_by_distributed_attention": float((top < test_order).sum().item() / test_order),
                "top_patch_label": patch_label(ranges[int(torch.argmax(patch_attention).item())]),
            }
        )
        for patch_idx, lags in enumerate(ranges):
            patch_rows.append(
                {
                    "test_ar_order": test_order,
                    "layer": layer,
                    "patch_index": patch_idx + 1,
                    "patch_label": patch_label(lags),
                    "patch_min_lag": min(lags) if lags else "",
                    "patch_max_lag": max(lags) if lags else "",
                    "patch_overlaps_true_support": int(any(lag <= test_order for lag in lags)),
                    "context_normalized_attention": float(patch_attention[patch_idx].item()),
                }
            )
        for lag in range(1, train_args.context_length + 1):
            lag_rows.append(
                {
                    "test_ar_order": test_order,
                    "layer": layer,
                    "lag": lag,
                    "is_within_ar_order": int(lag <= test_order),
                    "distributed_context_normalized_attention": float(distributed[lag - 1].item()),
                }
            )
    return summaries, patch_rows, lag_rows


def classify_mixed(summaries, min_p, max_p, factor=1.25):
    in_range = [float(s["median_mase_to_noisy_next_x"]) for s in summaries if min_p <= int(s["test_ar_order"]) <= max_p]
    reference = sum(in_range) / len(in_range)
    for s in summaries:
        p = int(s["test_ar_order"])
        mase = float(s["median_mase_to_noisy_next_x"])
        s["mase_over_mean_in_range_mase"] = mase / reference
        if min_p <= p <= max_p:
            s["failure_mode"] = "in_training_range"
        elif mase >= 1.0:
            s["failure_mode"] = "catastrophic"
        elif mase / reference <= factor:
            s["failure_mode"] = "good_extrapolation"
        else:
            s["failure_mode"] = "graceful_degradation"


def plot_lag_attention(rows, orders, layer, max_lag, path):
    fig, axes = plt.subplots(len(orders), 1, figsize=(12, 2.0 * len(orders)), sharex=True)
    if len(orders) == 1:
        axes = [axes]
    for ax, p in zip(axes, orders):
        current = sorted(
            [r for r in rows if int(r["test_ar_order"]) == p and int(r["layer"]) == layer and int(r["lag"]) <= max_lag],
            key=lambda r: int(r["lag"]),
        )
        ax.plot([r["lag"] for r in current], [r["distributed_context_normalized_attention"] for r in current], marker="o")
        ax.axvline(p + 0.5, linestyle="--", linewidth=1.0)
        ax.set_ylabel(f"AR({p})", rotation=0, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Lag (distributed from patch attention); dashed line = true p")
    fig.suptitle(f"Mixed patched checkpoint: lag-distributed attention, layer {layer}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_patch_attention(rows, orders, layer, path):
    fig, axes = plt.subplots(len(orders), 1, figsize=(12, 2.2 * len(orders)))
    if len(orders) == 1:
        axes = [axes]
    for ax, p in zip(axes, orders):
        current = sorted(
            [r for r in rows if int(r["test_ar_order"]) == p and int(r["layer"]) == layer],
            key=lambda r: int(r["patch_index"]),
        )
        ax.bar([r["patch_label"] for r in current], [r["context_normalized_attention"] for r in current], edgecolor="black", linewidth=0.5)
        ax.set_ylabel(f"AR({p})", rotation=0, ha="right")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Mixed patched checkpoint: true patch-level attention, layer {layer}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_permutation(rows, orders, max_lag, path):
    fig, axes = plt.subplots(len(orders), 1, figsize=(12, 2.0 * len(orders)), sharex=True)
    if len(orders) == 1:
        axes = [axes]
    for ax, p in zip(axes, orders):
        current = sorted(
            [r for r in rows if int(r["test_ar_order"]) == p and int(r["lag"]) <= max_lag],
            key=lambda r: int(r["lag"]),
        )
        ax.plot([r["lag"] for r in current], [r["mse_after_permutation"] for r in current], marker="o")
        ax.axvline(p + 0.5, linestyle="--", linewidth=1.0)
        ax.set_ylabel(f"AR({p})", rotation=0, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Permuted scalar lag before patching; dashed line = true p")
    fig.suptitle("Mixed patched checkpoint: MSE after permuting one scalar lag")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = as_namespace(checkpoint["args"])
    model = build_model(train_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    orders = list(args.test_orders)
    summaries = []
    permutation_rows = []
    gaussian_rows = []
    attention_summaries = []
    patch_rows = []
    lag_rows = []
    datasets = {}

    for p in orders:
        batch, context, true_horizon, conditional_mean = generate_fixed_order_data(
            train_args=train_args,
            test_order=p,
            eval_n=args.eval_n,
            seed=args.seed + 100_000 * p,
            device=device,
        )
        metrics, _ = forecast_metrics(
            model=model,
            batch=batch,
            context=context,
            true_horizon=true_horizon,
            conditional_mean=conditional_mean,
            test_order=p,
            seasonality=args.seasonality,
        )
        summary = {"test_ar_order": p, **metrics}
        summaries.append(summary)
        datasets[p] = (context, true_horizon, conditional_mean)

        perm = evaluate_permutation_by_lag(
            model=model,
            context=context,
            true_horizon=true_horizon,
            conditional_mean=conditional_mean,
            test_order=p,
            seed=args.seed + 10_000 + p,
        )
        permutation_rows.extend(perm)
        summary["permutation_top_p_recall"] = top_p_recall(perm, "mse_increase_after_permutation", p)
        summary["permutation_first_p_positive_share"] = first_p_positive_share(perm, "mse_increase_after_permutation", p)

        gaussian_rows.extend(
            evaluate_gaussian_strengths_by_lag(
                model=model,
                context=context,
                true_horizon=true_horizon,
                conditional_mean=conditional_mean,
                test_order=p,
                strengths=args.strengths,
                seed=args.seed + 20_000 + p,
            )
        )

    captured = attach_attention_capture(model)
    for p in orders:
        context, true_horizon, _ = datasets[p]
        s, pr, lr = attention_all_layers(
            model, train_args, context, true_horizon, args.attention_batch_size, captured, p
        )
        attention_summaries.extend(s)
        patch_rows.extend(pr)
        lag_rows.extend(lr)

    min_p = int(get_arg(train_args, "min_ar_order", 1))
    max_p = int(get_arg(train_args, "max_ar_order", 5))
    classify_mixed(summaries, min_p, max_p)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "train_order_range": [min_p, max_p],
        "patch_size": train_args.patch_size,
        "patch_stride": train_args.patch_stride,
        "summaries": summaries,
        "attention_summaries": attention_summaries,
        "attention_patch_rows": patch_rows,
        "attention_lag_distributed_rows": lag_rows,
        "permutation_rows": permutation_rows,
        "gaussian_rows": gaussian_rows,
    }
    json_path = args.output_dir / f"{args.run_name}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.output_dir / f"{args.run_name}_summary.csv", summaries)
    write_csv(args.output_dir / f"{args.run_name}_attention_summary_all_layers.csv", attention_summaries)
    write_csv(args.output_dir / f"{args.run_name}_attention_patch_all_layers.csv", patch_rows)
    write_csv(args.output_dir / f"{args.run_name}_attention_lag_distributed_all_layers.csv", lag_rows)
    write_csv(args.output_dir / f"{args.run_name}_permutation_by_lag.csv", permutation_rows)
    write_csv(args.output_dir / f"{args.run_name}_gaussian_strengths_by_lag.csv", gaussian_rows)

    simple_order_bar(
        orders,
        [float(s["median_mase_to_noisy_next_x"]) for s in summaries],
        title="Mixed patched checkpoint: MASE by test order",
        ylabel="MASE to realized next value",
        path=args.output_dir / f"{args.run_name}_mase_by_order.png",
        reference=1.0,
    )
    simple_order_bar(
        orders,
        [float(s["mean_wql_to_noisy_next_x"]) for s in summaries],
        title="Mixed patched checkpoint: mean WQL by test order",
        ylabel="Mean WQL",
        path=args.output_dir / f"{args.run_name}_wql_by_order.png",
    )
    simple_order_bar(
        orders,
        [float(s["mase_improvement_vs_zero_percent"]) for s in summaries],
        title="Mixed patched checkpoint: MASE improvement over zero predictor",
        ylabel="Improvement over zero (%)",
        path=args.output_dir / f"{args.run_name}_improvement_vs_zero.png",
        reference=0.0,
    )
    plot_permutation(
        permutation_rows,
        orders,
        args.max_lag_plot,
        args.output_dir / f"{args.run_name}_mse_after_permutation_by_lag.png",
    )
    for p in orders:
        gaussian_strength_plot(
            gaussian_rows,
            test_order=p,
            path=args.output_dir / f"{args.run_name}_gaussian_mse_ar{p}.png",
        )
    for layer in range(int(train_args.n_layers)):
        plot_lag_attention(
            lag_rows,
            orders,
            layer,
            args.max_lag_plot,
            args.output_dir / f"{args.run_name}_lag_distributed_attention_layer{layer}.png",
        )
        plot_patch_attention(
            patch_rows,
            orders,
            layer,
            args.output_dir / f"{args.run_name}_patch_attention_layer{layer}.png",
        )

    print("\nTask 4b mixed patching summary")
    for s in summaries:
        print(
            f"AR({s['test_ar_order']}) | MASE={s['median_mase_to_noisy_next_x']:.3f} | "
            f"WQL={s['mean_wql_to_noisy_next_x']:.3f} | "
            f"vs-zero={s['mase_improvement_vs_zero_percent']:.1f}% | "
            f"{s['failure_mode']}"
        )
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

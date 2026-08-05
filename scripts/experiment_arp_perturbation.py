"""Perturbation test for AR(p) UVModel sanity checkpoints.

For each lag, replace that context value and measure how much the median
forecast q0.5 changes. For scalar AR(p), the strongest effects should usually
fall within the first p lags.
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
    parser = argparse.ArgumentParser(description="Run lag perturbation analysis on a trained AR(p) UVModel checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "outputs" / "ar_sanity_uvmodel_best_model.pt")
    parser.add_argument("--eval-n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test-ar-order", type=int, default=None, help="Override/evaluate on a specific AR order; required for mixed checkpoints")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--perturbation", type=str, default="permute", choices=["permute", "zero", "noise"])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--run-name", type=str, default="ar_perturbation")
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
    if label == "mixed":
        raise ValueError("--test-ar-order is required when checkpoint was trained with --ar-order mixed")
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


@torch.no_grad()
def median_forecast(model: UVModel, context: torch.Tensor, true_horizon: torch.Tensor) -> torch.Tensor:
    model.eval()
    _, quantile_forecast = model(context=context, n_horizon=1, true_horizon=true_horizon)
    return quantile_forecast[:, 4, 0]  # q0.5, because quantiles are 0.1, ..., 0.9.


def perturb_context(
    context: torch.Tensor,
    *,
    lag: int,
    mode: str,
    generator: torch.Generator,
    noise_std: float,
) -> torch.Tensor:
    perturbed = context.clone()
    pos = context.shape[1] - lag

    if mode == "permute":
        # Replace x_{t-lag+1} by another series' value at the same lag.
        # This keeps the marginal scale realistic while breaking the information in that lag.
        perm = torch.randperm(context.shape[0], generator=generator, device=context.device)
        perturbed[:, pos, 0] = context[perm, pos, 0]
    elif mode == "zero":
        perturbed[:, pos, 0] = 0.0
    elif mode == "noise":
        noise = torch.randn(context.shape[0], generator=generator, device=context.device) * noise_std
        perturbed[:, pos, 0] = noise
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")

    return perturbed


def save_plot(rows: list[dict[str, float]], path: Path, ar_order: int) -> None:
    lags = [int(r["lag"]) for r in rows]
    mean_abs_delta = [r["mean_abs_delta"] for r in rows]
    mse_increase = [r["mse_increase_to_conditional_mean"] for r in rows]

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(lags, mean_abs_delta, marker="o", label="mean |Δ q0.5|")
    plt.plot(lags, mse_increase, marker="o", label="MSE increase")
    if ar_order > 1:
        plt.axvline(ar_order, linestyle="--", linewidth=1.0, label=f"AR order p={ar_order}")
    plt.xlabel("Lag: 1 means latest context value")
    plt.ylabel("Perturbation effect")
    plt.title(f"AR({ar_order}) perturbation test by lag")
    plt.xticks(lags)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def top_p_recall(rows: list[dict[str, float]], key: str, ar_order: int) -> float:
    top = sorted(rows, key=lambda r: r[key], reverse=True)[:ar_order]
    hits = sum(1 for r in top if int(r["lag"]) <= ar_order)
    return hits / ar_order


def share_first_p(rows: list[dict[str, float]], key: str, ar_order: int, *, positive_only: bool = False) -> float:
    vals = []
    first = []
    for r in rows:
        v = float(r[key])
        if positive_only:
            v = max(v, 0.0)
        vals.append(v)
        if int(r["lag"]) <= ar_order:
            first.append(v)
    total = sum(vals)
    return float(sum(first) / total) if total > 0 else 0.0


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
        print("WARNING: checkpoint is patched; lag-level interpretation is less direct than patch_size=1.")

    model = build_model(train_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

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

    base_pred = median_forecast(model, context, true_horizon)
    base_mse = float((base_pred - conditional_mean).square().mean().cpu())

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)

    rows: list[dict[str, float]] = []
    for lag in range(1, train_args.context_length + 1):
        perturbed_context = perturb_context(
            context,
            lag=lag,
            mode=args.perturbation,
            generator=generator,
            noise_std=train_args.noise_std,
        )
        perturbed_pred = median_forecast(model, perturbed_context, true_horizon)
        delta = perturbed_pred - base_pred
        perturbed_mse = float((perturbed_pred - conditional_mean).square().mean().cpu())

        rows.append(
            {
                "lag": float(lag),
                "is_within_ar_order": float(lag <= ar_order),
                "mean_abs_delta": float(delta.abs().mean().cpu()),
                "mean_squared_delta": float(delta.square().mean().cpu()),
                "mse_to_conditional_mean_after_perturb": perturbed_mse,
                "mse_increase_to_conditional_mean": perturbed_mse - base_mse,
            }
        )

    best_by_delta = max(rows, key=lambda r: r["mean_abs_delta"])
    best_by_mse = max(rows, key=lambda r: r["mse_increase_to_conditional_mean"])

    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "seed": args.seed,
        "eval_n": args.eval_n,
        "perturbation": args.perturbation,
        "train_ar_order": train_ar_order,
        "test_ar_order": ar_order,
        "ar_order": ar_order,
        "pacf_low": pacf_low,
        "pacf_high": pacf_high,
        "context_length": train_args.context_length,
        "patch_size": train_args.patch_size,
        "patch_stride": train_args.patch_stride,
        "use_rope": train_args.use_rope,
        "use_arcsinh": train_args.use_arcsinh,
        "base_median_mse_to_conditional_mean": base_mse,
        "top_lag_by_mean_abs_delta": int(best_by_delta["lag"]),
        "top_lag_by_mse_increase": int(best_by_mse["lag"]),
        "top_p_recall_by_mean_abs_delta": top_p_recall(rows, "mean_abs_delta", ar_order),
        "top_p_recall_by_mse_increase": top_p_recall(rows, "mse_increase_to_conditional_mean", ar_order),
        "first_p_share_of_mean_abs_delta": share_first_p(rows, "mean_abs_delta", ar_order),
        "first_p_share_of_positive_mse_increase": share_first_p(rows, "mse_increase_to_conditional_mean", ar_order, positive_only=True),
        "rows": rows,
    }

    # Backward-compatible aliases for old AR(1) logs.
    if ar_order == 1:
        summary["base_median_mse_to_phi_xt"] = base_mse

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.run_name}_{args.perturbation}.json"
    csv_path = args.output_dir / f"{args.run_name}_{args.perturbation}.csv"
    plot_path = args.output_dir / f"{args.run_name}_{args.perturbation}.png"

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_plot(rows, plot_path, ar_order=ar_order)

    print("Perturbation summary")
    print(f"checkpoint: {args.checkpoint}")
    print(f"train_ar_order: {train_ar_order}")
    print(f"test_ar_order: {ar_order}")
    print(f"base_median_mse_to_conditional_mean: {base_mse:.6f}")
    print(f"top_lag_by_mean_abs_delta: {int(best_by_delta['lag'])}")
    print(f"top_lag_by_mse_increase: {int(best_by_mse['lag'])}")
    print(f"top_p_recall_by_mean_abs_delta: {summary['top_p_recall_by_mean_abs_delta']:.3f}")
    print(f"first_p_share_of_mean_abs_delta: {summary['first_p_share_of_mean_abs_delta']:.3f}")
    print("\nTop 5 lags by mean_abs_delta")
    for r in sorted(rows, key=lambda x: x["mean_abs_delta"], reverse=True)[:5]:
        print(
            f"lag {int(r['lag']):02d} | mean_abs_delta={r['mean_abs_delta']:.6f} | "
            f"mse_increase={r['mse_increase_to_conditional_mean']:.6f}"
        )

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()

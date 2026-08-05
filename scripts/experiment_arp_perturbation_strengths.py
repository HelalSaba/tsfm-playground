"""Additive-Gaussian perturbation strength sweep for AR(p) UVModel checkpoints.

For every lag, add Gaussian noise at several strengths and plot the resulting
MSE.  Strength is measured in standard deviations of the selected lag across
the evaluation batch:

    x_lag_perturbed = x_lag + strength * std(x_lag) * epsilon,
    epsilon ~ N(0, 1)

The strength-0 line is the unperturbed baseline.  The same standard-normal
noise draw is reused across strengths for a given lag, so differences between
lines are caused only by the strength multiplier.

Typical usage:
python scripts\\experiment_arp_perturbation_strengths.py ^
  --checkpoint outputs\\ar1\\scalar\\sanity\\ar1_scalar_rope_seed7_best_model.pt ^
  --test-ar-order 1 ^
  --strengths 0 0.25 0.5 1 2 ^
  --eval-n 4000 ^
  --device cpu ^
  --output-dir outputs\\ar1\\scalar\\revised_eval ^
  --run-name ar1_scalar
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    parser = argparse.ArgumentParser(description="Run an additive-Gaussian lag perturbation strength sweep.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test-ar-order",
        type=int,
        default=None,
        help="Evaluation AR order. Required for a mixed-order checkpoint.",
    )
    parser.add_argument("--eval-n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--mse-target",
        type=str,
        default="conditional_mean",
        choices=["conditional_mean", "noisy_target"],
        help="Target used on the plot's y-axis.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--run-name", type=str, default="ar_perturbation_strengths")
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


def resolve_test_order(args: argparse.Namespace, train_args: SimpleNamespace) -> int:
    if args.test_ar_order is not None:
        return int(args.test_ar_order)
    label = str(get_arg(train_args, "ar_order", 1)).lower()
    if label == "mixed" or label.startswith("mixed"):
        raise ValueError("--test-ar-order is required for a mixed-order checkpoint")
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
    return quantile_forecast[:, 4, 0]


def perturb_additive_gaussian(
    context: torch.Tensor,
    *,
    lag: int,
    strength: float,
    standard_normal: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, float]:
    perturbed = context.clone()
    pos = context.shape[1] - lag
    lag_values = context[:, pos, 0]
    lag_scale = lag_values.std(unbiased=False).clamp_min(eps)
    perturbed[:, pos, 0] = lag_values + float(strength) * lag_scale * standard_normal
    return perturbed, float(lag_scale.cpu())


def top_p_recall(rows: list[dict[str, Any]], ar_order: int) -> float:
    top = sorted(rows, key=lambda r: float(r["mse_increase"]), reverse=True)[:ar_order]
    hits = sum(1 for r in top if int(r["lag"]) <= ar_order)
    return hits / ar_order


def save_plot(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    ar_order: int,
    strengths: list[float],
    mse_target: str,
) -> None:
    plt.figure(figsize=(10.0, 5.8))
    for strength in strengths:
        current = sorted(
            [r for r in rows if math.isclose(float(r["strength"]), strength, rel_tol=0.0, abs_tol=1e-12)],
            key=lambda r: int(r["lag"]),
        )
        lags = [int(r["lag"]) for r in current]
        mses = [float(r["mse_after_perturbation"]) for r in current]
        plt.plot(lags, mses, marker="o", markersize=3, linewidth=1.4, label=f"strength={strength:g}")

    if ar_order < max(int(r["lag"]) for r in rows):
        plt.axvline(ar_order + 0.5, linestyle="--", linewidth=1.0, label=f"true boundary p={ar_order}")
    plt.xlabel("Perturbed lag (1 = latest context value)")
    target_label = "analytic conditional mean" if mse_target == "conditional_mean" else "realized noisy next value"
    plt.ylabel(f"MSE after perturbation to {target_label}")
    plt.title(f"AR({ar_order}) additive Gaussian perturbation by lag and strength")
    plt.xticks(range(1, max(int(r["lag"]) for r in rows) + 1))
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    strengths = sorted(set(float(s) for s in args.strengths))
    if not strengths:
        raise ValueError("--strengths must contain at least one value")
    if any(s < 0 for s in strengths):
        raise ValueError("Perturbation strengths must be non-negative")
    if 0.0 not in strengths:
        strengths = [0.0, *strengths]
        print("Added strength 0 automatically to provide the unperturbed baseline line.")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = as_namespace(checkpoint["args"])
    ar_order = resolve_test_order(args, train_args)

    model = build_model(train_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    batch = generate_arp_batch(
        batch_size=args.eval_n,
        context_length=train_args.context_length,
        ar_order=ar_order,
        pacf_low=float(get_arg(train_args, "pacf_low", -0.9)),
        pacf_high=float(get_arg(train_args, "pacf_high", 0.9)),
        noise_std=train_args.noise_std,
        burn_in=train_args.burn_in,
        device=device,
    )
    context, true_horizon, conditional_mean_uv = ar_batch_to_uvmodel(batch)
    conditional_mean = conditional_mean_uv[:, 0, 0]
    target = conditional_mean if args.mse_target == "conditional_mean" else batch.target

    base_pred = median_forecast(model, context, true_horizon)
    base_mse = float((base_pred - target).square().mean().cpu())

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)
    standard_normals = {
        lag: torch.randn(context.shape[0], generator=generator, device=device)
        for lag in range(1, train_args.context_length + 1)
    }

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for strength in strengths:
        strength_rows: list[dict[str, Any]] = []
        for lag in range(1, train_args.context_length + 1):
            if strength == 0.0:
                perturbed_pred = base_pred
                lag_scale = float(context[:, context.shape[1] - lag, 0].std(unbiased=False).cpu())
            else:
                perturbed_context, lag_scale = perturb_additive_gaussian(
                    context,
                    lag=lag,
                    strength=strength,
                    standard_normal=standard_normals[lag],
                )
                perturbed_pred = median_forecast(model, perturbed_context, true_horizon)

            mse_after = float((perturbed_pred - target).square().mean().cpu())
            row = {
                "strength": strength,
                "lag": lag,
                "is_within_ar_order": int(lag <= ar_order),
                "lag_empirical_std": lag_scale,
                "mse_after_perturbation": mse_after,
                "mse_increase": mse_after - base_mse,
                "mean_abs_forecast_change": float((perturbed_pred - base_pred).abs().mean().cpu()),
            }
            rows.append(row)
            strength_rows.append(row)

        if strength == 0.0:
            summaries.append(
                {
                    "strength": strength,
                    "top_lag_by_mse_increase": None,
                    "top_p_recall_by_mse_increase": None,
                    "max_mse_increase": 0.0,
                }
            )
        else:
            best = max(strength_rows, key=lambda r: float(r["mse_increase"]))
            summaries.append(
                {
                    "strength": strength,
                    "top_lag_by_mse_increase": int(best["lag"]),
                    "top_p_recall_by_mse_increase": top_p_recall(strength_rows, ar_order),
                    "max_mse_increase": float(best["mse_increase"]),
                }
            )

    output = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "seed": args.seed,
        "eval_n": args.eval_n,
        "train_ar_order": str(get_arg(train_args, "ar_order", "unknown")),
        "test_ar_order": ar_order,
        "context_length": train_args.context_length,
        "patch_size": train_args.patch_size,
        "patch_stride": train_args.patch_stride,
        "perturbation": "additive Gaussian noise",
        "strength_definition": "strength times the empirical standard deviation of the selected lag",
        "mse_target": args.mse_target,
        "base_mse": base_mse,
        "strengths": strengths,
        "strength_summaries": summaries,
        "rows": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.run_name}_gaussian_strengths.json"
    csv_path = args.output_dir / f"{args.run_name}_gaussian_strengths.csv"
    plot_path = args.output_dir / f"{args.run_name}_gaussian_strengths.png"

    json_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_plot(rows, path=plot_path, ar_order=ar_order, strengths=strengths, mse_target=args.mse_target)

    print("Gaussian perturbation strength sweep")
    print(f"checkpoint: {args.checkpoint}")
    print(f"test_ar_order: {ar_order}")
    print(f"base_mse ({args.mse_target}): {base_mse:.6f}")
    for summary in summaries:
        if summary["strength"] == 0.0:
            print("strength=0 | unperturbed baseline")
        else:
            print(
                f"strength={summary['strength']:g} | "
                f"top_lag={summary['top_lag_by_mse_increase']} | "
                f"top_p_recall={summary['top_p_recall_by_mse_increase']:.3f} | "
                f"max_mse_increase={summary['max_mse_increase']:.6f}"
            )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()

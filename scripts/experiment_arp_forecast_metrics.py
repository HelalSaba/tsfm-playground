r"""Evaluate an AR(p) UVModel checkpoint with point and probabilistic forecast metrics.

It reuses an existing checkpoint and reports:

Point forecast metrics for q0.5:
- MASE against the realized noisy next value
- sMAPE against the realized noisy next value

Probabilistic forecast metric:
- weighted quantile loss (WQL) across q=0.1,...,0.9 against the realized noisy next value

Typical usage:
python scripts\experiment_arp_forecast_metrics.py ^
  --checkpoint outputs\ar1\scalar\sanity\ar1_scalar_rope_seed7_best_model.pt ^
  --test-ar-order 1 ^
  --eval-n 4000 ^
  --device cpu ^
  --output-dir outputs\ar1\scalar\revised_eval ^
  --run-name ar1_scalar
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

from playground.data.ar import ar_batch_to_uvmodel, arp_ols_forecast, generate_arp_batch
from playground.model.registry.uv import UVModel


QUANTILE_VALUES = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an AR(p) UVModel checkpoint with MASE, sMAPE, and WQL."
    )
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
    parser.add_argument(
        "--seasonality",
        type=int,
        default=1,
        help="MASE naive seasonal period; use 1 for these non-seasonal AR experiments.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--run-name", type=str, default="ar_forecast_metrics")
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
def quantile_forecast(
    model: UVModel,
    context: torch.Tensor,
    true_horizon: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    _, forecast = model(context=context, n_horizon=1, true_horizon=true_horizon)
    if forecast.ndim != 3 or forecast.shape[1] != 9 or forecast.shape[2] != 1:
        raise ValueError(f"Expected forecast shape (batch, 9, 1), got {tuple(forecast.shape)}")
    return forecast[:, :, 0]


def corrcoef(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = (x0.square().sum().sqrt() * y0.square().sum().sqrt()).clamp_min(eps)
    return float(((x0 * y0).sum() / denom).cpu())


def no_intercept_slope(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    numerator = (pred * target).sum()
    denominator = target.square().sum().clamp_min(eps)
    return float((numerator / denominator).cpu())


def mase_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    insample: torch.Tensor,
    *,
    seasonality: int = 1,
    eps: float = 1e-8,
) -> tuple[float, float, float, int]:
    """Return panel MASE, MAE, mean naive scale, and invalid-scale count."""
    if insample.ndim != 2:
        raise ValueError(f"Expected insample shape (batch, time), got {tuple(insample.shape)}")
    if pred.ndim != 1 or target.ndim != 1:
        raise ValueError("pred and target must be one-dimensional")
    if len(pred) != len(target) or len(pred) != insample.shape[0]:
        raise ValueError("pred, target, and insample batch dimensions must match")
    if seasonality < 1 or insample.shape[1] <= seasonality:
        raise ValueError("seasonality must be positive and smaller than the context length")

    naive_scale = (insample[:, seasonality:] - insample[:, :-seasonality]).abs().mean(dim=1)
    valid = naive_scale > eps
    invalid_count = int((~valid).sum().item())
    if not bool(valid.any()):
        raise ValueError("MASE is undefined because every in-sample naive scale is zero")

    abs_error = (pred - target).abs()
    scaled_error = abs_error[valid] / naive_scale[valid]
    return (
        float(scaled_error.mean().cpu()),
        float(abs_error.mean().cpu()),
        float(naive_scale[valid].mean().cpu()),
        invalid_count,
    )


def smape(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[float, float]:
    """Return sMAPE as a fraction and percentage."""
    numerator = 2.0 * (pred - target).abs()
    denominator = pred.abs() + target.abs()
    values = torch.where(denominator > eps, numerator / denominator, torch.zeros_like(denominator))
    fraction = float(values.mean().cpu())
    return fraction, 100.0 * fraction


def weighted_quantile_loss(
    forecasts: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[float, dict[str, float], float]:
    """Compute scale-normalized weighted quantile loss.

    For quantile q:
        wQL_q = 2 * sum(pinball_loss_q) / sum(|target|)

    The reported mean WQL is the arithmetic mean over q=0.1,...,0.9.
    """
    if forecasts.ndim != 2:
        raise ValueError(f"Expected forecasts shape (batch, quantiles), got {tuple(forecasts.shape)}")
    if target.ndim != 1:
        raise ValueError("target must be one-dimensional")
    if forecasts.shape[0] != target.shape[0]:
        raise ValueError("forecast and target batch dimensions must match")
    if forecasts.shape[1] != len(quantiles):
        raise ValueError("forecast quantile dimension does not match quantile values")

    q = quantiles.to(device=forecasts.device, dtype=forecasts.dtype).view(1, -1)
    error = target.view(-1, 1) - forecasts
    pinball = torch.maximum(q * error, (q - 1.0) * error)

    target_scale = target.abs().sum().clamp_min(eps)
    per_quantile = 2.0 * pinball.sum(dim=0) / target_scale
    mean_wql = per_quantile.mean()

    per_q_dict = {
        f"q{float(qv):.1f}": float(loss.cpu())
        for qv, loss in zip(quantiles.tolist(), per_quantile)
    }
    return float(mean_wql.cpu()), per_q_dict, float(target_scale.cpu())


def save_bar_plot(
    *,
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    path: Path,
    reference_line: float | None = None,
    reference_label: str | None = None,
) -> None:
    plt.figure(figsize=(8.8, 5.2))
    bars = plt.bar(labels, values, edgecolor="black", linewidth=0.8)
    if reference_line is not None:
        plt.axhline(
            reference_line,
            linestyle="--",
            linewidth=1.0,
            label=reference_label or str(reference_line),
        )
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    plt.ylabel(ylabel)
    plt.xlabel("Forecast method")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    if reference_line is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_wql_plot(per_quantile: dict[str, float], mean_wql: float, path: Path, ar_order: int) -> None:
    labels = list(per_quantile.keys())
    values = list(per_quantile.values())

    plt.figure(figsize=(8.5, 5.0))
    bars = plt.bar(labels, values, edgecolor="black", linewidth=0.8)
    plt.axhline(mean_wql, linestyle="--", linewidth=1.0, label=f"mean WQL = {mean_wql:.3f}")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.ylabel("Weighted quantile loss (lower is better)")
    plt.xlabel("Forecast quantile")
    plt.title(f"AR({ar_order}) UVModel probabilistic forecast accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.seasonality < 1:
        raise ValueError("--seasonality must be at least 1")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = as_namespace(checkpoint["args"])
    test_order = resolve_test_order(args, train_args)

    model = build_model(train_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    batch = generate_arp_batch(
        batch_size=args.eval_n,
        context_length=train_args.context_length,
        ar_order=test_order,
        pacf_low=float(get_arg(train_args, "pacf_low", -0.9)),
        pacf_high=float(get_arg(train_args, "pacf_high", 0.9)),
        noise_std=train_args.noise_std,
        burn_in=train_args.burn_in,
        device=device,
    )
    context, true_horizon, conditional_mean_uv = ar_batch_to_uvmodel(batch)
    conditional_mean = conditional_mean_uv[:, 0, 0]
    noisy_target = batch.target

    all_quantiles = quantile_forecast(model, context, true_horizon)
    median_pred = all_quantiles[:, 4]
    ols_pred, _ = arp_ols_forecast(batch.context, ar_order=test_order)
    last_value_pred = batch.context[:, -1]
    zero_pred = torch.zeros_like(noisy_target)

    point_predictions = {
        "median": median_pred,
        "ols": ols_pred,
        "analytic_optimum": conditional_mean,
        "last_value": last_value_pred,
        "zero": zero_pred,
    }

    metrics: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "seed": args.seed,
        "eval_n": args.eval_n,
        "test_ar_order": test_order,
        "context_length": train_args.context_length,
        "patch_size": train_args.patch_size,
        "patch_stride": train_args.patch_stride,
        "seasonality": args.seasonality,
        "primary_point_metrics": ["MASE", "sMAPE"],
        "primary_probabilistic_metric": "mean weighted quantile loss across q=0.1,...,0.9",
        "forecast_metric_target": "realized noisy next value",
        "diagnostic_target": "analytic conditional mean",
        "mase_scaling": "per-series in-context seasonal-naive MAE, then averaged across series",
        "wql_definition": "2 * summed pinball loss / summed absolute realized target; averaged over quantiles",
        "median_mse_to_conditional_mean": float((median_pred - conditional_mean).square().mean().cpu()),
        "median_mae_to_conditional_mean": float((median_pred - conditional_mean).abs().mean().cpu()),
        "median_corr_with_conditional_mean_diagnostic": corrcoef(median_pred, conditional_mean),
        "median_slope_vs_conditional_mean_diagnostic": no_intercept_slope(median_pred, conditional_mean),
    }

    invalid_counts: list[int] = []
    for name, pred in point_predictions.items():
        mase_score, mae, scale_mean, invalid_count = mase_components(
            pred,
            noisy_target,
            batch.context,
            seasonality=args.seasonality,
        )
        smape_fraction, smape_percent = smape(pred, noisy_target)

        metrics[f"{name}_mase_to_noisy_next_x"] = mase_score
        metrics[f"{name}_mae_to_noisy_next_x"] = mae
        metrics[f"{name}_smape_fraction_to_noisy_next_x"] = smape_fraction
        metrics[f"{name}_smape_percent_to_noisy_next_x"] = smape_percent
        invalid_counts.append(invalid_count)
        metrics["mase_naive_scale_mean"] = scale_mean

    metrics["mase_invalid_scale_series"] = max(invalid_counts)

    mean_wql, per_quantile_wql, wql_target_scale = weighted_quantile_loss(
        all_quantiles,
        noisy_target,
        QUANTILE_VALUES,
    )
    metrics["uvmodel_mean_wql_to_noisy_next_x"] = mean_wql
    metrics["uvmodel_wql_target_absolute_sum"] = wql_target_scale
    for label, value in per_quantile_wql.items():
        metrics[f"uvmodel_wql_{label}_to_noisy_next_x"] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.run_name}_forecast_metrics.json"
    csv_path = args.output_dir / f"{args.run_name}_forecast_metrics.csv"
    mase_plot_path = args.output_dir / f"{args.run_name}_mase.png"
    smape_plot_path = args.output_dir / f"{args.run_name}_smape.png"
    wql_plot_path = args.output_dir / f"{args.run_name}_wql.png"

    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    display_labels = ["UVModel q0.5", "OLS", "Analytic optimum", "Last value", "Zero"]
    names = ["median", "ols", "analytic_optimum", "last_value", "zero"]

    save_bar_plot(
        labels=display_labels,
        values=[metrics[f"{name}_mase_to_noisy_next_x"] for name in names],
        ylabel="MASE to realized next value (lower is better)",
        title=f"AR({test_order}) one-step point forecast: MASE",
        path=mase_plot_path,
        reference_line=1.0,
        reference_label="MASE = 1",
    )
    save_bar_plot(
        labels=display_labels,
        values=[metrics[f"{name}_smape_percent_to_noisy_next_x"] for name in names],
        ylabel="sMAPE (%) to realized next value (lower is better)",
        title=f"AR({test_order}) one-step point forecast: sMAPE",
        path=smape_plot_path,
    )
    save_wql_plot(per_quantile_wql, mean_wql, wql_plot_path, test_order)

    print("Forecast metric evaluation")
    print(f"checkpoint: {args.checkpoint}")
    print(f"test_ar_order: {test_order}")
    print(f"UVModel MASE: {metrics['median_mase_to_noisy_next_x']:.6f}")
    print(f"UVModel sMAPE: {metrics['median_smape_percent_to_noisy_next_x']:.3f}%")
    print(f"UVModel mean WQL: {metrics['uvmodel_mean_wql_to_noisy_next_x']:.6f}")
    print(
        "Diagnostic only | "
        f"corr={metrics['median_corr_with_conditional_mean_diagnostic']:.6f} | "
        f"slope={metrics['median_slope_vs_conditional_mean_diagnostic']:.6f}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {mase_plot_path}")
    print(f"Wrote {smape_plot_path}")
    print(f"Wrote {wql_plot_path}")


if __name__ == "__main__":
    main()

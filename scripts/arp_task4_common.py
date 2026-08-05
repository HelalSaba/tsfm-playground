# Shared utilities for the revised Task 4 AR evaluations.

from __future__ import annotations

import csv
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

from playground.data.ar import ar_batch_to_uvmodel, arp_ols_forecast, generate_arp_batch
from playground.model.registry.uv import UVModel


QUANTILES = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(device_arg)


def as_namespace(d: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**d)


def get_arg(args: SimpleNamespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


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


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_fixed_order_data(
    *,
    train_args: SimpleNamespace,
    test_order: int,
    eval_n: int,
    seed: int,
    device: torch.device,
):
    seed_everything(seed)
    batch = generate_arp_batch(
        batch_size=eval_n,
        context_length=train_args.context_length,
        ar_order=test_order,
        pacf_low=float(get_arg(train_args, "pacf_low", -0.9)),
        pacf_high=float(get_arg(train_args, "pacf_high", 0.9)),
        noise_std=train_args.noise_std,
        burn_in=train_args.burn_in,
        device=device,
    )
    context, true_horizon, conditional_mean_uv = ar_batch_to_uvmodel(batch)
    return batch, context, true_horizon, conditional_mean_uv[:, 0, 0]


@torch.no_grad()
def all_quantile_forecasts(
    model: UVModel,
    context: torch.Tensor,
    true_horizon: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    _, forecast = model(context=context, n_horizon=1, true_horizon=true_horizon)
    if forecast.ndim != 3 or forecast.shape[1:] != (9, 1):
        raise ValueError(f"Expected forecast shape (batch, 9, 1), got {tuple(forecast.shape)}")
    return forecast[:, :, 0]


@torch.no_grad()
def median_forecast(
    model: UVModel,
    context: torch.Tensor,
    true_horizon: torch.Tensor,
) -> torch.Tensor:
    return all_quantile_forecasts(model, context, true_horizon)[:, 4]


def mase(
    pred: torch.Tensor,
    target: torch.Tensor,
    insample: torch.Tensor,
    *,
    seasonality: int = 1,
    eps: float = 1e-8,
) -> float:
    if seasonality < 1 or insample.shape[1] <= seasonality:
        raise ValueError("seasonality must be positive and smaller than context length")
    scale = (insample[:, seasonality:] - insample[:, :-seasonality]).abs().mean(dim=1)
    valid = scale > eps
    if not bool(valid.any()):
        raise ValueError("MASE is undefined because all naive scales are zero")
    return float(((pred[valid] - target[valid]).abs() / scale[valid]).mean().cpu())


def smape_percent(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    denominator = pred.abs() + target.abs()
    value = torch.where(
        denominator > eps,
        2.0 * (pred - target).abs() / denominator,
        torch.zeros_like(denominator),
    )
    return 100.0 * float(value.mean().cpu())


def weighted_quantile_loss(
    forecasts: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[float, dict[str, float]]:
    q = QUANTILES.to(device=forecasts.device, dtype=forecasts.dtype).view(1, -1)
    error = target.view(-1, 1) - forecasts
    pinball = torch.maximum(q * error, (q - 1.0) * error)
    scale = target.abs().sum().clamp_min(eps)
    per_q = 2.0 * pinball.sum(dim=0) / scale
    return (
        float(per_q.mean().cpu()),
        {f"wql_q{float(qv):.1f}": float(v.cpu()) for qv, v in zip(QUANTILES, per_q)},
    )


@torch.no_grad()
def forecast_metrics(
    *,
    model: UVModel,
    batch,
    context: torch.Tensor,
    true_horizon: torch.Tensor,
    conditional_mean: torch.Tensor,
    test_order: int,
    seasonality: int = 1,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    forecasts = all_quantile_forecasts(model, context, true_horizon)
    median = forecasts[:, 4]
    noisy_target = batch.target
    zero = torch.zeros_like(noisy_target)
    last = batch.context[:, -1]
    ols, _ = arp_ols_forecast(batch.context, ar_order=test_order)

    mean_wql, per_q = weighted_quantile_loss(forecasts, noisy_target)
    model_mase = mase(median, noisy_target, batch.context, seasonality=seasonality)
    zero_mase = mase(zero, noisy_target, batch.context, seasonality=seasonality)
    last_mase = mase(last, noisy_target, batch.context, seasonality=seasonality)
    ols_mase = mase(ols, noisy_target, batch.context, seasonality=seasonality)
    analytic_mase = mase(conditional_mean, noisy_target, batch.context, seasonality=seasonality)

    model_mse_cond = float((median - conditional_mean).square().mean().cpu())
    zero_mse_cond = float(conditional_mean.square().mean().cpu())

    metrics: dict[str, Any] = {
        "test_ar_order": int(test_order),
        "median_mase_to_noisy_next_x": model_mase,
        "median_smape_percent_to_noisy_next_x": smape_percent(median, noisy_target),
        "mean_wql_to_noisy_next_x": mean_wql,
        "zero_mase_to_noisy_next_x": zero_mase,
        "last_value_mase_to_noisy_next_x": last_mase,
        "ols_mase_to_noisy_next_x": ols_mase,
        "analytic_optimum_mase_to_noisy_next_x": analytic_mase,
        "median_mae_to_noisy_next_x": float((median - noisy_target).abs().mean().cpu()),
        "median_mse_to_noisy_next_x": float((median - noisy_target).square().mean().cpu()),
        "median_mse_to_conditional_mean": model_mse_cond,
        "zero_mse_to_conditional_mean": zero_mse_cond,
        "model_mse_over_zero_mse": model_mse_cond / max(zero_mse_cond, 1e-12),
        "mase_improvement_vs_zero_percent": 100.0 * (zero_mase - model_mase) / max(zero_mase, 1e-12),
        **per_q,
    }
    tensors = {
        "forecasts": forecasts,
        "median": median,
        "noisy_target": noisy_target,
        "conditional_mean": conditional_mean,
    }
    return metrics, tensors


def evaluate_permutation_by_lag(
    *,
    model: UVModel,
    context: torch.Tensor,
    true_horizon: torch.Tensor,
    conditional_mean: torch.Tensor,
    test_order: int,
    seed: int,
) -> list[dict[str, Any]]:
    base = median_forecast(model, context, true_horizon)
    base_mse = float((base - conditional_mean).square().mean().cpu())
    generator = torch.Generator(device=context.device)
    generator.manual_seed(seed)

    rows: list[dict[str, Any]] = []
    for lag in range(1, context.shape[1] + 1):
        perturbed = context.clone()
        pos = context.shape[1] - lag
        perm = torch.randperm(context.shape[0], generator=generator, device=context.device)
        perturbed[:, pos, 0] = context[perm, pos, 0]
        pred = median_forecast(model, perturbed, true_horizon)
        mse_after = float((pred - conditional_mean).square().mean().cpu())
        rows.append(
            {
                "test_ar_order": test_order,
                "lag": lag,
                "is_within_ar_order": int(lag <= test_order),
                "base_mse_to_conditional_mean": base_mse,
                "mse_after_permutation": mse_after,
                "mse_increase_after_permutation": mse_after - base_mse,
                "mean_abs_forecast_change": float((pred - base).abs().mean().cpu()),
            }
        )
    return rows


def evaluate_gaussian_strengths_by_lag(
    *,
    model: UVModel,
    context: torch.Tensor,
    true_horizon: torch.Tensor,
    conditional_mean: torch.Tensor,
    test_order: int,
    strengths: list[float],
    seed: int,
) -> list[dict[str, Any]]:
    strengths = sorted(set(float(s) for s in strengths))
    if 0.0 not in strengths:
        strengths = [0.0, *strengths]
    if any(s < 0 for s in strengths):
        raise ValueError("Gaussian perturbation strengths must be non-negative")

    base = median_forecast(model, context, true_horizon)
    base_mse = float((base - conditional_mean).square().mean().cpu())
    nonzero = [s for s in strengths if s > 0]

    generator = torch.Generator(device=context.device)
    generator.manual_seed(seed)
    rows: list[dict[str, Any]] = []

    for lag in range(1, context.shape[1] + 1):
        rows.append(
            {
                "test_ar_order": test_order,
                "lag": lag,
                "strength": 0.0,
                "is_within_ar_order": int(lag <= test_order),
                "base_mse_to_conditional_mean": base_mse,
                "mse_after_gaussian_perturbation": base_mse,
                "mse_increase_after_gaussian_perturbation": 0.0,
            }
        )
        if not nonzero:
            continue

        pos = context.shape[1] - lag
        lag_values = context[:, pos, 0]
        lag_scale = lag_values.std(unbiased=False).clamp_min(1e-8)
        standard_normal = torch.randn(
            context.shape[0], generator=generator, device=context.device
        )

        contexts: list[torch.Tensor] = []
        for strength in nonzero:
            perturbed = context.clone()
            perturbed[:, pos, 0] = lag_values + strength * lag_scale * standard_normal
            contexts.append(perturbed)

        stacked_context = torch.cat(contexts, dim=0)
        stacked_horizon = true_horizon.repeat(len(nonzero), 1, 1)
        stacked_pred = median_forecast(model, stacked_context, stacked_horizon)
        split_preds = stacked_pred.split(context.shape[0])

        for strength, pred in zip(nonzero, split_preds):
            mse_after = float((pred - conditional_mean).square().mean().cpu())
            rows.append(
                {
                    "test_ar_order": test_order,
                    "lag": lag,
                    "strength": strength,
                    "is_within_ar_order": int(lag <= test_order),
                    "base_mse_to_conditional_mean": base_mse,
                    "mse_after_gaussian_perturbation": mse_after,
                    "mse_increase_after_gaussian_perturbation": mse_after - base_mse,
                }
            )
    return rows


def top_p_recall(rows: list[dict[str, Any]], key: str, p: int) -> float:
    top = sorted(rows, key=lambda r: float(r[key]), reverse=True)[:p]
    return sum(int(r["lag"]) <= p for r in top) / p


def first_p_positive_share(rows: list[dict[str, Any]], key: str, p: int) -> float:
    vals = [max(float(r[key]), 0.0) for r in rows]
    total = sum(vals)
    if total <= 0:
        return 0.0
    first = sum(max(float(r[key]), 0.0) for r in rows if int(r["lag"]) <= p)
    return first / total


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def simple_order_bar(
    orders: list[int],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    path: Path,
    reference: float | None = None,
) -> None:
    plt.figure(figsize=(9, 5))
    labels = [f"AR({p})" for p in orders]
    bars = plt.bar(labels, values, edgecolor="black", linewidth=0.8)
    if reference is not None:
        plt.axhline(reference, linestyle="--", linewidth=1.0)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xlabel("Test order")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def gaussian_strength_plot(
    rows: list[dict[str, Any]],
    *,
    test_order: int,
    path: Path,
) -> None:
    current = [r for r in rows if int(r["test_ar_order"]) == test_order]
    strengths = sorted({float(r["strength"]) for r in current})

    plt.figure(figsize=(10, 5.8))
    for strength in strengths:
        line = sorted(
            [r for r in current if math.isclose(float(r["strength"]), strength)],
            key=lambda r: int(r["lag"]),
        )
        plt.plot(
            [int(r["lag"]) for r in line],
            [float(r["mse_after_gaussian_perturbation"]) for r in line],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"strength={strength:g}",
        )
    plt.axvline(test_order + 0.5, linestyle="--", linewidth=1.0, label=f"true p={test_order}")
    plt.xlabel("Perturbed lag (1 = latest context value)")
    plt.ylabel("MSE to analytic conditional mean after perturbation")
    plt.title(f"AR({test_order}) Gaussian perturbation by lag and strength")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

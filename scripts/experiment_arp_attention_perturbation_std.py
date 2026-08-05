"""Evaluate AR(1)-AR(7 specialists with attention and perturbation uncertainty bands.

This script reruns both analyses on each specialist checkpoint and writes all
results into one output folder.  It produces aggregate CSV/JSON files, compact
per-example NPZ files, and attention/perturbation plots with mean +/- one
standard-deviation bands.

Definition of the plotted standard deviations
---------------------------------------------
Attention:
    For each evaluation example, context-normalized horizon-token attention is
    averaged over attention heads.  The plotted curve is the mean over examples
    and the band is the population standard deviation over examples.

Perturbation:
    For each lag and evaluation example, the value is
        abs(q0.5_perturbed - q0.5_base).
    The plotted curve is the mean over examples and the band is the population
    standard deviation over examples.

Typical usage from the repository root:

    python scripts/experiment_arp_attention_perturbation_std.py --device cpu

Default checkpoints:

    outputs/ar{p}/scalar/sanity/ar{p}_scalar_rope_seed7_best_model.pt

Default output folder:

    outputs/ar1_to_ar7_std_eval
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


def find_repo_root(script_path: Path) -> Path:
    """Find the repository root from a script placed in root or scripts/."""
    candidates = [script_path.parent, *script_path.parents]
    for candidate in candidates:
        if (candidate / "src" / "playground").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the repository root containing src/playground. "
        "Place this file in the repository root or in its scripts directory."
    )


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from playground.data.ar import ar_batch_to_uvmodel, generate_arp_batch
from playground.model.registry.uv import UVModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AR specialist models and save attention/perturbation "
            "means, standard deviations, raw per-example arrays, and plots."
        )
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=list(range(1, 8)),
        help="AR orders to evaluate. Default: 1 2 3 4 5 6 7.",
    )
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        default="outputs/ar{p}/scalar/sanity/ar{p}_scalar_rope_seed7_best_model.pt",
        help=(
            "Checkpoint path template. Use {p} for the AR order. Relative paths "
            "are resolved from the repository root."
        ),
    )
    parser.add_argument("--eval-n", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--perturbation",
        type=str,
        default="permute",
        choices=["permute", "zero", "noise"],
    )
    parser.add_argument(
        "--attention-layer",
        type=int,
        default=1,
        help="Layer index plotted in the combined attention figure. Use -1 for last layer.",
    )
    parser.add_argument(
        "--std-multiplier",
        type=float,
        default=1.0,
        help="Width of uncertainty bands in standard deviations. Default: 1.0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ar1_to_ar7_std_eval"),
        help="One folder that receives all generated files.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="ar1_to_ar7_attention_perturbation_std",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--no-save-per-example",
        action="store_true",
        help="Do not save the compact per-example NPZ files.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(device_arg)


def resolve_output_dir(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_checkpoint(template: str, order: int) -> Path:
    formatted = template.format(p=order, order=order)
    if os.name != "nt":
        formatted = formatted.replace("\\", "/")
    candidate = Path(formatted).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if candidate.is_file():
        return candidate

    # Helpful fallback when the run name differs but there is only one specialist
    # checkpoint in the usual sanity directory.
    fallback_dir = REPO_ROOT / "outputs" / f"ar{order}" / "scalar" / "sanity"
    fallback = sorted(fallback_dir.glob("*_best_model.pt")) if fallback_dir.is_dir() else []
    if len(fallback) == 1:
        print(f"Checkpoint not found at {candidate}; using {fallback[0]}")
        return fallback[0]

    detail = f"Expected checkpoint: {candidate}"
    if fallback:
        detail += "\nCandidate checkpoints:\n  " + "\n  ".join(str(path) for path in fallback)
    raise FileNotFoundError(detail)


def as_namespace(value: Any) -> SimpleNamespace:
    if isinstance(value, SimpleNamespace):
        return value
    if isinstance(value, argparse.Namespace):
        return SimpleNamespace(**vars(value))
    if isinstance(value, dict):
        return SimpleNamespace(**value)
    raise TypeError(f"Unsupported checkpoint args type: {type(value)!r}")


def get_arg(train_args: SimpleNamespace, name: str, default: Any) -> Any:
    return getattr(train_args, name, default)


def train_order_label(train_args: SimpleNamespace) -> str:
    return str(get_arg(train_args, "ar_order", 1))


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
    """Capture the same attention weights used by experiment_arp_attention.py."""
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
def median_forecast(
    model: UVModel,
    context: torch.Tensor,
    true_horizon: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    _, quantile_forecast = model(
        context=context,
        n_horizon=1,
        true_horizon=true_horizon,
    )
    return quantile_forecast[:, 4, 0]


@torch.no_grad()
def batched_median_forecast(
    model: UVModel,
    context: torch.Tensor,
    true_horizon: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for start in range(0, context.shape[0], batch_size):
        end = min(start + batch_size, context.shape[0])
        pred = median_forecast(model, context[start:end], true_horizon[start:end])
        predictions.append(pred.detach().cpu())
    return torch.cat(predictions, dim=0)


def perturb_context(
    context: torch.Tensor,
    *,
    lag: int,
    mode: str,
    generator: torch.Generator,
    noise_std: float,
) -> torch.Tensor:
    perturbed = context.clone()
    position = context.shape[1] - lag

    if mode == "permute":
        permutation = torch.randperm(
            context.shape[0],
            generator=generator,
            device=context.device,
        )
        perturbed[:, position, 0] = context[permutation, position, 0]
    elif mode == "zero":
        perturbed[:, position, 0] = 0.0
    elif mode == "noise":
        noise = torch.randn(
            context.shape[0],
            generator=generator,
            device=context.device,
        ) * noise_std
        perturbed[:, position, 0] = noise
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")

    return perturbed


def resolve_plot_layer(requested: int, n_layers: int) -> int:
    resolved = requested if requested >= 0 else n_layers + requested
    if not 0 <= resolved < n_layers:
        raise ValueError(
            f"Requested attention layer {requested}, but model has layers 0..{n_layers - 1}"
        )
    return resolved


def tensor_mean_std_sem(values: torch.Tensor, dim: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=dim)
    std = values.std(dim=dim, unbiased=False)
    n = values.shape[dim]
    sem = std / math.sqrt(n)
    return mean, std, sem


@torch.no_grad()
def collect_attention(
    model: UVModel,
    context: torch.Tensor,
    true_horizon: torch.Tensor,
    *,
    n_layers: int,
    context_length: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    captured = attach_attention_capture(model)
    norm_chunks: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    raw_chunks: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    context_mass_chunks: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    self_mass_chunks: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    for start in range(0, context.shape[0], batch_size):
        end = min(start + batch_size, context.shape[0])
        captured.clear()
        _ = median_forecast(model, context[start:end], true_horizon[start:end])

        if len(captured) != n_layers:
            raise RuntimeError(
                f"Expected {n_layers} captured attention tensors, got {len(captured)}"
            )

        for layer_idx, weights in captured:
            # weights: (examples, heads, sequence, sequence)
            horizon_to_context = weights[:, :, -1, :context_length]
            context_mass = horizon_to_context.sum(dim=-1)
            context_normalized = horizon_to_context / context_mass.unsqueeze(-1).clamp_min(1e-8)

            # Average over heads first.  The standard-deviation band is then
            # computed over examples, which is directly interpretable as
            # example-to-example variability.
            norm_per_example = context_normalized.mean(dim=1)
            raw_per_example = horizon_to_context.mean(dim=1)
            context_mass_per_example = context_mass.mean(dim=1)
            self_mass_per_example = weights[:, :, -1, -1].mean(dim=1)

            # Convert context-position order to lag order: lag 1 is latest.
            norm_chunks[layer_idx].append(torch.flip(norm_per_example, dims=[1]))
            raw_chunks[layer_idx].append(torch.flip(raw_per_example, dims=[1]))
            context_mass_chunks[layer_idx].append(context_mass_per_example)
            self_mass_chunks[layer_idx].append(self_mass_per_example)

    norm = torch.stack(
        [torch.cat(chunks, dim=0) for chunks in norm_chunks],
        dim=0,
    )
    raw = torch.stack(
        [torch.cat(chunks, dim=0) for chunks in raw_chunks],
        dim=0,
    )
    context_mass = torch.stack(
        [torch.cat(chunks, dim=0) for chunks in context_mass_chunks],
        dim=0,
    )
    self_mass = torch.stack(
        [torch.cat(chunks, dim=0) for chunks in self_mass_chunks],
        dim=0,
    )

    return {
        "context_normalized_per_example": norm,
        "raw_per_example": raw,
        "context_mass_per_example": context_mass,
        "self_mass_per_example": self_mass,
    }


@torch.no_grad()
def collect_perturbation(
    model: UVModel,
    context: torch.Tensor,
    true_horizon: torch.Tensor,
    conditional_mean: torch.Tensor,
    *,
    batch_size: int,
    mode: str,
    noise_std: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    base_prediction = batched_median_forecast(
        model,
        context,
        true_horizon,
        batch_size,
    )
    conditional_mean_cpu = conditional_mean.detach().cpu()
    base_squared_error = (base_prediction - conditional_mean_cpu).square()

    n_examples = context.shape[0]
    context_length = context.shape[1]
    signed_delta = torch.empty(n_examples, context_length, dtype=base_prediction.dtype)
    mse_increase = torch.empty_like(signed_delta)

    generator = torch.Generator(device=context.device)
    generator.manual_seed(seed)

    for lag in range(1, context_length + 1):
        perturbed_context = perturb_context(
            context,
            lag=lag,
            mode=mode,
            generator=generator,
            noise_std=noise_std,
        )
        perturbed_prediction = batched_median_forecast(
            model,
            perturbed_context,
            true_horizon,
            batch_size,
        )
        delta = perturbed_prediction - base_prediction
        signed_delta[:, lag - 1] = delta
        mse_increase[:, lag - 1] = (
            (perturbed_prediction - conditional_mean_cpu).square() - base_squared_error
        )
        del perturbed_context, perturbed_prediction, delta

    return {
        "base_prediction": base_prediction,
        "signed_delta_per_example": signed_delta,
        "abs_delta_per_example": signed_delta.abs(),
        "squared_delta_per_example": signed_delta.square(),
        "mse_increase_per_example": mse_increase,
        "base_squared_error_per_example": base_squared_error,
    }


def attention_rows_for_order(
    order: int,
    checkpoint: Path,
    attention: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    normalized = attention["context_normalized_per_example"]
    raw = attention["raw_per_example"]
    n_layers, n_examples, context_length = normalized.shape

    rows: list[dict[str, Any]] = []
    for layer in range(n_layers):
        norm_mean, norm_std, norm_sem = tensor_mean_std_sem(normalized[layer], dim=0)
        raw_mean, raw_std, raw_sem = tensor_mean_std_sem(raw[layer], dim=0)

        for lag in range(1, context_length + 1):
            index = lag - 1
            rows.append(
                {
                    "ar_order": order,
                    "checkpoint": str(checkpoint),
                    "layer": layer,
                    "lag": lag,
                    "is_within_ar_order": int(lag <= order),
                    "n_examples": n_examples,
                    "context_normalized_attention": float(norm_mean[index]),
                    "context_normalized_attention_std": float(norm_std[index]),
                    "context_normalized_attention_sem": float(norm_sem[index]),
                    "raw_attention": float(raw_mean[index]),
                    "raw_attention_std": float(raw_std[index]),
                    "raw_attention_sem": float(raw_sem[index]),
                }
            )
    return rows


def perturbation_rows_for_order(
    order: int,
    checkpoint: Path,
    perturbation: dict[str, torch.Tensor],
    mode: str,
) -> list[dict[str, Any]]:
    signed = perturbation["signed_delta_per_example"]
    absolute = perturbation["abs_delta_per_example"]
    squared = perturbation["squared_delta_per_example"]
    mse_increase = perturbation["mse_increase_per_example"]
    base_squared_error = perturbation["base_squared_error_per_example"]
    n_examples, context_length = signed.shape

    signed_mean, signed_std, signed_sem = tensor_mean_std_sem(signed, dim=0)
    abs_mean, abs_std, abs_sem = tensor_mean_std_sem(absolute, dim=0)
    squared_mean, squared_std, squared_sem = tensor_mean_std_sem(squared, dim=0)
    mse_inc_mean, mse_inc_std, mse_inc_sem = tensor_mean_std_sem(mse_increase, dim=0)

    base_mse = float(base_squared_error.mean())
    rows: list[dict[str, Any]] = []
    for lag in range(1, context_length + 1):
        index = lag - 1
        rows.append(
            {
                "ar_order": order,
                "checkpoint": str(checkpoint),
                "perturbation": mode,
                "lag": lag,
                "is_within_ar_order": int(lag <= order),
                "n_examples": n_examples,
                "base_median_mse_to_conditional_mean": base_mse,
                "mean_abs_delta": float(abs_mean[index]),
                "std_abs_delta": float(abs_std[index]),
                "sem_abs_delta": float(abs_sem[index]),
                "mean_signed_delta": float(signed_mean[index]),
                "std_signed_delta": float(signed_std[index]),
                "sem_signed_delta": float(signed_sem[index]),
                "mean_squared_delta": float(squared_mean[index]),
                "std_squared_delta": float(squared_std[index]),
                "sem_squared_delta": float(squared_sem[index]),
                "mse_increase_to_conditional_mean": float(mse_inc_mean[index]),
                "mse_increase_std": float(mse_inc_std[index]),
                "mse_increase_sem": float(mse_inc_sem[index]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rows_for_order(
    rows: Iterable[dict[str, Any]],
    order: int,
    *,
    layer: int | None = None,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row["ar_order"]) == order]
    if layer is not None:
        selected = [row for row in selected if int(row["layer"]) == layer]
    return sorted(selected, key=lambda row: int(row["lag"]))


def add_ar_order_background(ax: plt.Axes, orders: list[int], colors: dict[int, Any]) -> None:
    for order in orders:
        ax.axvspan(
            order - 1,
            order,
            color=colors[order],
            alpha=0.08,
            linewidth=0,
            zorder=0,
        )


def draw_attention(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    orders: list[int],
    layer: int,
    colors: dict[int, Any],
    std_multiplier: float,
) -> int:
    add_ar_order_background(ax, orders, colors)
    max_lag = 0
    for order in orders:
        current = rows_for_order(rows, order, layer=layer)
        lags = np.asarray([int(row["lag"]) for row in current])
        mean = np.asarray([float(row["context_normalized_attention"]) for row in current])
        std = np.asarray([float(row["context_normalized_attention_std"]) for row in current])
        max_lag = max(max_lag, int(lags.max()))

        ax.fill_between(
            lags,
            np.maximum(mean - std_multiplier * std, 0.0),
            mean + std_multiplier * std,
            color=colors[order],
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            lags,
            mean,
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors[order],
            label=f"AR({order})",
            zorder=2,
        )

    ax.set_xlim(0, max_lag + 0.5)
    ax.set_xticks(range(1, max_lag + 1))
    ax.set_xlabel("Lag: 1 means latest context value")
    ax.set_ylabel("Context-normalized attention")
    ax.set_title(f"AR(1)-AR(7) horizon-token attention by lag, layer {layer}")
    ax.grid(axis="y", alpha=0.25)
    return max_lag


def draw_perturbation(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    orders: list[int],
    colors: dict[int, Any],
    std_multiplier: float,
) -> int:
    add_ar_order_background(ax, orders, colors)
    max_lag = 0
    for order in orders:
        current = rows_for_order(rows, order)
        lags = np.asarray([int(row["lag"]) for row in current])
        mean = np.asarray([float(row["mean_abs_delta"]) for row in current])
        std = np.asarray([float(row["std_abs_delta"]) for row in current])
        max_lag = max(max_lag, int(lags.max()))

        ax.fill_between(
            lags,
            np.maximum(mean - std_multiplier * std, 0.0),
            mean + std_multiplier * std,
            color=colors[order],
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            lags,
            mean,
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors[order],
            label=f"AR({order})",
            zorder=2,
        )

    ax.set_xlim(0, max_lag + 0.5)
    ax.set_xticks(range(1, max_lag + 1))
    ax.set_xlabel("Lag: 1 means latest context value")
    ax.set_ylabel("Mean |Delta q0.5|")
    ax.set_title("AR(1)-AR(7) perturbation test by lag")
    ax.grid(axis="y", alpha=0.25)
    return max_lag


def save_plots(
    attention_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
    *,
    orders: list[int],
    layer: int,
    std_multiplier: float,
    output_dir: Path,
    run_name: str,
    dpi: int,
) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    colors = {order: cmap(index % 10) for index, order in enumerate(orders)}
    output_paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(11, 6.5))
    draw_attention(
        ax,
        attention_rows,
        orders=orders,
        layer=layer,
        colors=colors,
        std_multiplier=std_multiplier,
    )
    ax.legend(ncol=2)
    fig.tight_layout()
    path = output_dir / f"{run_name}_attention_layer{layer}_with_std.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    output_paths["attention_plot"] = str(path)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    draw_perturbation(
        ax,
        perturbation_rows,
        orders=orders,
        colors=colors,
        std_multiplier=std_multiplier,
    )
    ax.legend(ncol=2)
    fig.tight_layout()
    path = output_dir / f"{run_name}_perturbation_mean_abs_delta_with_std.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    output_paths["perturbation_plot"] = str(path)

    fig, (ax_attention, ax_perturbation) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(18, 6.5),
    )
    draw_attention(
        ax_attention,
        attention_rows,
        orders=orders,
        layer=layer,
        colors=colors,
        std_multiplier=std_multiplier,
    )
    draw_perturbation(
        ax_perturbation,
        perturbation_rows,
        orders=orders,
        colors=colors,
        std_multiplier=std_multiplier,
    )
    handles, labels = ax_attention.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(orders),
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=3)
    path = output_dir / f"{run_name}_side_by_side_with_std.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    output_paths["side_by_side_plot"] = str(path)

    return output_paths


def selected_train_config(train_args: SimpleNamespace) -> dict[str, Any]:
    names = [
        "ar_order",
        "context_length",
        "burn_in",
        "noise_std",
        "pacf_low",
        "pacf_high",
        "patch_size",
        "patch_stride",
        "d_model",
        "d_ff",
        "d_kv",
        "n_heads",
        "n_layers",
        "dropout",
        "activation",
        "use_rope",
        "use_arcsinh",
    ]
    return {name: getattr(train_args, name, None) for name in names}


def save_per_example_npz(
    path: Path,
    *,
    order: int,
    batch: Any,
    perturbation: dict[str, torch.Tensor],
    attention: dict[str, torch.Tensor],
) -> None:
    np.savez_compressed(
        path,
        ar_order=np.asarray(order, dtype=np.int64),
        lags=np.arange(1, batch.context.shape[1] + 1, dtype=np.int64),
        context=batch.context.detach().cpu().numpy(),
        target=batch.target.detach().cpu().numpy(),
        conditional_mean=batch.conditional_mean.detach().cpu().numpy(),
        coefficients=batch.coeffs.detach().cpu().numpy(),
        pacf=batch.pacf.detach().cpu().numpy(),
        base_prediction=perturbation["base_prediction"].numpy(),
        attention_context_normalized_per_example=attention[
            "context_normalized_per_example"
        ].numpy(),
        attention_raw_per_example=attention["raw_per_example"].numpy(),
        attention_context_mass_per_example=attention[
            "context_mass_per_example"
        ].numpy(),
        attention_horizon_self_mass_per_example=attention[
            "self_mass_per_example"
        ].numpy(),
        perturbation_signed_delta_per_example=perturbation[
            "signed_delta_per_example"
        ].numpy(),
        perturbation_abs_delta_per_example=perturbation[
            "abs_delta_per_example"
        ].numpy(),
        perturbation_squared_delta_per_example=perturbation[
            "squared_delta_per_example"
        ].numpy(),
        perturbation_mse_increase_per_example=perturbation[
            "mse_increase_per_example"
        ].numpy(),
        base_squared_error_per_example=perturbation[
            "base_squared_error_per_example"
        ].numpy(),
    )


def main() -> None:
    args = parse_args()
    if args.eval_n < 2:
        raise ValueError("--eval-n must be at least 2")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.std_multiplier < 0:
        raise ValueError("--std-multiplier must be non-negative")

    orders = sorted(set(int(order) for order in args.orders))
    if not orders or any(order < 1 for order in orders):
        raise ValueError("All AR orders must be positive integers")

    device = resolve_device(args.device)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository root: {REPO_ROOT}")
    print(f"Device: {device}")
    print(f"All outputs will be written to: {output_dir}")

    all_attention_rows: list[dict[str, Any]] = []
    all_perturbation_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    resolved_plot_layers: set[int] = set()

    for order in orders:
        checkpoint_path = resolve_checkpoint(args.checkpoint_template, order)
        print(f"\n=== AR({order}) ===")
        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if "args" not in checkpoint or "model_state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint {checkpoint_path} must contain 'args' and 'model_state_dict'"
            )

        train_args = as_namespace(checkpoint["args"])
        train_label = train_order_label(train_args).lower()
        if train_label not in {str(order), f"ar{order}"}:
            print(
                f"WARNING: checkpoint train ar_order is {train_label!r}, "
                f"but it is being evaluated as AR({order})."
            )

        if train_args.patch_size != 1 or train_args.patch_stride != 1:
            raise ValueError(
                f"AR({order}) checkpoint uses patch_size={train_args.patch_size}, "
                f"patch_stride={train_args.patch_stride}. Direct lag-level attention "
                "requires patch_size=1 and patch_stride=1."
            )
        if not getattr(train_args, "use_rope", False):
            print("WARNING: checkpoint was trained with use_rope=False.")

        model = build_model(train_args).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        plot_layer = resolve_plot_layer(args.attention_layer, train_args.n_layers)
        resolved_plot_layers.add(plot_layer)

        pacf_low = float(get_arg(train_args, "pacf_low", -0.9))
        pacf_high = float(get_arg(train_args, "pacf_high", 0.9))

        # Match the original scripts: every specialist is evaluated with the same
        # evaluation seed, but on data generated for its own AR order.
        torch.manual_seed(args.seed)
        batch = generate_arp_batch(
            batch_size=args.eval_n,
            context_length=train_args.context_length,
            ar_order=order,
            pacf_low=pacf_low,
            pacf_high=pacf_high,
            noise_std=train_args.noise_std,
            burn_in=train_args.burn_in,
            device=device,
        )
        context, true_horizon, conditional_mean_uv = ar_batch_to_uvmodel(batch)
        conditional_mean = conditional_mean_uv[:, 0, 0]

        # Perturbation is collected before monkey-patching attention capture, so
        # those repeated forward passes do not retain attention tensors.
        print("Collecting perturbation per-example values...")
        perturbation = collect_perturbation(
            model,
            context,
            true_horizon,
            conditional_mean,
            batch_size=args.batch_size,
            mode=args.perturbation,
            noise_std=train_args.noise_std,
            seed=args.seed + 1,
        )

        print("Collecting attention per-example values...")
        attention = collect_attention(
            model,
            context,
            true_horizon,
            n_layers=train_args.n_layers,
            context_length=train_args.context_length,
            batch_size=args.batch_size,
        )

        attention_rows = attention_rows_for_order(
            order,
            checkpoint_path,
            attention,
        )
        perturbation_rows = perturbation_rows_for_order(
            order,
            checkpoint_path,
            perturbation,
            args.perturbation,
        )
        all_attention_rows.extend(attention_rows)
        all_perturbation_rows.extend(perturbation_rows)

        raw_path: Path | None = None
        if not args.no_save_per_example:
            raw_path = output_dir / f"{args.run_name}_ar{order}_per_example.npz"
            save_per_example_npz(
                raw_path,
                order=order,
                batch=batch,
                perturbation=perturbation,
                attention=attention,
            )
            print(f"Wrote {raw_path}")

        selected_attention = rows_for_order(
            attention_rows,
            order,
            layer=plot_layer,
        )
        selected_perturbation = rows_for_order(perturbation_rows, order)
        top_attention = max(
            selected_attention,
            key=lambda row: float(row["context_normalized_attention"]),
        )
        top_perturbation = max(
            selected_perturbation,
            key=lambda row: float(row["mean_abs_delta"]),
        )

        model_summaries.append(
            {
                "ar_order": order,
                "checkpoint": str(checkpoint_path),
                "train_config": selected_train_config(train_args),
                "plot_attention_layer": plot_layer,
                "top_attention_lag": int(top_attention["lag"]),
                "top_perturbation_lag": int(top_perturbation["lag"]),
                "base_median_mse_to_conditional_mean": float(
                    perturbation["base_squared_error_per_example"].mean()
                ),
                "per_example_npz": str(raw_path) if raw_path else None,
            }
        )

        print(
            f"Top attention lag (layer {plot_layer}): {int(top_attention['lag'])}; "
            f"top perturbation lag: {int(top_perturbation['lag'])}"
        )

        del checkpoint, model, batch, context, true_horizon, conditional_mean_uv
        del conditional_mean, perturbation, attention
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(resolved_plot_layers) != 1:
        raise ValueError(
            "The requested attention layer resolves to different indices across models: "
            f"{sorted(resolved_plot_layers)}"
        )
    plot_layer = next(iter(resolved_plot_layers))

    attention_csv = output_dir / f"{args.run_name}_attention_by_lag.csv"
    perturbation_csv = output_dir / f"{args.run_name}_perturbation_by_lag.csv"
    write_csv(attention_csv, all_attention_rows)
    write_csv(perturbation_csv, all_perturbation_rows)

    plot_paths = save_plots(
        all_attention_rows,
        all_perturbation_rows,
        orders=orders,
        layer=plot_layer,
        std_multiplier=args.std_multiplier,
        output_dir=output_dir,
        run_name=args.run_name,
        dpi=args.dpi,
    )

    summary = {
        "repository_root": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "orders": orders,
        "device": str(device),
        "eval_n": args.eval_n,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "perturbation": args.perturbation,
        "attention_layer_plotted": plot_layer,
        "std_multiplier_plotted": args.std_multiplier,
        "standard_deviation_definition": {
            "attention": (
                "Population std across evaluation examples after averaging "
                "context-normalized attention over heads for each example."
            ),
            "perturbation": (
                "Population std across evaluation examples of "
                "abs(q0.5_perturbed - q0.5_base) at each lag."
            ),
            "lower_plot_band": "Clipped to zero because both plotted metrics are non-negative.",
        },
        "files": {
            "attention_csv": str(attention_csv),
            "perturbation_csv": str(perturbation_csv),
            **plot_paths,
        },
        "models": model_summaries,
        "attention_rows": all_attention_rows,
        "perturbation_rows": all_perturbation_rows,
    }
    summary_path = output_dir / f"{args.run_name}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\nFinished.")
    print(f"Wrote {attention_csv}")
    print(f"Wrote {perturbation_csv}")
    for path in plot_paths.values():
        print(f"Wrote {path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

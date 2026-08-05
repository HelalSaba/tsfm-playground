"""AR(p) sanity experiment using the UVModel skeleton.

The model is trained with quantile loss on the noisy next observation.
For evaluation, the median forecast q0.5 is compared against the analytic
AR(p) conditional mean.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from playground.data.ar import (
    ARBatch,
    ar_batch_to_uvmodel,
    arp_ols_forecast_by_order,
    generate_arp_batch,
    generate_mixed_arp_batch,
)
from playground.model.registry.uv import UVModel
from playground.training.loss import compute_quantile_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UVModel on synthetic AR(p) or mixed AR orders and evaluate q0.5 vs analytic conditional mean.")

    # Data / training.
    parser.add_argument("--ar-order", type=str, default="1", help="AR order integer, or 'mixed' for mixed-order training")
    parser.add_argument("--min-ar-order", type=int, default=1)
    parser.add_argument("--max-ar-order", type=int, default=5)
    parser.add_argument("--pacf-low", type=float, default=-0.9)
    parser.add_argument("--pacf-high", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-n", type=int, default=4000)
    parser.add_argument("--selection-eval-n", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--burn-in", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-seed", type=int, default=1_000_003)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-every", type=int, default=250)

    # UVModel architecture. patch_size=1, patch_stride=1 is the no-patching/scalar-token condition.
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--patch-stride", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--d-kv", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--use-arcsinh", action="store_true")

    # Outputs.
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--run-name", type=str, default="ar_sanity_uvmodel")
    parser.add_argument("--save-last-too", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(device_arg)


def build_model(args: argparse.Namespace) -> UVModel:
    # compute_quantile_loss in the skeleton hard-codes quantiles 0.1, ..., 0.9.
    n_quantiles = 9
    return UVModel(
        d_model=args.d_model,
        d_ff=args.d_ff,
        d_kv=args.d_kv,
        n_heads=args.n_heads,
        dropout=args.dropout,
        activation_fn=args.activation,
        n_quantiles=n_quantiles,
        n_encoder_layers=args.n_layers,
        pred_length=1,
        use_arcsinh=args.use_arcsinh,
        use_rope=args.use_rope,
        context_length=args.context_length,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
    )


def is_mixed_order(args: argparse.Namespace) -> bool:
    return str(args.ar_order).lower() == "mixed"


def fixed_ar_order(args: argparse.Namespace) -> int:
    if is_mixed_order(args):
        raise ValueError("ar_order is mixed, not a fixed integer")
    return int(args.ar_order)


def max_train_order(args: argparse.Namespace) -> int:
    return int(args.max_ar_order if is_mixed_order(args) else fixed_ar_order(args))


def order_label(args: argparse.Namespace) -> str:
    if is_mixed_order(args):
        return f"mixed_{args.min_ar_order}_{args.max_ar_order}"
    return str(fixed_ar_order(args))


def generate_batch(args: argparse.Namespace, batch_size: int, device: torch.device) -> ARBatch:
    if is_mixed_order(args):
        return generate_mixed_arp_batch(
            batch_size=batch_size,
            context_length=args.context_length,
            min_ar_order=args.min_ar_order,
            max_ar_order=args.max_ar_order,
            pacf_low=args.pacf_low,
            pacf_high=args.pacf_high,
            noise_std=args.noise_std,
            burn_in=args.burn_in,
            device=device,
        )
    return generate_arp_batch(
        batch_size=batch_size,
        context_length=args.context_length,
        ar_order=fixed_ar_order(args),
        pacf_low=args.pacf_low,
        pacf_high=args.pacf_high,
        noise_std=args.noise_std,
        burn_in=args.burn_in,
        device=device,
    )


def make_fixed_eval_batch(args: argparse.Namespace, device: torch.device) -> ARBatch:
    # Keep validation deterministic and avoid consuming the training RNG stream.
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    torch.manual_seed(args.seed + args.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + args.eval_seed)

    batch = generate_batch(args, batch_size=args.selection_eval_n, device=device)

    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return batch


def train_step(model: UVModel, optimizer: torch.optim.Optimizer, args: argparse.Namespace, device: torch.device) -> float:
    batch = generate_batch(args, batch_size=args.batch_size, device=device)
    context, true_horizon, _ = ar_batch_to_uvmodel(batch)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # UVModel returns an internally-computed loss, but in the current skeleton that loss is computed
    # before inverse-normalizing the forecast. We ignore it and compute quantile loss on the returned
    # unscaled forecast so prediction and target are on the same scale.
    internal_loss, quantile_forecast = model(context=context, n_horizon=1, true_horizon=true_horizon)
    del internal_loss

    loss = compute_quantile_loss(quantile_forecast, true_horizon.squeeze(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def no_intercept_slope(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    numerator = (pred * target).sum()
    denominator = target.square().sum().clamp_min(eps)
    return float((numerator / denominator).cpu())


def corrcoef(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = (x0.square().sum().sqrt() * y0.square().sum().sqrt()).clamp_min(eps)
    return float(((x0 * y0).sum() / denom).cpu())


def regression_metrics(name: str, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    err = pred - target
    return {
        f"{name}_mse_to_conditional_mean": float(err.square().mean().cpu()),
        f"{name}_mae_to_conditional_mean": float(err.abs().mean().cpu()),
        f"{name}_corr_with_conditional_mean": corrcoef(pred, target),
        f"{name}_slope_vs_conditional_mean": no_intercept_slope(pred, target),
    }


@torch.no_grad()
def evaluate(
    model: UVModel,
    args: argparse.Namespace,
    device: torch.device,
    batch: ARBatch | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if batch is None:
        batch = generate_batch(args, batch_size=args.eval_n, device=device)
    context, true_horizon, conditional_mean_uv = ar_batch_to_uvmodel(batch)

    model.eval()
    internal_loss, quantile_forecast = model(context=context, n_horizon=1, true_horizon=true_horizon)
    del internal_loss

    median_idx = 4  # quantiles are fixed as 0.1, ..., 0.9, so index 4 is q0.5.
    median_pred = quantile_forecast[:, median_idx, 0]
    conditional_mean = conditional_mean_uv[:, 0, 0]
    noisy_target = batch.target

    orders = batch.orders
    if orders is None:
        orders = torch.full((len(batch.target),), fixed_ar_order(args), device=device, dtype=torch.long)
    ols_forecast, coeff_hat = arp_ols_forecast_by_order(batch.context, orders, max_ar_order=batch.coeffs.shape[1])
    zero_forecast = torch.zeros_like(conditional_mean)
    last_value_forecast = batch.context[:, -1]

    metrics: dict[str, Any] = {
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_n": len(batch.target),
        "context_length": args.context_length,
        "burn_in": args.burn_in,
        "noise_std": args.noise_std,
        "irreducible_noise_mse_expected": args.noise_std**2,
        "ar_order": order_label(args),
        "min_ar_order": int(args.min_ar_order) if is_mixed_order(args) else fixed_ar_order(args),
        "max_ar_order": max_train_order(args),
        "pacf_low": args.pacf_low,
        "pacf_high": args.pacf_high,
        "patch_size": args.patch_size,
        "patch_stride": args.patch_stride,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "d_kv": args.d_kv,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "activation": args.activation,
        "use_rope": args.use_rope,
        "use_arcsinh": args.use_arcsinh,
        "n_quantiles": 9,
        "median_quantile_index": median_idx,
        "median_quantile_value": 0.5,
        "eval_quantile_loss_to_noisy_target": float(
            compute_quantile_loss(quantile_forecast, true_horizon.squeeze(-1)).cpu()
        ),
        "median_mse_to_noisy_next_x": float((median_pred - noisy_target).square().mean().cpu()),
        "coeff_mean_abs": float(batch.coeffs.abs().mean().cpu()),
        "pacf_mean_abs": float(batch.pacf.abs().mean().cpu()),
        "active_pacf_mean_abs": float(batch.pacf[(torch.arange(batch.pacf.shape[1], device=device).unsqueeze(0) < orders.unsqueeze(1))].abs().mean().cpu()),
        "ols_coeff_hat_mse_to_true_coeffs": float((coeff_hat - batch.coeffs).square().mean().cpu()),
    }
    metrics.update(regression_metrics("median", median_pred, conditional_mean))
    metrics.update(regression_metrics("ols_baseline", ols_forecast, conditional_mean))
    metrics.update(regression_metrics("zero_baseline", zero_forecast, conditional_mean))
    metrics.update(regression_metrics("last_value_baseline", last_value_forecast, conditional_mean))

    # Backward-compatible aliases for old AR(1) notebooks/logs.
    if (not is_mixed_order(args)) and fixed_ar_order(args) == 1:
        for key, value in list(metrics.items()):
            if key.endswith("_conditional_mean") or "_conditional_mean" in key:
                metrics[key.replace("conditional_mean", "phi_xt")] = value

    tensors = {
        "median_pred": median_pred.detach().cpu(),
        "conditional_mean": conditional_mean.detach().cpu(),
        "ols_forecast": ols_forecast.detach().cpu(),
    }
    return metrics, tensors


def save_scatter(tensors: dict[str, torch.Tensor], output_path: Path, ar_label: str) -> None:
    target = tensors["conditional_mean"].numpy()
    median_pred = tensors["median_pred"].numpy()
    ols_pred = tensors["ols_forecast"].numpy()

    max_points = min(1500, len(target))
    target = target[:max_points]
    median_pred = median_pred[:max_points]
    ols_pred = ols_pred[:max_points]

    lo = float(min(target.min(), median_pred.min(), ols_pred.min()))
    hi = float(max(target.max(), median_pred.max(), ols_pred.max()))

    plt.figure(figsize=(6.5, 6.0))
    plt.scatter(target, median_pred, s=8, alpha=0.35, label="UVModel q0.5")
    plt.scatter(target, ols_pred, s=8, alpha=0.25, label="OLS baseline")
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, label="ideal")
    plt.xlabel("Analytic conditional mean")
    plt.ylabel("Forecast")
    plt.title(f"AR({ar_label}) sanity: median forecast vs analytic optimum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.patch_size < 1 or args.patch_stride < 1:
        raise ValueError("patch_size and patch_stride must be positive")
    if args.d_model != args.n_heads * args.d_kv:
        raise ValueError("For this script, require d_model == n_heads * d_kv")
    if is_mixed_order(args) and args.min_ar_order > args.max_ar_order:
        raise ValueError("min_ar_order must be <= max_ar_order")
    if args.context_length <= max_train_order(args):
        raise ValueError("context_length must be greater than the maximum AR order")
    if not args.use_rope:
        print("WARNING: UVModel has no learned positional embedding; --use-rope is recommended.")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    fixed_eval_batch = make_fixed_eval_batch(args, device)

    best_metric = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None

    train_losses: list[float] = []
    for step in range(1, args.steps + 1):
        loss = train_step(model, optimizer, args, device)
        train_losses.append(loss)

        should_eval = step == 1 or step % args.eval_every == 0 or step == args.steps
        if should_eval:
            selection_metrics, _ = evaluate(model, args, device, batch=fixed_eval_batch)
            selection_metric = selection_metrics["median_mse_to_conditional_mean"]
            is_best = selection_metric < best_metric
            if is_best:
                best_metric = selection_metric
                best_step = step
                best_state = copy.deepcopy(model.state_dict())

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                suffix = " | best" if is_best else ""
                print(
                    f"step {step:04d}/{args.steps} | train_quantile_loss={loss:.4f} | "
                    f"median_mse_to_conditional_mean={selection_metric:.6f}{suffix}"
                )

    if best_state is None:
        raise RuntimeError("No best checkpoint was selected")

    last_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    metrics, tensors = evaluate(model, args, device, batch=fixed_eval_batch)
    metrics["train_loss_first"] = train_losses[0]
    metrics["train_loss_last"] = train_losses[-1]
    metrics["train_loss_last_50_mean"] = sum(train_losses[-50:]) / min(50, len(train_losses))
    metrics["selection_metric"] = "median_mse_to_conditional_mean"
    metrics["best_selection_metric_value"] = best_metric
    metrics["best_step"] = best_step
    metrics["selection_eval_n"] = len(fixed_eval_batch.target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.run_name}_best_metrics.json"
    scatter_path = args.output_dir / f"{args.run_name}_best_scatter.png"
    model_path = args.output_dir / f"{args.run_name}_best_model.pt"

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    save_scatter(tensors, scatter_path, ar_label=order_label(args))
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "metrics": metrics}, model_path)

    if args.save_last_too:
        last_path = args.output_dir / f"{args.run_name}_last_model.pt"
        torch.save({"model_state_dict": last_state, "args": vars(args)}, last_path)

    print("\nFinal metrics for best checkpoint on fixed evaluation set")
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print(f"\nSelected best checkpoint from step {best_step} using median_mse_to_conditional_mean={best_metric:.6f}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {scatter_path}")
    print(f"Wrote {model_path}")


if __name__ == "__main__":
    main()

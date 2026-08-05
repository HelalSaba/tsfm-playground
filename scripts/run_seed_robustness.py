"""Run the scalar AR specialist seed-robustness experiment.

The script is deliberately self-contained at the orchestration level: it reuses
only the existing ``src/playground`` model, data and loss components.

It separates three seed roles:

* training seed: varied across independently trained specialists;
* checkpoint-selection seed: fixed for every training seed;
* evaluation seed: five fixed seeds by default for robustness analysis.

The output is JSON-first so a separate notebook can load, aggregate and plot the
results without rerunning training.

Recommended commands (Windows CMD or PowerShell):

    python scripts/run_seed_robustness.py --stage pilot --device auto
    python scripts/run_seed_robustness.py --stage main --device auto

``pilot`` trains AR(6) with five seeds. ``main`` uses the same seed prefix and
runs AR(2), AR(6) and AR(7) with six seeds each, so completed pilot runs are
reused automatically. All defaults can be overridden from the command line.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

# Allow execution from a fresh checkout without ``pip install -e .``.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from playground.data.ar import ARBatch, ar_batch_to_uvmodel, arp_ols_forecast, generate_arp_batch
from playground.model.registry.uv import UVModel
from playground.training.loss import compute_quantile_loss


SCHEMA_VERSION = 1
DEFAULT_MASTER_SEED = 20_260_721
DEFAULT_SELECTION_SEED = 1_000_003
DEFAULT_EVAL_SEEDS = [123, 456, 789, 2027, 31_415]
DEFAULT_STEPS_BY_ORDER = {1: 5000, 2: 5000, 3: 5000, 4: 5000, 5: 5000, 6: 8000, 7: 8000}
QUANTILE_VALUES = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


@dataclass(frozen=True)
class RunPlan:
    orders: list[int]
    train_seeds: list[int]
    eval_seeds: list[int]
    steps_by_order: dict[int, int]


class RunLogger:
    """Minimal console/file logger for long command-line runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str = "") -> None:
        line = str(message)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_signature(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_key_value_ints(values: Sequence[str] | None) -> dict[int, int]:
    result: dict[int, int] = {}
    for item in values or []:
        separator = "=" if "=" in item else ":" if ":" in item else None
        if separator is None:
            raise ValueError(f"Expected ORDER=STEPS or ORDER:STEPS, got {item!r}")
        left, right = item.split(separator, maxsplit=1)
        order = int(left)
        steps = int(right)
        if order < 1 or steps < 1:
            raise ValueError(f"Invalid steps override {item!r}")
        result[order] = steps
    return result


def generate_train_seeds(master_seed: int, count: int, include_seed: int = 7) -> list[int]:
    if count < 1:
        raise ValueError("The number of training seeds must be at least one")
    seeds = [int(include_seed)]
    rng = random.Random(master_seed)
    while len(seeds) < count:
        candidate = rng.randrange(1, 2**31 - 1)
        if candidate not in seeds:
            seeds.append(candidate)
    return seeds[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate scalar AR specialists across training and evaluation seeds."
    )
    parser.add_argument(
        "--stage",
        choices=["pilot", "main", "custom"],
        default="pilot",
        help=(
            "pilot: AR(6), five training seeds; main: AR(2), AR(6), AR(7), six training seeds; "
            "custom: requires --orders or uses AR(6)."
        ),
    )
    parser.add_argument("--orders", type=int, nargs="+", default=None, help="Override AR orders for the selected stage")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=None, help="Explicit training seeds")
    parser.add_argument("--num-train-seeds", type=int, default=None, help="Number of deterministic seeds to generate")
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)

    parser.add_argument("--steps", type=int, default=None, help="Use one training length for every AR order")
    parser.add_argument(
        "--steps-per-order",
        nargs="*",
        default=None,
        metavar="ORDER=STEPS",
        help="Per-order overrides, for example --steps-per-order 2=5000 6=8000 7=8000",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--selection-eval-n", type=int, default=4000)
    parser.add_argument("--eval-n", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--burn-in", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--pacf-low", type=float, default=-0.9)
    parser.add_argument("--pacf-high", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--d-kv", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--use-arcsinh", action="store_true")

    parser.add_argument("--seasonality", type=int, default=1)
    parser.add_argument("--perturbation", choices=["permute", "zero", "noise"], default="permute")
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--attention-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--torch-num-threads", type=int, default=4, help="CPU intra-op threads; avoids severe oversubscription"
    )
    parser.add_argument("--torch-num-interop-threads", type=int, default=1, help="CPU inter-op threads")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "seed_robustness",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--eval-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite completed runs even if signatures match")
    parser.add_argument("--save-last-too", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="Do not request deterministic PyTorch algorithms",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Debugging aid: stop after this many order/training-seed combinations",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = [
        "batch_size",
        "selection_eval_n",
        "eval_n",
        "eval_every",
        "log_every",
        "context_length",
        "burn_in",
        "seasonality",
        "inference_batch_size",
        "attention_batch_size",
        "torch_num_threads",
        "torch_num_interop_threads",
    ]
    for name in positive_int_fields:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.d_model != args.n_heads * args.d_kv:
        raise ValueError("Require d_model == n_heads * d_kv")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("--dropout must lie in [0, 1)")
    if not (-1.0 < args.pacf_low < args.pacf_high < 1.0):
        raise ValueError("Require -1 < pacf-low < pacf-high < 1")
    if args.noise_std <= 0:
        raise ValueError("--noise-std must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer hyperparameters")
    if not args.eval_seeds:
        raise ValueError("At least one evaluation seed is required")
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        raise ValueError("Evaluation seeds must be unique")
    if args.train_seeds and len(set(args.train_seeds)) != len(args.train_seeds):
        raise ValueError("Training seeds must be unique")
    all_seeds = [args.master_seed, args.selection_seed, *args.eval_seeds, *(args.train_seeds or [])]
    if any(seed < 0 or seed >= 2**31 - 1 for seed in all_seeds):
        raise ValueError("Seeds must lie in [0, 2**31 - 2]")


def make_plan(args: argparse.Namespace) -> RunPlan:
    if args.stage == "pilot":
        default_orders = [6]
        default_seed_count = 5
    elif args.stage == "main":
        default_orders = [2, 6, 7]
        default_seed_count = 6
    else:
        default_orders = [6]
        default_seed_count = 5

    orders = sorted(set(args.orders if args.orders is not None else default_orders))
    if any(order < 1 for order in orders):
        raise ValueError("AR orders must be positive")
    if any(order >= args.context_length for order in orders):
        raise ValueError("Every AR order must be smaller than the context length")
    if args.burn_in < max(orders):
        raise ValueError("burn-in must be at least the largest AR order")

    if args.train_seeds is not None:
        train_seeds = [int(seed) for seed in args.train_seeds]
    else:
        count = int(args.num_train_seeds or default_seed_count)
        train_seeds = generate_train_seeds(args.master_seed, count)

    step_overrides = parse_key_value_ints(args.steps_per_order)
    steps_by_order: dict[int, int] = {}
    for order in orders:
        if args.steps is not None:
            steps = int(args.steps)
        elif order in step_overrides:
            steps = step_overrides[order]
        else:
            steps = DEFAULT_STEPS_BY_ORDER.get(order, 8000)
        steps_by_order[order] = steps

    return RunPlan(
        orders=orders,
        train_seeds=train_seeds,
        eval_seeds=[int(seed) for seed in args.eval_seeds],
        steps_by_order=steps_by_order,
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(device_arg)


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def move_batch(batch: ARBatch, device: torch.device) -> ARBatch:
    return ARBatch(
        context=batch.context.to(device),
        target=batch.target.to(device),
        conditional_mean=batch.conditional_mean.to(device),
        coeffs=batch.coeffs.to(device),
        pacf=batch.pacf.to(device),
        orders=None if batch.orders is None else batch.orders.to(device),
    )


def generate_seeded_eval_batch(config: dict[str, Any], seed: int, n: int, device: torch.device) -> ARBatch:
    """Generate on CPU under an isolated RNG state, then move to the run device.

    CPU generation keeps evaluation sets identical when a run is moved between
    CPU and CUDA and prevents evaluation generation from consuming training RNG.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        batch = generate_arp_batch(
            batch_size=n,
            context_length=int(config["context_length"]),
            ar_order=int(config["ar_order"]),
            pacf_low=float(config["pacf_low"]),
            pacf_high=float(config["pacf_high"]),
            noise_std=float(config["noise_std"]),
            burn_in=int(config["burn_in"]),
            device="cpu",
        )
    return move_batch(batch, device)


def generate_training_batch(config: dict[str, Any], device: torch.device) -> ARBatch:
    return generate_arp_batch(
        batch_size=int(config["batch_size"]),
        context_length=int(config["context_length"]),
        ar_order=int(config["ar_order"]),
        pacf_low=float(config["pacf_low"]),
        pacf_high=float(config["pacf_high"]),
        noise_std=float(config["noise_std"]),
        burn_in=int(config["burn_in"]),
        device=device,
    )


def build_model(config: dict[str, Any]) -> UVModel:
    return UVModel(
        d_model=int(config["d_model"]),
        d_ff=int(config["d_ff"]),
        d_kv=int(config["d_kv"]),
        n_heads=int(config["n_heads"]),
        dropout=float(config["dropout"]),
        activation_fn=str(config["activation"]),
        n_quantiles=9,
        n_encoder_layers=int(config["n_layers"]),
        pred_length=1,
        use_arcsinh=bool(config["use_arcsinh"]),
        use_rope=True,
        context_length=int(config["context_length"]),
        patch_size=1,
        patch_stride=1,
    )


def checkpoint_args(config: dict[str, Any]) -> dict[str, Any]:
    """Compatibility payload understood by the repository's evaluation scripts."""
    return {
        "ar_order": str(config["ar_order"]),
        "pacf_low": config["pacf_low"],
        "pacf_high": config["pacf_high"],
        "steps": config["steps"],
        "batch_size": config["batch_size"],
        "eval_n": config["selection_eval_n"],
        "selection_eval_n": config["selection_eval_n"],
        "eval_every": config["eval_every"],
        "log_every": config["log_every"],
        "context_length": config["context_length"],
        "burn_in": config["burn_in"],
        "noise_std": config["noise_std"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "seed": config["train_seed"],
        "eval_seed": config["selection_seed"],
        "selection_seed": config["selection_seed"],
        "patch_size": 1,
        "patch_stride": 1,
        "d_model": config["d_model"],
        "d_ff": config["d_ff"],
        "d_kv": config["d_kv"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "dropout": config["dropout"],
        "activation": config["activation"],
        "use_rope": True,
        "use_arcsinh": config["use_arcsinh"],
    }


def make_training_config(
    args: argparse.Namespace, order: int, train_seed: int, steps: int, device: torch.device
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "scalar_ar_seed_robustness",
        "ar_order": int(order),
        "train_seed": int(train_seed),
        "selection_seed": int(args.selection_seed),
        "training_device": str(device),
        "deterministic_algorithms_requested": not args.allow_nondeterministic,
        "steps": int(steps),
        "batch_size": int(args.batch_size),
        "selection_eval_n": int(args.selection_eval_n),
        "eval_every": int(args.eval_every),
        "log_every": int(args.log_every),
        "context_length": int(args.context_length),
        "burn_in": int(args.burn_in),
        "noise_std": float(args.noise_std),
        "pacf_low": float(args.pacf_low),
        "pacf_high": float(args.pacf_high),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patch_size": 1,
        "patch_stride": 1,
        "d_model": int(args.d_model),
        "d_ff": int(args.d_ff),
        "d_kv": int(args.d_kv),
        "n_heads": int(args.n_heads),
        "n_layers": int(args.n_layers),
        "dropout": float(args.dropout),
        "activation": str(args.activation),
        "use_rope": True,
        "use_arcsinh": bool(args.use_arcsinh),
        "torch_num_threads": int(args.torch_num_threads),
        "torch_num_interop_threads": int(args.torch_num_interop_threads),
        "checkpoint_selection_metric": "median_mse_to_conditional_mean",
        "selection_batch_is_fixed_across_training_seeds": True,
        "selection_batch_generated_on_cpu": True,
    }


def make_evaluation_config(
    args: argparse.Namespace,
    training_signature: str,
    order: int,
    train_seed: int,
    eval_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "scalar_ar_seed_robustness",
        "training_signature": training_signature,
        "ar_order": int(order),
        "train_seed": int(train_seed),
        "selection_seed": int(args.selection_seed),
        "evaluation_seed": int(eval_seed),
        "evaluation_device": str(device),
        "eval_n": int(args.eval_n),
        "seasonality": int(args.seasonality),
        "perturbation": str(args.perturbation),
        "inference_batch_size": int(args.inference_batch_size),
        "attention_batch_size": int(args.attention_batch_size),
        "torch_num_threads": int(args.torch_num_threads),
        "torch_num_interop_threads": int(args.torch_num_interop_threads),
        "evaluation_batch_generated_on_cpu": True,
        "forecast_metric_target": "realized noisy next value",
        "diagnostic_target": "analytic conditional mean",
    }


def train_step(
    model: UVModel,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: torch.device,
) -> float:
    batch = generate_training_batch(config, device)
    context, true_horizon, _ = ar_batch_to_uvmodel(batch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    internal_loss, quantile_forecast = model(context=context, n_horizon=1, true_horizon=true_horizon)
    del internal_loss
    loss = compute_quantile_loss(quantile_forecast, true_horizon.squeeze(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach().cpu())


@torch.inference_mode()
def quantile_forecast_batched(
    model: UVModel,
    context: torch.Tensor,
    true_horizon: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, context.shape[0], batch_size):
        end = min(start + batch_size, context.shape[0])
        _, forecast = model(
            context=context[start:end],
            n_horizon=1,
            true_horizon=true_horizon[start:end],
        )
        if forecast.ndim != 3 or forecast.shape[1:] != (9, 1):
            raise ValueError(f"Expected forecast shape (batch, 9, 1), got {tuple(forecast.shape)}")
        chunks.append(forecast[:, :, 0])
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def selection_mse(
    model: UVModel,
    batch: ARBatch,
    inference_batch_size: int,
) -> float:
    context, true_horizon, conditional_mean_uv = ar_batch_to_uvmodel(batch)
    forecast = quantile_forecast_batched(model, context, true_horizon, inference_batch_size)
    median = forecast[:, 4]
    conditional_mean = conditional_mean_uv[:, 0, 0]
    return float((median - conditional_mean).square().mean().cpu())


def train_specialist(
    *,
    config: dict[str, Any],
    signature: str,
    device: torch.device,
    deterministic: bool,
    run_dir: Path,
    inference_batch_size: int,
    save_last_too: bool,
    logger: RunLogger,
) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    configure_reproducibility(int(config["train_seed"]), deterministic=deterministic)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    selection_batch = generate_seeded_eval_batch(
        config,
        seed=int(config["selection_seed"]),
        n=int(config["selection_eval_n"]),
        device=device,
    )

    best_metric = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    train_losses: list[float] = []
    selection_history: list[dict[str, Any]] = []

    logger.log(
        f"Training AR({config['ar_order']}) seed={config['train_seed']} for {config['steps']} steps "
        f"on {device}; fixed selection seed={config['selection_seed']}"
    )

    for step in range(1, int(config["steps"]) + 1):
        loss = train_step(model, optimizer, config, device)
        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}: {loss}")
        train_losses.append(loss)

        should_eval = step == 1 or step % int(config["eval_every"]) == 0 or step == int(config["steps"])
        if should_eval:
            metric = selection_mse(model, selection_batch, inference_batch_size)
            if not math.isfinite(metric):
                raise RuntimeError(f"Non-finite selection metric at step {step}: {metric}")
            is_best = metric < best_metric
            if is_best:
                best_metric = metric
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            selection_history.append(
                {
                    "step": step,
                    "train_quantile_loss": loss,
                    "median_mse_to_conditional_mean": metric,
                    "is_best_so_far": is_best,
                }
            )

            if step == 1 or step % int(config["log_every"]) == 0 or step == int(config["steps"]):
                suffix = " | best" if is_best else ""
                logger.log(
                    f"step {step:05d}/{config['steps']} | train_quantile_loss={loss:.6f} | "
                    f"selection_mse={metric:.6f}{suffix}"
                )
        elif step % int(config["log_every"]) == 0:
            logger.log(f"step {step:05d}/{config['steps']} | train_quantile_loss={loss:.6f}")

    if best_state is None:
        raise RuntimeError("No best checkpoint was selected")

    last_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    final_selection_metric = selection_mse(model, selection_batch, inference_batch_size)
    elapsed = time.perf_counter() - started

    checkpoint_path = run_dir / "best_model.pt"
    training_json_path = run_dir / "training_result.json"
    last_path = run_dir / "last_model.pt"

    training_result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "training_signature": signature,
        "training_config": config,
        "device": str(device),
        "deterministic_algorithms_requested": deterministic,
        "best_step": best_step,
        "best_selection_metric_value": best_metric,
        "recomputed_best_selection_metric_value": final_selection_metric,
        "selection_metric": "median_mse_to_conditional_mean",
        "selection_history": selection_history,
        "train_loss_first": train_losses[0],
        "train_loss_last": train_losses[-1],
        "train_loss_last_50_mean": float(np.mean(train_losses[-50:])),
        "train_loss_history": train_losses,
        "elapsed_seconds": elapsed,
        "checkpoint": repo_relative(checkpoint_path),
    }

    atomic_torch_save(
        checkpoint_path,
        {
            "model_state_dict": model.state_dict(),
            "args": checkpoint_args(config),
            "metrics": training_result,
            "seed_experiment": {
                "training_signature": signature,
                "selection_seed": config["selection_seed"],
            },
        },
    )
    if save_last_too:
        atomic_torch_save(
            last_path,
            {
                "model_state_dict": last_state,
                "args": checkpoint_args(config),
                "seed_experiment": {"training_signature": signature},
            },
        )
        training_result["last_checkpoint"] = repo_relative(last_path)

    atomic_write_json(training_json_path, training_result)
    logger.log(
        f"Finished training: best step={best_step}, selection MSE={best_metric:.6f}, elapsed={elapsed / 60.0:.1f} min"
    )
    return checkpoint_path, training_result


def load_model_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[UVModel, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = SimpleNamespace(**checkpoint["args"])
    config = {
        "ar_order": int(args.ar_order),
        "pacf_low": float(getattr(args, "pacf_low", -0.9)),
        "pacf_high": float(getattr(args, "pacf_high", 0.9)),
        "noise_std": float(args.noise_std),
        "burn_in": int(args.burn_in),
        "context_length": int(args.context_length),
        "d_model": int(args.d_model),
        "d_ff": int(args.d_ff),
        "d_kv": int(args.d_kv),
        "n_heads": int(args.n_heads),
        "n_layers": int(args.n_layers),
        "dropout": float(args.dropout),
        "activation": str(args.activation),
        "use_arcsinh": bool(args.use_arcsinh),
    }
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def corrcoef(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float | None:
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = x0.square().sum().sqrt() * y0.square().sum().sqrt()
    if float(denom.cpu()) <= eps:
        return None
    return float(((x0 * y0).sum() / denom).cpu())


def no_intercept_slope(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float | None:
    denominator = target.square().sum()
    if float(denominator.cpu()) <= eps:
        return None
    return float(((pred * target).sum() / denominator).cpu())


def mase_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    insample: torch.Tensor,
    seasonality: int,
    eps: float = 1e-8,
) -> tuple[float, float, float, int]:
    naive_scale = (insample[:, seasonality:] - insample[:, :-seasonality]).abs().mean(dim=1)
    valid = naive_scale > eps
    if not bool(valid.any()):
        raise ValueError("MASE is undefined because every in-sample naive scale is zero")
    abs_error = (pred - target).abs()
    scaled_error = abs_error[valid] / naive_scale[valid]
    return (
        float(scaled_error.mean().cpu()),
        float(abs_error.mean().cpu()),
        float(naive_scale[valid].mean().cpu()),
        int((~valid).sum().item()),
    )


def smape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> tuple[float, float]:
    numerator = 2.0 * (pred - target).abs()
    denominator = pred.abs() + target.abs()
    values = torch.where(denominator > eps, numerator / denominator, torch.zeros_like(denominator))
    fraction = float(values.mean().cpu())
    return fraction, 100.0 * fraction


def weighted_quantile_loss(
    forecasts: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float, dict[str, float], float]:
    quantiles = QUANTILE_VALUES.to(device=forecasts.device, dtype=forecasts.dtype)
    error = target.view(-1, 1) - forecasts
    q = quantiles.view(1, -1)
    pinball = torch.maximum(q * error, (q - 1.0) * error)
    target_scale = target.abs().sum().clamp_min(eps)
    per_quantile = 2.0 * pinball.sum(dim=0) / target_scale
    per_q = {f"q{float(qv):.1f}": float(value.cpu()) for qv, value in zip(quantiles, per_quantile)}
    return float(per_quantile.mean().cpu()), per_q, float(target_scale.cpu())


def coefficient_summary(batch: ARBatch) -> dict[str, Any]:
    coeffs = batch.coeffs.detach().cpu()
    pacf = batch.pacf.detach().cpu()
    return {
        "coefficient_mean_by_lag": coeffs.mean(dim=0).tolist(),
        "coefficient_std_by_lag": coeffs.std(dim=0, unbiased=False).tolist(),
        "coefficient_mean_abs_by_lag": coeffs.abs().mean(dim=0).tolist(),
        "pacf_mean_by_lag": pacf.mean(dim=0).tolist(),
        "pacf_std_by_lag": pacf.std(dim=0, unbiased=False).tolist(),
        "pacf_mean_abs_by_lag": pacf.abs().mean(dim=0).tolist(),
        "target_mean": float(batch.target.mean().cpu()),
        "target_std": float(batch.target.std(unbiased=False).cpu()),
        "conditional_mean_std": float(batch.conditional_mean.std(unbiased=False).cpu()),
    }


def forecast_metrics(
    model: UVModel,
    batch: ARBatch,
    order: int,
    seasonality: int,
    inference_batch_size: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    context_uv, true_horizon, conditional_mean_uv = ar_batch_to_uvmodel(batch)
    all_quantiles = quantile_forecast_batched(model, context_uv, true_horizon, inference_batch_size)
    median = all_quantiles[:, 4]
    conditional_mean = conditional_mean_uv[:, 0, 0]
    target = batch.target
    ols, _ = arp_ols_forecast(batch.context, ar_order=order)
    point_predictions = {
        "median": median,
        "ols": ols,
        "analytic_optimum": conditional_mean,
        "last_value": batch.context[:, -1],
        "zero": torch.zeros_like(target),
    }

    metrics: dict[str, Any] = {
        "primary_point_metrics": ["MASE", "sMAPE"],
        "primary_probabilistic_metric": "mean weighted quantile loss across q=0.1,...,0.9",
        "forecast_metric_target": "realized noisy next value",
        "diagnostic_target": "analytic conditional mean",
        "mase_scaling": "per-series in-context seasonal-naive MAE, then averaged across series",
        "wql_definition": "2 * summed pinball loss / summed absolute realized target; averaged over quantiles",
        "median_mse_to_conditional_mean": float((median - conditional_mean).square().mean().cpu()),
        "median_mae_to_conditional_mean": float((median - conditional_mean).abs().mean().cpu()),
        "median_corr_with_conditional_mean_diagnostic": corrcoef(median, conditional_mean),
        "median_slope_vs_conditional_mean_diagnostic": no_intercept_slope(median, conditional_mean),
    }

    invalid_counts: list[int] = []
    for name, pred in point_predictions.items():
        mase, mae, naive_scale, invalid_count = mase_components(pred, target, batch.context, seasonality)
        smape_fraction, smape_percent = smape(pred, target)
        metrics[f"{name}_mase_to_noisy_next_x"] = mase
        metrics[f"{name}_mae_to_noisy_next_x"] = mae
        metrics[f"{name}_smape_fraction_to_noisy_next_x"] = smape_fraction
        metrics[f"{name}_smape_percent_to_noisy_next_x"] = smape_percent
        metrics["mase_naive_scale_mean"] = naive_scale
        invalid_counts.append(invalid_count)
    metrics["mase_invalid_scale_series"] = max(invalid_counts)

    mean_wql, per_quantile, target_scale = weighted_quantile_loss(all_quantiles, target)
    metrics["uvmodel_mean_wql_to_noisy_next_x"] = mean_wql
    metrics["uvmodel_wql_target_absolute_sum"] = target_scale
    metrics["uvmodel_wql_by_quantile"] = per_quantile
    return metrics, median


def rank_descending(values: Sequence[float], index: int) -> int:
    target = float(values[index])
    return 1 + sum(float(value) > target for value in values)


def top_indices(values: Sequence[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda idx: (-float(values[idx]), idx))[:count]


def top_p_recall(values: Sequence[float], order: int) -> float:
    top = top_indices(values, order)
    return sum(index < order for index in top) / order


def first_p_share(values: Sequence[float], order: int, positive_only: bool = False) -> float:
    adjusted = [max(float(value), 0.0) if positive_only else float(value) for value in values]
    total = sum(adjusted)
    return float(sum(adjusted[:order]) / total) if total > 0 else 0.0


def entropy(values: Sequence[float], eps: float = 1e-12) -> float:
    arr = np.asarray(values, dtype=np.float64)
    total = arr.sum()
    if total <= eps:
        return 0.0
    probs = arr / total
    probs = probs[probs > eps]
    return float(-(probs * np.log(probs)).sum())


def attach_attention_accumulator(model: UVModel, context_length: int) -> list[dict[str, Any]]:
    """Patch attention to accumulate only the horizon row, avoiding full-map storage."""
    captured: list[dict[str, Any]] = []
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
            horizon_to_context = weights[:, :, -1, :context_length]
            context_mass = horizon_to_context.sum(dim=-1)
            context_norm = horizon_to_context / context_mass.unsqueeze(-1).clamp_min(1e-8)
            captured.append(
                {
                    "layer": layer_idx,
                    "normalized_sum": context_norm.sum(dim=(0, 1)).detach().cpu(),
                    "raw_sum": horizon_to_context.sum(dim=(0, 1)).detach().cpu(),
                    "context_mass_sum": float(context_mass.sum().detach().cpu()),
                    "self_mass_sum": float(weights[:, :, -1, -1].sum().detach().cpu()),
                    "count": int(horizon_to_context.shape[0] * horizon_to_context.shape[1]),
                }
            )
            return torch.matmul(weights, v)

        mha._attention = wrapped_attention
    return captured


@torch.inference_mode()
def attention_metrics(
    checkpoint_path: Path,
    batch: ARBatch,
    device: torch.device,
    order: int,
    attention_batch_size: int,
) -> dict[str, Any]:
    model, model_config, _ = load_model_from_checkpoint(checkpoint_path, device)
    if model.patch_size != 1 or model.patch_stride != 1:
        raise ValueError("Seed robustness attention analysis requires scalar patch_size=1, patch_stride=1")

    context_uv, true_horizon, _ = ar_batch_to_uvmodel(batch)
    captured = attach_attention_accumulator(model, int(model_config["context_length"]))
    n_layers = int(model_config["n_layers"])
    context_length = int(model_config["context_length"])
    normalized_sums = torch.zeros(n_layers, context_length)
    raw_sums = torch.zeros(n_layers, context_length)
    context_mass_sums = torch.zeros(n_layers)
    self_mass_sums = torch.zeros(n_layers)
    counts = torch.zeros(n_layers)

    for start in range(0, context_uv.shape[0], attention_batch_size):
        end = min(start + attention_batch_size, context_uv.shape[0])
        captured.clear()
        model(context=context_uv[start:end], n_horizon=1, true_horizon=true_horizon[start:end])
        if len(captured) != n_layers:
            raise RuntimeError(f"Expected {n_layers} captured layers, got {len(captured)}")
        for item in captured:
            layer_idx = int(item["layer"])
            normalized_sums[layer_idx] += item["normalized_sum"]
            raw_sums[layer_idx] += item["raw_sum"]
            context_mass_sums[layer_idx] += item["context_mass_sum"]
            self_mass_sums[layer_idx] += item["self_mass_sum"]
            counts[layer_idx] += item["count"]

    layers: list[dict[str, Any]] = []
    for layer_idx in range(n_layers):
        norm_by_position = normalized_sums[layer_idx] / counts[layer_idx].clamp_min(1)
        raw_by_position = raw_sums[layer_idx] / counts[layer_idx].clamp_min(1)
        norm_by_lag = torch.flip(norm_by_position, dims=[0]).tolist()
        raw_by_lag = torch.flip(raw_by_position, dims=[0]).tolist()
        top_lag = top_indices(norm_by_lag, 1)[0] + 1
        layer_summary = {
            "layer": layer_idx,
            "context_normalized_attention_by_lag": norm_by_lag,
            "raw_attention_by_lag": raw_by_lag,
            "top_lag_by_context_normalized_attention": top_lag,
            "lag1_rank": rank_descending(norm_by_lag, 0),
            "lag2_rank": rank_descending(norm_by_lag, 1) if context_length >= 2 else None,
            "lag1_context_normalized_attention": norm_by_lag[0],
            "lag2_context_normalized_attention": norm_by_lag[1] if context_length >= 2 else None,
            "lag2_to_lag1_attention_ratio": (
                float(norm_by_lag[1] / max(norm_by_lag[0], 1e-12)) if context_length >= 2 else None
            ),
            "first_p_context_normalized_attention_mass": float(sum(norm_by_lag[:order])),
            "top_p_recall_by_context_normalized_attention": top_p_recall(norm_by_lag, order),
            "attention_entropy": entropy(norm_by_lag),
            "normalized_attention_entropy": entropy(norm_by_lag) / math.log(context_length),
            "mean_context_attention_mass": float(context_mass_sums[layer_idx] / counts[layer_idx].clamp_min(1)),
            "mean_horizon_self_attention_mass": float(self_mass_sums[layer_idx] / counts[layer_idx].clamp_min(1)),
        }
        layers.append(layer_summary)

    return {
        "method": "horizon-token attention averaged across examples and heads",
        "context_attention_is_renormalized_before_averaging": True,
        "layers": layers,
        "last_layer": layers[-1],
    }


def perturb_context(
    context: torch.Tensor,
    lag: int,
    mode: str,
    cpu_generator: torch.Generator,
    noise_std: float,
) -> torch.Tensor:
    perturbed = context.clone()
    position = context.shape[1] - lag
    if mode == "permute":
        permutation = torch.randperm(context.shape[0], generator=cpu_generator, device="cpu").to(context.device)
        perturbed[:, position] = context[permutation, position]
    elif mode == "zero":
        perturbed[:, position] = 0.0
    elif mode == "noise":
        noise = torch.randn(context.shape[0], generator=cpu_generator, device="cpu") * noise_std
        perturbed[:, position] = noise.to(context.device)
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")
    return perturbed


@torch.inference_mode()
def perturbation_metrics(
    model: UVModel,
    batch: ARBatch,
    base_median: torch.Tensor,
    order: int,
    mode: str,
    eval_seed: int,
    inference_batch_size: int,
    noise_std: float,
) -> dict[str, Any]:
    _, true_horizon, conditional_mean_uv = ar_batch_to_uvmodel(batch)
    conditional_mean = conditional_mean_uv[:, 0, 0]
    base_mse = float((base_median - conditional_mean).square().mean().cpu())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(eval_seed) + 1)

    rows: list[dict[str, Any]] = []
    for lag in range(1, batch.context.shape[1] + 1):
        perturbed_context = perturb_context(batch.context, lag, mode, generator, noise_std)
        perturbed_uv = perturbed_context.unsqueeze(-1)
        forecast = quantile_forecast_batched(model, perturbed_uv, true_horizon, inference_batch_size)
        perturbed_median = forecast[:, 4]
        delta = perturbed_median - base_median
        perturbed_mse = float((perturbed_median - conditional_mean).square().mean().cpu())
        rows.append(
            {
                "lag": lag,
                "is_within_ar_order": lag <= order,
                "mean_abs_delta": float(delta.abs().mean().cpu()),
                "mean_squared_delta": float(delta.square().mean().cpu()),
                "mse_to_conditional_mean_after_perturb": perturbed_mse,
                "mse_increase_to_conditional_mean": perturbed_mse - base_mse,
            }
        )

    mean_abs = [row["mean_abs_delta"] for row in rows]
    mse_increase = [row["mse_increase_to_conditional_mean"] for row in rows]
    return {
        "perturbation": mode,
        "permutation_seed": int(eval_seed) + 1,
        "base_median_mse_to_conditional_mean": base_mse,
        "mean_abs_delta_by_lag": mean_abs,
        "mse_increase_to_conditional_mean_by_lag": mse_increase,
        "rows": rows,
        "top_lag_by_mean_abs_delta": top_indices(mean_abs, 1)[0] + 1,
        "top_lag_by_mse_increase": top_indices(mse_increase, 1)[0] + 1,
        "lag1_rank_by_mean_abs_delta": rank_descending(mean_abs, 0),
        "lag2_rank_by_mean_abs_delta": rank_descending(mean_abs, 1),
        "lag1_rank_by_mse_increase": rank_descending(mse_increase, 0),
        "lag2_rank_by_mse_increase": rank_descending(mse_increase, 1),
        "top_p_recall_by_mean_abs_delta": top_p_recall(mean_abs, order),
        "top_p_recall_by_mse_increase": top_p_recall(mse_increase, order),
        "first_p_share_of_mean_abs_delta": first_p_share(mean_abs, order),
        "first_p_share_of_positive_mse_increase": first_p_share(mse_increase, order, positive_only=True),
    }


def average_ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    start = 0
    while start < len(arr):
        end = start + 1
        while end < len(arr) and arr[order[end]] == arr[order[start]]:
            end += 1
        average = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    rx = average_ranks(x)
    ry = average_ranks(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if math.isfinite(value) else None


def alignment_metrics(attention: dict[str, Any], perturbation: dict[str, Any], order: int) -> dict[str, Any]:
    attn = [float(value) for value in attention["last_layer"]["context_normalized_attention_by_lag"]]
    mse = [float(value) for value in perturbation["mse_increase_to_conditional_mean_by_lag"]]
    delta = [float(value) for value in perturbation["mean_abs_delta_by_lag"]]
    attn_top = set(top_indices(attn, order))
    mse_top = set(top_indices(mse, order))
    delta_top = set(top_indices(delta, order))

    lag1_attn_rank = rank_descending(attn, 0)
    lag1_mse_rank = rank_descending(mse, 0)
    top_attn_lag = top_indices(attn, 1)[0] + 1
    top_mse_lag = top_indices(mse, 1)[0] + 1

    underrepresents_lag1 = lag1_attn_rank > order and lag1_mse_rank <= 2
    strict_ar6_pattern = (
        order == 6 and top_attn_lag == 2 and lag1_attn_rank > 6 and top_mse_lag == 2 and lag1_mse_rank <= 2
    )
    return {
        "last_layer_attention_vs_mse_increase_spearman": spearman_correlation(attn, mse),
        "last_layer_attention_vs_mean_abs_delta_spearman": spearman_correlation(attn, delta),
        "top_p_attention_mse_overlap_fraction": len(attn_top & mse_top) / order,
        "top_p_attention_mean_abs_delta_overlap_fraction": len(attn_top & delta_top) / order,
        "top_attention_and_mse_lag_agree": top_attn_lag == top_mse_lag,
        "top_attention_lag": top_attn_lag,
        "top_mse_increase_lag": top_mse_lag,
        "lag1_attention_rank": lag1_attn_rank,
        "lag1_mse_increase_rank": lag1_mse_rank,
        "lag1_attention_minus_mse_rank": lag1_attn_rank - lag1_mse_rank,
        "lag2_to_lag1_attention_ratio": float(attn[1] / max(attn[0], 1e-12)),
        "attention_underrepresents_lag1_relative_to_sensitivity": underrepresents_lag1,
        "ar6_report_pattern_strict": strict_ar6_pattern,
        "ar6_pattern_definition": (
            "AR(6) only: top attention lag=2, lag-1 attention rank>6, top MSE-increase lag=2, "
            "and lag-1 MSE-increase rank<=2"
        ),
    }


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    training_signature: str,
    evaluation_config: dict[str, Any],
    evaluation_signature: str,
    device: torch.device,
    result_path: Path,
    logger: RunLogger,
) -> dict[str, Any]:
    started = time.perf_counter()
    model, model_config, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    order = int(evaluation_config["ar_order"])
    eval_seed = int(evaluation_config["evaluation_seed"])
    batch_generation_config = {
        "ar_order": order,
        "context_length": model_config["context_length"],
        "burn_in": model_config["burn_in"],
        "noise_std": model_config["noise_std"],
        "pacf_low": model_config["pacf_low"],
        "pacf_high": model_config["pacf_high"],
    }
    batch = generate_seeded_eval_batch(
        batch_generation_config,
        seed=eval_seed,
        n=int(evaluation_config["eval_n"]),
        device=device,
    )

    logger.log(
        f"Evaluating AR({order}) train_seed={evaluation_config['train_seed']} "
        f"eval_seed={eval_seed} (n={evaluation_config['eval_n']})"
    )
    forecast, base_median = forecast_metrics(
        model,
        batch,
        order=order,
        seasonality=int(evaluation_config["seasonality"]),
        inference_batch_size=int(evaluation_config["inference_batch_size"]),
    )
    perturbation = perturbation_metrics(
        model,
        batch,
        base_median=base_median,
        order=order,
        mode=str(evaluation_config["perturbation"]),
        eval_seed=eval_seed,
        inference_batch_size=int(evaluation_config["inference_batch_size"]),
        noise_std=float(model_config["noise_std"]),
    )
    attention = attention_metrics(
        checkpoint_path,
        batch,
        device=device,
        order=order,
        attention_batch_size=int(evaluation_config["attention_batch_size"]),
    )
    alignment = alignment_metrics(attention, perturbation, order)
    elapsed = time.perf_counter() - started

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "training_signature": training_signature,
        "evaluation_signature": evaluation_signature,
        "evaluation_config": evaluation_config,
        "checkpoint": repo_relative(checkpoint_path),
        "checkpoint_best_step": checkpoint.get("metrics", {}).get("best_step"),
        "device": str(device),
        "forecast_metrics": forecast,
        "coefficient_regime_summary": coefficient_summary(batch),
        "attention": attention,
        "perturbation": perturbation,
        "attention_sensitivity_alignment": alignment,
        "elapsed_seconds": elapsed,
    }
    atomic_write_json(result_path, result)
    logger.log(
        f"Evaluation finished in {elapsed:.1f}s | MASE={forecast['median_mase_to_noisy_next_x']:.4f} | "
        f"WQL={forecast['uvmodel_mean_wql_to_noisy_next_x']:.4f} | "
        f"top-attn={alignment['top_attention_lag']} | top-pert={alignment['top_mse_increase_lag']} | "
        f"lag1 ranks attn/pert={alignment['lag1_attention_rank']}/{alignment['lag1_mse_increase_rank']}"
    )
    return result


def system_metadata() -> dict[str, Any]:
    git_commit = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "git_commit": git_commit,
    }


def update_manifest(args: argparse.Namespace, plan: RunPlan, device: torch.device) -> None:
    path = args.output_dir / "experiment_manifest.json"
    if path.exists():
        manifest = load_json(path)
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "scalar_ar_seed_robustness",
            "created_at_utc": utc_now(),
            "system": system_metadata(),
            "invocations": [],
        }
    manifest["updated_at_utc"] = utc_now()
    manifest["seed_roles"] = {
        "training_seed": "varied; controls model initialization and stochastic training stream",
        "selection_seed": "fixed; controls a common checkpoint-selection batch across training seeds",
        "evaluation_seed": "varied; controls held-out coefficients, series and perturbation permutations",
    }
    manifest["default_eval_seeds"] = DEFAULT_EVAL_SEEDS
    manifest["canonical_eval_seed"] = plan.eval_seeds[0]
    manifest["invocations"].append(
        {
            "timestamp_utc": utc_now(),
            "stage": args.stage,
            "orders": plan.orders,
            "train_seeds": plan.train_seeds,
            "eval_seeds": plan.eval_seeds,
            "selection_seed": args.selection_seed,
            "steps_by_order": {str(key): value for key, value in plan.steps_by_order.items()},
            "device": str(device),
            "train_only": args.train_only,
            "eval_only": args.eval_only,
            "force": args.force,
            "deterministic_algorithms_requested": not args.allow_nondeterministic,
            "torch_num_threads": args.torch_num_threads,
            "torch_num_interop_threads": args.torch_num_interop_threads,
        }
    )
    atomic_write_json(path, manifest)


def build_index(args: argparse.Namespace, plan: RunPlan) -> None:
    training_runs: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for order in plan.orders:
        for train_seed in plan.train_seeds:
            run_dir = args.output_dir / f"ar{order}" / f"train_seed_{train_seed:010d}"
            training_path = run_dir / "training_result.json"
            checkpoint_path = run_dir / "best_model.pt"
            training_runs.append(
                {
                    "ar_order": order,
                    "train_seed": train_seed,
                    "status": "complete" if training_path.exists() and checkpoint_path.exists() else "missing",
                    "training_result": repo_relative(training_path),
                    "checkpoint": repo_relative(checkpoint_path),
                }
            )
            for eval_seed in plan.eval_seeds:
                result_path = run_dir / "evaluations" / f"eval_seed_{eval_seed:010d}.json"
                evaluations.append(
                    {
                        "ar_order": order,
                        "train_seed": train_seed,
                        "evaluation_seed": eval_seed,
                        "status": "complete" if result_path.exists() else "missing",
                        "result": repo_relative(result_path),
                    }
                )
    index = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "scalar_ar_seed_robustness",
        "updated_at_utc": utc_now(),
        "orders": plan.orders,
        "train_seeds": plan.train_seeds,
        "evaluation_seeds": plan.eval_seeds,
        "canonical_evaluation_seed": plan.eval_seeds[0],
        "training_runs": training_runs,
        "evaluations": evaluations,
    }
    atomic_write_json(args.output_dir / "experiment_index.json", index)


def validate_existing_signature(path: Path, key: str, expected: str, force: bool) -> bool:
    """Return True when a valid completed result can be reused."""
    if not path.exists() or force:
        return False
    payload = load_json(path)
    found = payload.get(key)
    if found != expected:
        raise RuntimeError(
            f"Existing result {path} has {key}={found!r}, expected {expected!r}. "
            "Use --force or choose a different --output-dir."
        )
    return payload.get("status") == "complete"


def validate_checkpoint_signature(checkpoint_path: Path, expected: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    found = checkpoint.get("seed_experiment", {}).get("training_signature")
    if found is not None and found != expected:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has training signature {found!r}, expected {expected!r}. "
            "Use the matching experiment configuration or retrain with --force."
        )


def write_failure(path: Path, context: dict[str, Any], error: BaseException) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "created_at_utc": utc_now(),
        "context": context,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    atomic_write_json(path, payload)


def print_plan(args: argparse.Namespace, plan: RunPlan, device: torch.device) -> None:
    total_trainings = len(plan.orders) * len(plan.train_seeds)
    total_evaluations = total_trainings * len(plan.eval_seeds)
    print("Seed robustness plan")
    print(f"  stage: {args.stage}")
    print(f"  device: {device}")
    print(f"  orders: {plan.orders}")
    print(f"  training seeds ({len(plan.train_seeds)}): {plan.train_seeds}")
    print(f"  fixed selection seed: {args.selection_seed}")
    print(f"  evaluation seeds ({len(plan.eval_seeds)}): {plan.eval_seeds}")
    print(f"  steps by order: {plan.steps_by_order}")
    print(f"  planned training runs: {total_trainings}")
    print(f"  planned evaluation bundles: {total_evaluations}")
    print(f"  PyTorch CPU threads: {args.torch_num_threads} intra-op, {args.torch_num_interop_threads} inter-op")
    print(f"  output directory: {args.output_dir}")
    if args.stage == "main":
        print("  note: matching completed pilot AR(6) runs are reused automatically")


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.torch_num_threads)
    try:
        torch.set_num_interop_threads(args.torch_num_interop_threads)
    except RuntimeError as error:
        raise RuntimeError("Could not set PyTorch inter-op threads before execution") from error
    plan = make_plan(args)
    device = resolve_device(args.device)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_plan(args, plan, device)
    if args.dry_run:
        return

    update_manifest(args, plan, device)
    build_index(args, plan)

    deterministic = not args.allow_nondeterministic
    run_counter = 0
    failures = 0

    for order in plan.orders:
        for train_seed in plan.train_seeds:
            if args.max_runs is not None and run_counter >= args.max_runs:
                build_index(args, plan)
                print(f"Stopped after --max-runs {args.max_runs}")
                return
            run_counter += 1

            run_dir = args.output_dir / f"ar{order}" / f"train_seed_{train_seed:010d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            logger = RunLogger(run_dir / "run.log")
            training_path = run_dir / "training_result.json"
            checkpoint_path = run_dir / "best_model.pt"
            failure_path = run_dir / "failure.json"

            training_config = make_training_config(
                args,
                order=order,
                train_seed=train_seed,
                steps=plan.steps_by_order[order],
                device=device,
            )
            training_signature = stable_signature(training_config)

            try:
                reusable_training = validate_existing_signature(
                    training_path,
                    key="training_signature",
                    expected=training_signature,
                    force=args.force,
                )
                if reusable_training and not checkpoint_path.exists():
                    raise RuntimeError(f"Training JSON exists but checkpoint is missing: {checkpoint_path}")

                if args.eval_only:
                    if not checkpoint_path.exists():
                        raise FileNotFoundError(
                            f"--eval-only requested but checkpoint does not exist: {checkpoint_path}"
                        )
                    logger.log(f"Using existing checkpoint for AR({order}) train_seed={train_seed}")
                elif reusable_training:
                    logger.log(f"Skipping completed training AR({order}) train_seed={train_seed}")
                else:
                    checkpoint_path, _ = train_specialist(
                        config=training_config,
                        signature=training_signature,
                        device=device,
                        deterministic=deterministic,
                        run_dir=run_dir,
                        inference_batch_size=args.inference_batch_size,
                        save_last_too=args.save_last_too,
                        logger=logger,
                    )
                    if failure_path.exists():
                        failure_path.unlink()

                validate_checkpoint_signature(checkpoint_path, training_signature)

                if args.train_only:
                    build_index(args, plan)
                    continue

                evaluations_dir = run_dir / "evaluations"
                evaluations_dir.mkdir(parents=True, exist_ok=True)
                for eval_seed in plan.eval_seeds:
                    evaluation_config = make_evaluation_config(
                        args,
                        training_signature=training_signature,
                        order=order,
                        train_seed=train_seed,
                        eval_seed=eval_seed,
                        device=device,
                    )
                    evaluation_signature = stable_signature(evaluation_config)
                    result_path = evaluations_dir / f"eval_seed_{eval_seed:010d}.json"
                    reusable_eval = validate_existing_signature(
                        result_path,
                        key="evaluation_signature",
                        expected=evaluation_signature,
                        force=args.force,
                    )
                    if reusable_eval:
                        logger.log(
                            f"Skipping completed evaluation AR({order}) train_seed={train_seed} eval_seed={eval_seed}"
                        )
                        continue
                    evaluate_checkpoint(
                        checkpoint_path=checkpoint_path,
                        training_signature=training_signature,
                        evaluation_config=evaluation_config,
                        evaluation_signature=evaluation_signature,
                        device=device,
                        result_path=result_path,
                        logger=logger,
                    )
                if failure_path.exists():
                    failure_path.unlink()
                build_index(args, plan)

            except Exception as error:
                failures += 1
                logger.log(f"FAILED AR({order}) train_seed={train_seed}: {type(error).__name__}: {error}")
                write_failure(
                    failure_path,
                    {
                        "ar_order": order,
                        "train_seed": train_seed,
                        "training_signature": training_signature,
                    },
                    error,
                )
                build_index(args, plan)
                if args.fail_fast:
                    raise

    build_index(args, plan)
    if failures:
        raise SystemExit(f"Pipeline completed with {failures} failed training-seed run(s). See failure.json files.")
    print("Pipeline completed successfully.")
    print(f"Index: {args.output_dir / 'experiment_index.json'}")


if __name__ == "__main__":
    main()

"""Orchestrate train ± context-shift × AR orders × perturbation methods.

For each train mode (train_context_shift=0 and =N), AR order, and perturbation:
  1) Train a fresh sanity checkpoint (retrain; do not reuse old checkpoints)
  2) Run attention extraction on that checkpoint
  3) Run perturbation at eval context-shift 0 and 1
  4) Write one comparison plot with both blue and both orange curves

Layout under --output-root (default outputs/context_shift_suite_seed{seed}):

  trainshift{T}/
    ar{p}/
      sanity/                         # metrics, scatter, checkpoint
      attention/                      # lag attention JSON/CSV/PNGs
      perturbation/
        {permute|zero|noise}/
          evalshift0/                 # raw perturbation outputs
          evalshift1/
          compare_blue_orange_evalshift0_vs_evalshift1_seed{seed}.png

Default grid: T in {0,1}, p in 1..7, methods in {permute,zero,noise}
→ 14 trainings + 14 attention runs + 14 sanity scatters + 42 comparison plots.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SANITY_SCRIPT = REPO_ROOT / "scripts" / "experiment_arp_sanity.py"
PERTURB_SCRIPT = REPO_ROOT / "scripts" / "experiment_arp_perturbation.py"
ATTENTION_SCRIPT = REPO_ROOT / "scripts" / "experiment_arp_attention.py"

COLOR_BLUE_SHIFT0 = "#9ecae1"
COLOR_BLUE_SHIFT1 = "#08519c"
COLOR_ORANGE_SHIFT0 = "#fdae6b"
COLOR_ORANGE_SHIFT1 = "#a63603"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train±shift × AR-order × perturbation suite.")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument(
        "--train-context-shifts",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Train modes to run (0 = normal, 1 = withhold lag 1, etc.)",
    )
    parser.add_argument(
        "--eval-context-shifts",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Eval context-shifts to run and overlay on comparison plots",
    )
    parser.add_argument(
        "--perturbations",
        type=str,
        nargs="+",
        default=["permute", "zero", "noise"],
        choices=["permute", "zero", "noise"],
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--perturb-seed",
        type=int,
        default=None,
        help="Seed for perturbation/attention eval (default: seed + 116)",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Suite root directory (default: outputs/context_shift_suite_seed{seed})",
    )
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing checkpoints if present")
    parser.add_argument("--skip-perturb", action="store_true", help="Reuse existing perturbation JSONs if present")
    parser.add_argument("--skip-attention", action="store_true", help="Reuse existing attention JSONs if present")
    parser.add_argument("--no-attention", action="store_true", help="Skip attention extraction entirely")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute")
    return parser.parse_args()


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root
    return REPO_ROOT / "outputs" / f"context_shift_suite_seed{args.seed}"


def run_name(order: int, train_shift: int, seed: int) -> str:
    if train_shift == 0:
        return f"ar{order}_scalar_rope_seed{seed}"
    return f"ar{order}_scalar_rope_seed{seed}_trainshift{train_shift}"


def train_dir(root: Path, train_shift: int, order: int) -> Path:
    return root / f"trainshift{train_shift}" / f"ar{order}" / "sanity"


def attention_dir(root: Path, train_shift: int, order: int) -> Path:
    return root / f"trainshift{train_shift}" / f"ar{order}" / "attention"


def attention_json_path(root: Path, train_shift: int, order: int, seed: int) -> Path:
    return attention_dir(root, train_shift, order) / f"{run_name(order, train_shift, seed)}_attention.json"


def perturb_dir(root: Path, train_shift: int, order: int, method: str, eval_shift: int) -> Path:
    return root / f"trainshift{train_shift}" / f"ar{order}" / "perturbation" / method / f"evalshift{eval_shift}"


def compare_plot_path(root: Path, train_shift: int, order: int, method: str, eval_shifts: list[int], seed: int) -> Path:
    tag = "_vs_".join(f"evalshift{e}" for e in sorted(eval_shifts))
    return (
        root
        / f"trainshift{train_shift}"
        / f"ar{order}"
        / "perturbation"
        / method
        / f"compare_blue_orange_{tag}_seed{seed}.png"
    )


def checkpoint_path(root: Path, train_shift: int, order: int, seed: int) -> Path:
    return train_dir(root, train_shift, order) / f"{run_name(order, train_shift, seed)}_best_model.pt"


def perturb_json_path(root: Path, train_shift: int, order: int, method: str, eval_shift: int, seed: int) -> Path:
    name = run_name(order, train_shift, seed)
    stem = f"{name}_perturbation_{method}"
    if eval_shift > 0:
        stem = f"{stem}_shift{eval_shift}"
    return perturb_dir(root, train_shift, order, method, eval_shift) / f"{stem}.json"


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def train_one(args: argparse.Namespace, order: int, train_shift: int) -> Path:
    out = train_dir(args.output_root, train_shift, order)
    ckpt = checkpoint_path(args.output_root, train_shift, order, args.seed)
    if args.skip_train and ckpt.is_file():
        print(f"SKIP train (exists): {ckpt}")
        return ckpt

    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        str(SANITY_SCRIPT),
        "--ar-order",
        str(order),
        "--steps",
        str(args.steps),
        "--use-rope",
        "--train-context-shift",
        str(train_shift),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--output-dir",
        str(out),
        "--run-name",
        run_name(order, train_shift, args.seed),
    ]
    run_cmd(cmd, dry_run=args.dry_run)
    return ckpt


def attention_one(
    args: argparse.Namespace,
    *,
    order: int,
    train_shift: int,
    ckpt: Path,
) -> Path:
    out = attention_dir(args.output_root, train_shift, order)
    json_path = attention_json_path(args.output_root, train_shift, order, args.seed)
    if args.skip_attention and json_path.is_file():
        print(f"SKIP attention (exists): {json_path}")
        return json_path

    out.mkdir(parents=True, exist_ok=True)
    name = run_name(order, train_shift, args.seed)
    cmd = [
        "uv",
        "run",
        "python",
        str(ATTENTION_SCRIPT),
        "--checkpoint",
        str(ckpt),
        "--eval-n",
        str(args.eval_n),
        "--seed",
        str(args.perturb_seed),
        "--device",
        args.device,
        "--output-dir",
        str(out),
        "--run-name",
        f"{name}_attention",
    ]
    run_cmd(cmd, dry_run=args.dry_run)
    return json_path


def perturb_one(
    args: argparse.Namespace,
    *,
    order: int,
    train_shift: int,
    method: str,
    eval_shift: int,
    ckpt: Path,
) -> Path:
    out = perturb_dir(args.output_root, train_shift, order, method, eval_shift)
    json_path = perturb_json_path(args.output_root, train_shift, order, method, eval_shift, args.seed)
    if args.skip_perturb and json_path.is_file():
        print(f"SKIP perturb (exists): {json_path}")
        return json_path

    out.mkdir(parents=True, exist_ok=True)
    name = run_name(order, train_shift, args.seed)
    cmd = [
        "uv",
        "run",
        "python",
        str(PERTURB_SCRIPT),
        "--checkpoint",
        str(ckpt),
        "--eval-n",
        str(args.eval_n),
        "--seed",
        str(args.perturb_seed),
        "--perturbation",
        method,
        "--context-shift",
        str(eval_shift),
        "--device",
        args.device,
        "--output-dir",
        str(out),
        "--run-name",
        f"{name}_perturbation",
    ]
    run_cmd(cmd, dry_run=args.dry_run)
    return json_path


def load_rows(path: Path) -> tuple[list[int], list[float], list[float], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    lags = [int(r["lag"]) for r in rows]
    blue = [float(r["mean_abs_delta"]) for r in rows]
    # Raw MSE after perturb (vs full-info CM), not the increase over base.
    orange = [float(r["mse_to_conditional_mean_after_perturb"]) for r in rows]
    base_mse = float(data["base_median_mse_to_conditional_mean"])
    return lags, blue, orange, base_mse


def save_comparison_plot(
    *,
    path: Path,
    order: int,
    train_shift: int,
    method: str,
    seed: int,
    json_by_eval_shift: dict[int, Path],
) -> None:
    shifts = sorted(json_by_eval_shift.keys())
    if not shifts:
        raise ValueError("No perturbation JSONs to plot")

    fig, (ax_blue, ax_orange) = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

    blue_colors = {0: COLOR_BLUE_SHIFT0, 1: COLOR_BLUE_SHIFT1}
    orange_colors = {0: COLOR_ORANGE_SHIFT0, 1: COLOR_ORANGE_SHIFT1}
    extra_blues = ["#6baed6", "#3182bd", "#08519c", "#08306b"]
    extra_oranges = ["#fdae6b", "#fd8d3c", "#e6550d", "#a63603"]

    lags_ref: list[int] | None = None
    for i, es in enumerate(shifts):
        lags, blue, orange, base_mse = load_rows(json_by_eval_shift[es])
        lags_ref = lags
        bcol = blue_colors.get(es, extra_blues[i % len(extra_blues)])
        ocol = orange_colors.get(es, extra_oranges[i % len(extra_oranges)])
        ax_blue.plot(lags, blue, marker="o", color=bcol, label=f"mean |Δ q0.5| (eval shift={es})")
        ax_orange.plot(lags, orange, marker="o", color=ocol, label=f"MSE after perturb (eval shift={es})")
        ax_orange.axhline(
            base_mse,
            linestyle=":",
            linewidth=1.2,
            color=ocol,
            alpha=0.85,
            label=f"base MSE (eval shift={es})",
        )

    if order > 1:
        ax_blue.axvline(order, linestyle="--", linewidth=1.0, color="gray", label=f"AR order p={order}")
        ax_orange.axvline(order, linestyle="--", linewidth=1.0, color="gray", label=f"AR order p={order}")

    ax_blue.set_ylabel("mean |Δ q0.5|")
    ax_blue.set_title(
        f"AR({order}) | train_context_shift={train_shift} | perturbation={method} | seed={seed}\n"
        "Blue: forecast sensitivity | Orange: raw MSE vs full-info CM (dotted = unperturbed base)"
    )
    ax_blue.legend(loc="best", fontsize=8)
    ax_blue.grid(True, alpha=0.25)

    ax_orange.set_xlabel("Model lag (1 = latest token the model sees)")
    ax_orange.set_ylabel("MSE to conditional mean")
    ax_orange.legend(loc="best", fontsize=8)
    ax_orange.grid(True, alpha=0.25)
    if lags_ref is not None:
        ax_orange.set_xticks(lags_ref)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def write_manifest(args: argparse.Namespace) -> None:
    n_train = len(args.orders) * len(args.train_context_shifts)
    manifest = {
        "orders": args.orders,
        "train_context_shifts": args.train_context_shifts,
        "eval_context_shifts": args.eval_context_shifts,
        "perturbations": args.perturbations,
        "steps": args.steps,
        "eval_n": args.eval_n,
        "seed": args.seed,
        "perturb_seed": args.perturb_seed,
        "device": args.device,
        "output_root": str(args.output_root),
        "run_attention": not args.no_attention,
        "n_trainings": n_train,
        "n_attention_runs": 0 if args.no_attention else n_train,
        "n_comparison_plots": n_train * len(args.perturbations),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "suite_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    if any(p < 1 for p in args.orders):
        raise ValueError("--orders must be positive integers")
    if any(t < 0 for t in args.train_context_shifts):
        raise ValueError("--train-context-shifts must be >= 0")
    if any(e < 0 for e in args.eval_context_shifts):
        raise ValueError("--eval-context-shifts must be >= 0")

    if args.perturb_seed is None:
        args.perturb_seed = int(args.seed) + 116
    args.output_root = resolve_output_root(args)

    write_manifest(args)

    n_train = len(args.orders) * len(args.train_context_shifts)
    n_plots = n_train * len(args.perturbations)
    n_attn = 0 if args.no_attention else n_train
    print(
        f"Suite plan: seed={args.seed} | root={args.output_root} | "
        f"{n_train} trainings + sanity scatters, "
        f"{n_attn} attention runs, "
        f"{n_train * len(args.perturbations) * len(args.eval_context_shifts)} perturbation runs, "
        f"{n_plots} comparison plots"
    )

    for train_shift in args.train_context_shifts:
        for order in args.orders:
            print(f"\n===== seed={args.seed} | train_context_shift={train_shift} | AR({order}) =====")
            ckpt = train_one(args, order, train_shift)

            if not args.no_attention:
                if args.dry_run:
                    print(
                        "DRY-RUN would run attention → "
                        f"{attention_json_path(args.output_root, train_shift, order, args.seed)}"
                    )
                else:
                    if not args.skip_train and not ckpt.is_file():
                        raise FileNotFoundError(f"Missing checkpoint after train: {ckpt}")
                    attention_one(args, order=order, train_shift=train_shift, ckpt=ckpt)

            for method in args.perturbations:
                json_by_eval: dict[int, Path] = {}
                for eval_shift in args.eval_context_shifts:
                    if args.dry_run:
                        json_by_eval[eval_shift] = perturb_json_path(
                            args.output_root, train_shift, order, method, eval_shift, args.seed
                        )
                        print(
                            f"DRY-RUN would perturb method={method} eval_shift={eval_shift} "
                            f"→ {json_by_eval[eval_shift]}"
                        )
                        continue
                    if not args.skip_train and not ckpt.is_file():
                        raise FileNotFoundError(f"Missing checkpoint after train: {ckpt}")
                    json_by_eval[eval_shift] = perturb_one(
                        args,
                        order=order,
                        train_shift=train_shift,
                        method=method,
                        eval_shift=eval_shift,
                        ckpt=ckpt,
                    )

                plot_path = compare_plot_path(
                    args.output_root,
                    train_shift,
                    order,
                    method,
                    args.eval_context_shifts,
                    args.seed,
                )
                if args.dry_run:
                    print(f"DRY-RUN would write compare plot → {plot_path}")
                    continue
                save_comparison_plot(
                    path=plot_path,
                    order=order,
                    train_shift=train_shift,
                    method=method,
                    seed=args.seed,
                    json_by_eval_shift=json_by_eval,
                )

    print("\nSuite finished.")
    print(f"Root: {args.output_root}")
    print("Sanity scatters: trainshift*/ar*/sanity/*_best_scatter.png")
    print("Attention: trainshift*/ar*/attention/*_attention*.png")
    print("Comparison plots: trainshift*/ar*/perturbation/*/compare_blue_orange_*_seed*.png")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)

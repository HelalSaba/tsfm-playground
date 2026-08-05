"""Task 4a: cross-order specialist generalization.

Evaluates fixed-order checkpoints on other AR orders using:
- MASE and sMAPE for the q0.5 point forecast,
- WQL for the full probabilistic forecast,
- MSE/zero only as a retained diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from arp_task4_common import (
    as_namespace,
    build_model,
    forecast_metrics,
    generate_fixed_order_data,
    resolve_device,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 4a cross-order generalization.")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 7])
    parser.add_argument("--test-orders", type=int, nargs="+", default=[1, 7])
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        default=r"outputs\ar{p}\scalar\sanity\ar{p}_scalar_rope_seed7_best_model.pt",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Explicit label=path checkpoint. May be passed multiple times.",
    )
    parser.add_argument("--exclude-diagonal", action="store_true")
    parser.add_argument("--eval-n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seasonality", type=int, default=1)
    parser.add_argument("--graceful-factor", type=float, default=1.25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_order/revised"))
    parser.add_argument("--run-name", default="task4a_cross_order")
    return parser.parse_args()


def parse_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(spec)
    return path.stem, path


def checkpoint_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.checkpoint:
        return [parse_spec(x) for x in args.checkpoint]
    return [(f"ar{p}", Path(args.checkpoint_template.format(p=p))) for p in args.orders]


def classify(rows: list[dict[str, Any]], graceful_factor: float) -> None:
    matched = {
        int(r["test_ar_order"]): float(r["median_mase_to_noisy_next_x"])
        for r in rows
        if int(r["train_ar_order"]) == int(r["test_ar_order"])
    }
    for row in rows:
        train_p = int(row["train_ar_order"])
        test_p = int(row["test_ar_order"])
        mase = float(row["median_mase_to_noisy_next_x"])
        matched_mase = matched.get(test_p)
        ratio = mase / matched_mase if matched_mase and matched_mase > 0 else None
        row["mase_over_matched_specialist"] = ratio

        if train_p == test_p:
            mode = "in_distribution"
        elif mase >= 1.0:
            mode = "catastrophic"
        elif ratio is not None and ratio <= graceful_factor:
            mode = "good_transfer"
        else:
            mode = "graceful_degradation"
        row["failure_mode"] = mode


def matrix(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[int], np.ndarray]:
    train = sorted({int(r["train_ar_order"]) for r in rows})
    test = sorted({int(r["test_ar_order"]) for r in rows})
    values = np.full((len(train), len(test)), np.nan)
    ti = {p: i for i, p in enumerate(train)}
    tj = {p: j for j, p in enumerate(test)}
    for row in rows:
        value = row.get(key)
        if value is not None:
            values[ti[int(row["train_ar_order"])], tj[int(row["test_ar_order"])]] = float(value)
    return train, test, values


def heatmap(rows: list[dict[str, Any]], key: str, title: str, path: Path, fmt: str = ".3f") -> None:
    train, test, values = matrix(rows, key)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(test)), [f"AR({p})" for p in test])
    ax.set_yticks(range(len(train)), [f"AR({p})" for p in train])
    ax.set_xlabel("Test order")
    ax.set_ylabel("Train order")
    ax.set_title(title)
    for i in range(len(train)):
        for j in range(len(test)):
            if np.isfinite(values[i, j]):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center")
            else:
                ax.text(j, i, "—", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def failure_grid(rows: list[dict[str, Any]], path: Path) -> None:
    labels = {
        "in_distribution": 0,
        "good_transfer": 1,
        "graceful_degradation": 2,
        "catastrophic": 3,
    }
    train = sorted({int(r["train_ar_order"]) for r in rows})
    test = sorted({int(r["test_ar_order"]) for r in rows})
    values = np.full((len(train), len(test)), np.nan)
    text = [["—" for _ in test] for _ in train]
    ti = {p: i for i, p in enumerate(train)}
    tj = {p: j for j, p in enumerate(test)}
    short = {
        "in_distribution": "ID",
        "good_transfer": "GOOD",
        "graceful_degradation": "GRACE",
        "catastrophic": "CAT",
    }
    for row in rows:
        i, j = ti[int(row["train_ar_order"])], tj[int(row["test_ar_order"])]
        values[i, j] = labels[row["failure_mode"]]
        text[i][j] = short[row["failure_mode"]]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap="tab10", vmin=0, vmax=9)
    ax.set_xticks(range(len(test)), [f"AR({p})" for p in test])
    ax.set_yticks(range(len(train)), [f"AR({p})" for p in train])
    ax.set_xlabel("Test order")
    ax.set_ylabel("Train order")
    ax.set_title("Task 4a failure-mode classification")
    for i in range(len(train)):
        for j in range(len(test)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    rows: list[dict[str, Any]] = []

    for label, checkpoint_path in checkpoint_specs(args):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for {label}: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        train_args = as_namespace(checkpoint["args"])
        train_p = int(train_args.ar_order)
        model = build_model(train_args).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        for test_p in args.test_orders:
            if args.exclude_diagonal and train_p == test_p:
                continue
            eval_seed = args.seed + 100_000 * int(test_p)
            batch, context, true_horizon, conditional_mean = generate_fixed_order_data(
                train_args=train_args,
                test_order=int(test_p),
                eval_n=args.eval_n,
                seed=eval_seed,
                device=device,
            )
            metrics, _ = forecast_metrics(
                model=model,
                batch=batch,
                context=context,
                true_horizon=true_horizon,
                conditional_mean=conditional_mean,
                test_order=int(test_p),
                seasonality=args.seasonality,
            )
            rows.append(
                {
                    "checkpoint_label": label,
                    "checkpoint": str(checkpoint_path),
                    "train_ar_order": train_p,
                    "test_ar_order": int(test_p),
                    "is_diagonal": int(train_p == int(test_p)),
                    "patch_size": train_args.patch_size,
                    "patch_stride": train_args.patch_stride,
                    **metrics,
                }
            )

    if not rows:
        raise RuntimeError("No evaluations were run.")

    classify(rows, args.graceful_factor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.run_name}.json"
    csv_path = args.output_dir / f"{args.run_name}.csv"
    json_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(csv_path, rows)

    heatmap(
        rows,
        "median_mase_to_noisy_next_x",
        "Task 4a: MASE by train/test order",
        args.output_dir / f"{args.run_name}_mase_heatmap.png",
    )
    heatmap(
        rows,
        "mean_wql_to_noisy_next_x",
        "Task 4a: mean WQL by train/test order",
        args.output_dir / f"{args.run_name}_wql_heatmap.png",
    )
    heatmap(
        rows,
        "mase_over_matched_specialist",
        "Task 4a: transfer MASE / matched-specialist MASE",
        args.output_dir / f"{args.run_name}_mase_over_matched_heatmap.png",
    )
    heatmap(
        rows,
        "model_mse_over_zero_mse",
        "Retained diagnostic: conditional-mean MSE / zero MSE",
        args.output_dir / f"{args.run_name}_mse_over_zero_heatmap.png",
    )
    failure_grid(rows, args.output_dir / f"{args.run_name}_failure_mode_grid.png")

    print("\nTask 4a cross-order summary")
    for row in rows:
        ratio = row["mase_over_matched_specialist"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.3f}"
        print(
            f"AR({row['train_ar_order']}) -> AR({row['test_ar_order']}) | "
            f"MASE={row['median_mase_to_noisy_next_x']:.3f} | "
            f"sMAPE={row['median_smape_percent_to_noisy_next_x']:.1f}% | "
            f"WQL={row['mean_wql_to_noisy_next_x']:.3f} | "
            f"MASE/matched={ratio_text} | {row['failure_mode']}"
        )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

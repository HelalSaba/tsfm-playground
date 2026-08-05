# TSFM Playground

A modular Python environment for developing and ablating components of
**time-series foundation models (TSFMs)**, with a focus on interpreting
transformers trained on synthetic **autoregressive (AR)** processes.

Synthetic AR(\(p\)) series give a known lag support (the first \(p\) lags).
That ground truth lets us ask:

1. Which lags is the model most dependent on / sensitive to?
2. Do attention and perturbation agree, and is disagreement due to seed or context position?
3. Is forecast performance sustained under cross-order transfer and mixed-order training?

---

## Requirements

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/) (recommended) **or** `pip`

Core dependencies are listed in `requirements.txt` and `pyproject.toml`:

| Package    | Role                          |
|------------|-------------------------------|
| `torch`    | training and evaluation       |
| `numpy`    | numerics                      |
| `einops`   | tensor reshaping in UVModel   |
| `matplotlib` | plots                       |
| `jupyter`  | notebooks / reports           |

---

## Setup

### Option A — uv (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if needed
uv venv
uv sync
# optional, for local development hooks:
uv run pre-commit install
```

### Option B — pip + requirements.txt

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # installs the local `playground` package
```

GPU note: if you need a CUDA build of PyTorch, install the wheel from the
[official PyTorch index](https://pytorch.org/get-started/locally/) first, then
install the remaining requirements.

---

## Repository layout

```
src/playground/          # model, synthetic AR data, loss
scripts/                 # experiment entry points
outputs/                 # default run artifacts (created when you train)
notebooks/               # exploratory notebooks
reports/                 # writeups / static reports
```

There is no single `train.py` / `evaluate.py`. Training and analysis live in
`scripts/`. The local package is importable as `playground` after `uv sync`
or `pip install -e .`. Scripts also add `src/` to `sys.path` so they can run
from a fresh checkout.

---

## The three main experiments

These three suites cover the interpretability questions above. Prefer
`uv run python ...`; with a activated venv, plain `python ...` works too.
On Apple Silicon, `--device auto` selects MPS when CUDA is unavailable.

### 1) Seed robustness (attention vs perturbation across training seeds)

Trains independent scalar specialists, then evaluates attention and lag
perturbation under several fixed evaluation seeds.

| Stage   | Orders        | Training seeds | Notes |
|---------|---------------|----------------|-------|
| `pilot` | AR(6)         | 5              | quick check |
| `main`  | AR(2), AR(6), AR(7) | 6 each | reuses matching AR(6) pilot runs |

```bash
# Pilot (AR(6) only)
uv run python scripts/run_seed_robustness.py --stage pilot --device auto

# Full study (AR(2), AR(6), AR(7))
uv run python scripts/run_seed_robustness.py --stage main --device auto
```

Useful flags:

```bash
# Dry-run the plan without training
uv run python scripts/run_seed_robustness.py --stage main --dry-run

# Train or evaluate only
uv run python scripts/run_seed_robustness.py --stage main --train-only --device auto
uv run python scripts/run_seed_robustness.py --stage main --eval-only --device auto

# Custom orders / seed count
uv run python scripts/run_seed_robustness.py --stage custom --orders 2 6 7 \
  --num-train-seeds 6 --device auto
```

Default output directory: `outputs/seed_robustness/`

```
outputs/seed_robustness/
  experiment_manifest.json
  experiment_index.json
  ar{2,6,7}/
    train_seed_<SEED>/
      best_model.pt
      training_result.json
      evaluations/eval_seed_<SEED>.json
```

Each evaluation JSON includes attention profiles, perturbation profiles, and
forecast metrics — ready for offline aggregation without retraining.

---

### 2) Context-shift suite (slot vs content)

For each AR order and train-context-shift mode, trains a fresh checkpoint,
extracts attention, runs perturbations at eval shifts 0 and 1, and writes
comparison plots (forecast sensitivity vs MSE to the conditional mean).

```bash
# Full default grid: orders 1–7, train shifts {0,1}, perturbations permute/zero/noise
uv run python scripts/run_context_shift_suite.py --seed 13 --device auto

# Smaller debug grid
uv run python scripts/run_context_shift_suite.py \
  --orders 6 7 \
  --train-context-shifts 0 \
  --perturbations permute \
  --seed 13 \
  --device auto

# Reuse existing JSONs and only rewrite comparison plots
uv run python scripts/run_context_shift_suite.py \
  --seed 13 --skip-train --skip-perturb --device auto
```

Default output root: `outputs/context_shift_suite_seed{seed}/`

```
trainshift{0|1}/ar{p}/
  sanity/          # checkpoint + scatter
  attention/
  perturbation/{permute|zero|noise}/
    evalshift0/ evalshift1/
    compare_blue_orange_evalshift0_vs_evalshift1_seed{seed}.png
```

---

### 3) Cross-order transfer and mixed-order training

**A. Train scalar specialists** (one model per AR order), then measure
attention and perturbation:

```bash
# Train AR(1)–AR(7) specialists
for P in 1 2 3 4 5 6 7; do
  uv run python scripts/experiment_arp_sanity.py \
    --ar-order $P --steps 5000 --use-rope --device auto \
    --output-dir outputs/ar$P/scalar/sanity \
    --run-name ar${P}_scalar_rope_seed7
done

# Perturbation (permute) and attention on each checkpoint
for P in 1 2 3 4 5 6 7; do
  CKPT=outputs/ar$P/scalar/sanity/ar${P}_scalar_rope_seed7_best_model.pt
  uv run python scripts/experiment_arp_perturbation.py \
    --checkpoint $CKPT --eval-n 4000 --perturbation permute --device auto \
    --output-dir outputs/ar$P/scalar/perturbation \
    --run-name ar${P}_scalar_rope_seed7_perturbation
  uv run python scripts/experiment_arp_attention.py \
    --checkpoint $CKPT --eval-n 4000 --device auto \
    --output-dir outputs/ar$P/scalar/attention \
    --run-name ar${P}_scalar_rope_seed7_attention
done
```

**B. Cross-order specialist transfer** (every train order × every test order):

```bash
uv run python scripts/experiment_arp_cross_eval_all.py \
  --orders 1 2 3 4 5 6 7 --test-orders 1 2 3 4 5 6 7 \
  --eval-n 4000 --device auto \
  --output-dir outputs/cross_order/scalar/revised_all49 \
  --run-name task4a_scalar_all49
```

**C. Mixed-order training** (train on \(p \in \{1,\ldots,5\}\), test on 1–7):

```bash
uv run python scripts/experiment_arp_sanity.py \
  --ar-order mixed --min-ar-order 1 --max-ar-order 5 \
  --steps 12000 --use-rope --device auto \
  --output-dir outputs/mixed_1_5/scalar/sanity \
  --run-name mixed_ar1_5_scalar_rope_seed7

uv run python scripts/experiment_arp_mixed_icl.py \
  --checkpoint outputs/mixed_1_5/scalar/sanity/mixed_ar1_5_scalar_rope_seed7_best_model.pt \
  --test-orders 1 2 3 4 5 6 7 --eval-n 4000 --device auto \
  --output-dir outputs/mixed_1_5/scalar/task4b_revised \
  --run-name mixed_ar1_5_scalar_task4b
```

A longer command checklist lives in `scripts/scalar_experiment_commands.txt`.

---

## Quick smoke test

Train a short AR(1) sanity run to verify the install:

```bash
uv run python scripts/experiment_arp_sanity.py \
  --ar-order 1 --steps 500 --use-rope --device auto \
  --output-dir outputs/smoke/ar1 \
  --run-name ar1_smoke
```


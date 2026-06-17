## Development instructions

Read this file before making changes in the repository.

### Project overview

- Python version: 3.11+
- Package manager / environment tool: uv
- Code quality tools: ruff and pre-commit
- Utility scripts: scripts/training.py and scripts/evaluate.py

### 1) Set up the environment

If uv is not installed, install it first:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create and sync the virtual environment:

```sh
uv venv
uv sync
uv run pre-commit install
```

### 2) Run the project

The scripts directory currently contains scripts for training and evaluation:

```sh
uv run python scripts/training.py
uv run python scripts/evaluate.py
```

### 3) Formatting and linting

Run the same checks used by the repository hooks:

```sh
uv run ruff check .
uv run ruff format .
```

Pre-commit hooks are configured to run Ruff automatically before commits.


### 5) Before finishing a change

- Run formatting and linting.
- Verify the script you changed still runs.
- Make sure new files follow the existing repository layout.

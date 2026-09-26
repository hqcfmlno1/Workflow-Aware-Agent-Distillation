# Python Environment Design

## Objective

Create a reproducible Python environment for the Workflow-Aware Agent
Distillation project. The first phase calls teacher and student models through
OpenAI-compatible APIs. Later phases run SFT and LoRA training on a Linux GPU
instance rented from Vast.ai.

## Constraints

- Use `uv` for Python installation, dependency resolution, virtual environments,
  and lock files.
- Use Python 3.11 for broad compatibility with ML and agent tooling.
- Keep the local Windows environment lightweight and CUDA-independent.
- Never commit API keys, model credentials, generated datasets, checkpoints, or
  experiment logs.
- Make the environment reproducible on both the local machine and a Vast.ai
  Linux instance.

## Project Layout

```text
.
|-- .env.example
|-- .gitignore
|-- .python-version
|-- pyproject.toml
|-- uv.lock
|-- notebooks/
|-- src/workflow_distillation/
`-- tests/
```

The repository uses a `src` layout so local imports behave the same in
notebooks, tests, scripts, and remote training jobs.

## Dependency Groups

### Core

The default environment supports API calls, experiment configuration,
trajectory collection, dataset serialization, analysis, and command-line
utilities:

- `openai`, `httpx`, and `tenacity` for resilient OpenAI-compatible API calls.
- `pydantic-settings` and `python-dotenv` for typed configuration loaded from
  environment variables.
- `datasets`, `pandas`, `pyarrow`, `jsonlines`, and `orjson` for trajectory and
  dataset handling.
- `numpy`, `scipy`, and `scikit-learn` for behavioral analysis.
- `tqdm`, `rich`, and `typer` for experiment scripts and command-line tools.
- `wandb` for optional experiment tracking.

### Notebook Extra

The `notebook` extra contains `jupyterlab`, `ipykernel`, `ipywidgets`,
`matplotlib`, and `seaborn`. It is installed locally for exploration and
analysis but is not required by headless experiment workers.

### Training Extra

The `train` extra contains `torch`, `transformers`, `accelerate`, `peft`, `trl`,
`bitsandbytes`, `sentencepiece`, and `safetensors`. It is declared in the lock
file but not installed in the local Windows environment. It is installed on a
compatible Vast.ai Linux image.

CUDA-sensitive optimizers and kernels such as DeepSpeed, FlashAttention, and
Unsloth are intentionally deferred until the GPU type, CUDA version, and base
container image are selected.

### Development Group

The `dev` dependency group contains `pytest`, `pytest-asyncio`, and `ruff`.

## SWE-Agent and OmniCode

SWE-agent and OmniCode are not added as unconstrained PyPI dependencies. Their
repositories and revisions will be pinned when the experiment harness is
selected. If their dependency constraints conflict with the training stack,
they will use a separate `uv` project or container while exchanging trajectories
through versioned JSONL or Parquet files.

## Local and Cloud Workflow

The Git repository remains the source of truth. A Vast.ai instance receives a
clone of the repository through a private Git remote or a copy through `rsync`
or `scp`. Development can continue through VS Code Remote SSH, while Python and
GPU processes execute on the server.

Local setup installs the default, notebook, and development dependencies. A
future Vast.ai setup installs the default and training dependencies. Large
datasets, model weights, checkpoints, and run artifacts remain outside Git and
are transferred through Hugging Face Hub, Weights & Biases artifacts, or object
storage as appropriate.

## Verification

The initial environment is accepted when:

1. `uv sync` creates the Python 3.11 environment successfully.
2. The project package imports successfully.
3. Jupyter can load the project kernel.
4. A smoke test can instantiate an OpenAI-compatible client without embedding
   a credential in source code.
5. `pytest` and `ruff check` pass.


# Python Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and verify a reproducible API-first Python 3.11 environment with optional notebook and GPU-training dependencies.

**Architecture:** A single `uv` project provides the lightweight default environment used locally. Notebook and Linux/CUDA training packages are isolated as optional extras, while tests verify package imports and API configuration without making network calls.

**Tech Stack:** Python 3.11, uv, OpenAI SDK, Hugging Face Datasets, JupyterLab, pytest, Ruff, optional Transformers/PEFT/TRL training stack

**Spec:** `docs/superpowers/specs/2026-09-26-python-environment-design.md`

## Global Constraints

- Use `uv` for Python installation, dependency resolution, virtual environments, and lock files.
- Use Python 3.11.
- Do not install CUDA-sensitive training dependencies in the local Windows environment.
- Do not commit API keys, generated datasets, model weights, checkpoints, or experiment logs.
- Keep the project reproducible on Windows locally and Linux on Vast.ai.

## Review Focus

- A missing `.env` must not prevent importing the package or running tests.
- An API key must never appear in committed source or the example environment file.
- The default local sync must not install `torch`, `bitsandbytes`, or other CUDA-sensitive training packages.
- The package must import through the `src` layout without manual `PYTHONPATH` changes.
- Notebook and training dependency sets must remain independently selectable.

---

### Task 1: Create the uv project and dependency groups

**Files:**
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/workflow_distillation/__init__.py`
- Create: `notebooks/.gitkeep`
- Create: `tests/test_environment.py`

**Interfaces:**
- Consumes: The dependency and layout decisions in the approved spec.
- Produces: Importable package `workflow_distillation` and optional extras named `notebook` and `train`.

- [ ] **Step 1: Write the environment smoke tests**

Create tests that assert the package imports, required API/data libraries are importable, `.env.example` contains placeholder variable names but no secret value, and CUDA-sensitive packages are absent from the default dependency list.

- [ ] **Step 2: Run the tests to verify the uncreated package fails**

Run: `uv run pytest tests/test_environment.py -v`

Expected: FAIL because project metadata and package files do not exist yet.

- [ ] **Step 3: Create project metadata and scaffolding**

Set `requires-python = ">=3.11,<3.12"`, declare the core dependencies, add `notebook` and `train` optional extras, and add `pytest`, `pytest-asyncio`, and `ruff` to the `dev` dependency group. Configure Ruff and pytest in `pyproject.toml`.

- [ ] **Step 4: Generate the lock file and sync the local environment**

Run: `uv python install 3.11`

Run: `uv sync --extra notebook --group dev`

Expected: `.venv` and `uv.lock` are created without installing the `train` extra.

- [ ] **Step 5: Run the smoke tests and lint checks**

Run: `uv run pytest -v`

Run: `uv run ruff check .`

Expected: Both commands pass.

### Task 2: Verify notebook and API client readiness

**Files:**
- Modify: `tests/test_environment.py`

**Interfaces:**
- Consumes: The environment and package created in Task 1.
- Produces: Verification that Jupyter and the OpenAI-compatible client can initialize without a real credential or network request.

- [ ] **Step 1: Add notebook and client initialization tests**

Add tests that import `jupyterlab` and instantiate `openai.OpenAI(api_key="test-key", base_url="http://localhost:20128/v1")` without sending a request.

- [ ] **Step 2: Run the focused tests**

Run: `uv run pytest tests/test_environment.py -v`

Expected: PASS.

- [ ] **Step 3: Verify the resolved environment**

Run: `uv tree --depth 1`

Expected: Core, notebook, and development packages are present; training-only packages are not installed.

- [ ] **Step 4: Remove the requested design Markdown after successful verification**

Delete `docs/superpowers/specs/2026-09-26-python-environment-design.md`. Keep the deletion recoverable in Git history.

- [ ] **Step 5: Run final verification**

Run: `git diff --check`

Run: `uv run pytest -v`

Run: `uv run ruff check .`

Expected: All commands pass and the design spec is absent from the working tree.


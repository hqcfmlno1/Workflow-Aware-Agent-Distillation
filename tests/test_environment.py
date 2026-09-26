from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_package_and_core_dependencies_are_importable() -> None:
    modules = [
        "workflow_distillation",
        "openai",
        "httpx",
        "datasets",
        "pandas",
        "pyarrow",
        "numpy",
        "scipy",
        "sklearn",
        "pydantic_settings",
    ]

    for module in modules:
        assert importlib.import_module(module) is not None


def test_example_environment_contains_placeholders_only() -> None:
    contents = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY=" in contents
    assert "OPENAI_BASE_URL=http://localhost:20128/v1" in contents
    assert "sk-" not in contents


def test_training_dependencies_are_not_installed_by_default() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    default_dependencies = " ".join(project["project"]["dependencies"]).lower()
    training_dependencies = {"torch", "transformers", "bitsandbytes", "peft", "trl"}

    assert all(name not in default_dependencies for name in training_dependencies)


def test_jupyterlab_is_available() -> None:
    assert importlib.import_module("jupyterlab") is not None


def test_openai_compatible_client_can_be_initialized_without_network() -> None:
    from openai import OpenAI

    with OpenAI(
        api_key="test-key",
        base_url="http://localhost:20128/v1",
    ) as client:
        assert str(client.base_url) == "http://localhost:20128/v1/"

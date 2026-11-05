"""Shared fixtures: an isolated models dir wired into the backend registry."""
from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from app.core.config import settings
from app.integrations.ml import registry, router as ml_router

V1_SOURCE = settings.ml_models_dir / "dev_signal_ml1.json"


@pytest.fixture
def models_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "ml_models"
    d.mkdir()
    monkeypatch.setattr(registry, "settings", replace(settings, ml_models_dir=d))
    registry.notify_models_changed()
    ml_router._scorers.clear()
    yield d
    registry.notify_models_changed()


@pytest.fixture
def v1_model(models_dir) -> str:
    if not V1_SOURCE.is_file():
        pytest.skip(f"no v1 artifact at {V1_SOURCE}")
    shutil.copy(V1_SOURCE, models_dir / "dev_signal_ml1.json")
    registry.notify_models_changed()
    return "dev_signal_ml1"


@pytest.fixture
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(ml_router.router)  # no lifespan: the runner/queue aren't needed
    return TestClient(app)

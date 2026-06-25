"""Fixtures pytest partagées pour les tests Phase 3.0.

Ce conftest fournit :
- test_app  : application FastAPI minimale avec le router reasoning branché
              et l'authentification Supabase surchargée pour les tests.
- test_client : TestClient prêt à l'emploi sur test_app.
- authenticated_client : TestClient avec header Authorization Bearer valide.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.auth import require_supabase_user
from models.user import User
from routers.reasoning import router as reasoning_router


def _mock_user() -> User:
    """Utilisateur factice injecté à la place de require_supabase_user."""
    return User(id="test-user-123", email="test@makenbrain.local")


@pytest.fixture(scope="session")
def test_app() -> FastAPI:
    """Application FastAPI minimale pour les tests du router reasoning.

    - Ne charge pas main.py (évite ChromaDB, Supabase, Ollama).
    - Surcharge require_supabase_user avec un utilisateur factice.
    """
    app = FastAPI(title="MakenBrain Test — Phase 3.0")
    app.include_router(reasoning_router, prefix="/reasoning")
    app.dependency_overrides[require_supabase_user] = _mock_user
    return app


@pytest.fixture(scope="session")
def test_client(test_app: FastAPI) -> TestClient:
    """TestClient sans authentification (pour tester les 401)."""
    # dependency_overrides est retiré pour ce client : on test l'auth réelle.
    app_no_override = FastAPI(title="MakenBrain Test Auth")
    app_no_override.include_router(reasoning_router, prefix="/reasoning")
    return TestClient(app_no_override, raise_server_exceptions=False)


@pytest.fixture(scope="session")
def authenticated_client(test_app: FastAPI) -> TestClient:
    """TestClient avec auth surchargée (utilisateur factice injecté)."""
    return TestClient(test_app, raise_server_exceptions=True)

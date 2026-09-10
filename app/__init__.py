"""FSD spatial reasoning demo package.

Exposes the FastAPI ``app`` instance so ``uvicorn app:app`` resolves it
directly from the repo root.
"""
from .main import app

__all__ = ["app"]
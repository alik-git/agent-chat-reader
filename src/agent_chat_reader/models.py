"""Shared data models."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


class Turn(NamedTuple):
    """A single conversation turn."""

    role: str  # "USER" | "AGENT" | "CLAUDE"
    text: str
    timestamp: str


class SessionMeta(NamedTuple):
    """Metadata for a single chat session."""

    source: str  # "codex" | "claude"
    id: str
    path: Path
    mtime: float
    size_kb: int
    title: str

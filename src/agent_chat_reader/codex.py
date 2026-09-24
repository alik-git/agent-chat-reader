"""Codex CLI session reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from agent_chat_reader.models import SessionMeta, Turn

CODEX_SESSIONS = Path.home() / ".codex" / "sessions"


class SessionMetadata(NamedTuple):
    """Stable metadata read from the beginning of a Codex rollout."""

    title: str
    is_subagent: bool


def session_metadata(path: Path) -> SessionMetadata:
    """Read a rollout's title and subagent state in one pass."""
    title = ""
    is_subagent = False
    found_session_meta = False

    with path.open() as fh:
        for raw in fh:
            try:
                rec = json.loads(raw.strip())
                if rec.get("type") == "session_meta" and not found_session_meta:
                    payload = rec.get("payload", {})
                    source = payload.get("source", {})
                    is_subagent = payload.get("thread_source") == "subagent" or (
                        isinstance(source, dict) and "subagent" in source
                    )
                    found_session_meta = True
                elif not title:
                    turn = _event_turn(rec)
                    if turn is not None and turn.role == "USER":
                        title = turn.text.replace("\n", " ")[:80]
            except Exception:
                pass
            if found_session_meta and title:
                break

    return SessionMetadata(title=title, is_subagent=is_subagent)


def is_subagent(path: Path) -> bool:
    """Return True if this Codex session is a guardian/subagent session."""
    return session_metadata(path).is_subagent


def _session_id_from_path(path: Path) -> str:
    """Extract the UUID from a rollout filename."""
    name = path.stem.replace("rollout-", "")
    parts = name.split("-")
    # UUID is the last 5 hyphen-groups (8-4-4-4-12)
    return "-".join(parts[-5:]) if len(parts) >= 5 else name


def list_sessions(*, include_subagents: bool = False) -> list[SessionMeta]:
    """List all Codex sessions, sorted by most recent first."""
    if not CODEX_SESSIONS.exists():
        return []
    files = sorted(
        CODEX_SESSIONS.rglob("*.jsonl"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    results = []
    for f in files:
        metadata = session_metadata(f)
        if not include_subagents and metadata.is_subagent:
            continue
        stat = f.stat()
        results.append(
            SessionMeta(
                source="codex",
                id=_session_id_from_path(f),
                path=f,
                mtime=stat.st_mtime,
                size_kb=stat.st_size // 1024,
                title=metadata.title,
            )
        )
    return results


def read_turns(path: Path, *, tail: int | None = None) -> list[Turn]:
    """Read conversation turns from a Codex session file.

    Extracts user messages, agent commentary (event_msg/agent_message), and
    full assistant responses (response_item). Adjacent duplicate text between
    agent_message and response_item records is deduplicated.

    Args:
        path: Path to the session JSONL file.
        tail: If set, return only turns from the last N user messages onward.

    Returns:
        List of Turn namedtuples in conversation order.
    """
    turns, _offset = read_turns_from(path, offset=0)
    return _apply_tail(turns, tail)


def read_turns_from(
    path: Path,
    *,
    offset: int,
    last_assistant_text: str | None = None,
) -> tuple[list[Turn], int]:
    """Read complete Codex records after a safe byte offset."""
    turns: list[Turn] = []

    with path.open("rb") as fh:
        fh.seek(offset)
        for raw in fh:
            line_start = fh.tell() - len(raw)
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
                t = rec.get("type", "")
                ts: str = rec.get("timestamp", "")
                turn = _event_turn(rec)
                if turn is not None:
                    if turn.role == "USER":
                        turns.append(turn)
                        last_assistant_text = None
                    elif turn.text != last_assistant_text:
                        turns.append(turn)
                        last_assistant_text = turn.text
                elif t == "response_item":
                    payload = rec.get("payload", {})
                    if payload.get("role") == "assistant" and payload.get(
                        "channel"
                    ) in (None, "commentary", "final"):
                        text = _content_text(payload.get("content", []))
                        if text and text != last_assistant_text:
                            turns.append(Turn("AGENT", text, ts))
                            last_assistant_text = text
            except Exception:
                if not raw.endswith(b"\n"):
                    return turns, line_start

        return turns, fh.tell()


def _content_text(content: object) -> str:
    """Extract visible text blocks without images, reasoning or tool payloads."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") in {"text", "Text", "input_text", "output_text"}
        and isinstance(block.get("text"), str)
    ).strip()


def _event_turn(record: dict) -> Turn | None:
    """Read canonical user/agent events, including current completed items.

    User response items also contain injected context and replayed history, so
    only canonical user events count as real user turns.
    """
    if record.get("type") != "event_msg":
        return None
    payload = record.get("payload", {})
    kind = payload.get("type")
    role = {"user_message": "USER", "agent_message": "AGENT"}.get(kind)
    text = payload.get("message", "") if role else ""
    if kind == "item_completed":
        item = payload.get("item", {})
        role = {"UserMessage": "USER", "AgentMessage": "AGENT"}.get(item.get("type"))
        if role == "AGENT" and item.get("phase") not in (None, "commentary", "final"):
            return None
        text = _content_text(item.get("content", [])) if role else ""
    if role and isinstance(text, str) and text.strip():
        return Turn(role, text.strip(), record.get("timestamp", ""))
    return None


def _apply_tail(turns: list[Turn], tail: int | None) -> list[Turn]:
    """Trim to the last N user-message turns and everything after."""
    if tail is None:
        return turns
    user_indices = [i for i, t in enumerate(turns) if t.role == "USER"]
    if tail >= len(user_indices):
        return turns
    return turns[user_indices[-tail] :]

"""Codex CLI session reader."""

from __future__ import annotations

import json
from pathlib import Path

from agent_chat_reader.models import SessionMeta, Turn

CODEX_SESSIONS = Path.home() / ".codex" / "sessions"


def is_subagent(path: Path) -> bool:
    """Return True if this Codex session is a guardian/subagent session."""
    with path.open() as fh:
        for raw in fh:
            try:
                rec = json.loads(raw.strip())
                if rec.get("type") == "session_meta":
                    payload = rec.get("payload", {})
                    if payload.get("thread_source") == "subagent":
                        return True
                    source = payload.get("source", {})
                    return isinstance(source, dict) and "subagent" in source
            except Exception:
                pass
    return False


def _first_user_message(path: Path) -> str:
    """Return the first user message text, truncated to 80 chars."""
    with path.open() as fh:
        for raw in fh:
            try:
                rec = json.loads(raw.strip())
                if (
                    rec.get("type") == "event_msg"
                    and rec.get("payload", {}).get("type") == "user_message"
                ):
                    return rec["payload"].get("message", "").replace("\n", " ")[:80]
            except Exception:
                pass
    return ""


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
        if not include_subagents and is_subagent(f):
            continue
        stat = f.stat()
        results.append(
            SessionMeta(
                source="codex",
                id=_session_id_from_path(f),
                path=f,
                mtime=stat.st_mtime,
                size_kb=stat.st_size // 1024,
                title=_first_user_message(f),
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
    turns: list[Turn] = []
    last_assistant_text: str | None = None

    with path.open() as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
                t = rec.get("type", "")
                pt = rec.get("payload", {}).get("type", "")
                ts: str = rec.get("timestamp", "")

                if t == "event_msg" and pt == "user_message":
                    msg = rec["payload"].get("message", "").strip()
                    if msg:
                        turns.append(Turn("USER", msg, ts))
                        last_assistant_text = None

                elif t == "event_msg" and pt == "agent_message":
                    msg = rec["payload"].get("message", "").strip()
                    if msg and msg != last_assistant_text:
                        turns.append(Turn("AGENT", msg, ts))
                        last_assistant_text = msg

                elif t == "response_item":
                    role = rec.get("payload", {}).get("role", "")
                    if role == "assistant":
                        content = rec.get("payload", {}).get("content", [])
                        text = ""
                        if isinstance(content, str):
                            text = content.strip()
                        elif isinstance(content, list):
                            text = "\n".join(
                                b.get("text", "")
                                for b in content
                                if isinstance(b, dict) and b.get("type") == "text"
                            ).strip()
                        if text and text != last_assistant_text:
                            turns.append(Turn("AGENT", text, ts))
                            last_assistant_text = text

            except Exception:
                pass

    return _apply_tail(turns, tail)


def _apply_tail(turns: list[Turn], tail: int | None) -> list[Turn]:
    """Trim to the last N user-message turns and everything after."""
    if tail is None:
        return turns
    user_indices = [i for i, t in enumerate(turns) if t.role == "USER"]
    if tail >= len(user_indices):
        return turns
    return turns[user_indices[-tail] :]

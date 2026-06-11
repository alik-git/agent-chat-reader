"""Claude Code session reader."""

from __future__ import annotations

import json
from pathlib import Path

from agent_chat_reader.models import SessionMeta, Turn

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"


def _decode_project_path(dirname: str) -> str:
    """Decode '-home-ali' -> '/home/ali'."""
    return dirname.replace("-", "/")


def _session_title(path: Path) -> str:
    """Return the last ai-title from a Claude session file."""
    last_title = ""
    with path.open() as fh:
        for raw in fh:
            try:
                rec = json.loads(raw.strip())
                if rec.get("type") == "ai-title":
                    t = rec.get("aiTitle", "").strip()
                    if t:
                        last_title = t
            except Exception:
                pass
    return last_title


def _extract_user_text(content: str | list) -> str | None:  # type: ignore[type-arg]
    """Extract real user text; return None for tool-result carriers."""
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        has_tool_result = any(
            b.get("type") == "tool_result" for b in content if isinstance(b, dict)
        )
        if has_tool_result:
            return None
        text = " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
        return text or None
    return None


def list_sessions() -> list[SessionMeta]:
    """List all Claude Code sessions, sorted by most recent first."""
    if not CLAUDE_PROJECTS.exists():
        return []
    results = []
    for proj_dir in CLAUDE_PROJECTS.iterdir():
        if not proj_dir.is_dir():
            continue
        for f in proj_dir.glob("*.jsonl"):
            stat = f.stat()
            results.append(
                SessionMeta(
                    source="claude",
                    id=f.stem,
                    path=f,
                    mtime=stat.st_mtime,
                    size_kb=stat.st_size // 1024,
                    title=_session_title(f) or "(untitled)",
                )
            )
    return sorted(results, key=lambda s: s.mtime, reverse=True)


def read_turns(
    path: Path,
    *,
    verbose: bool = False,
    include_subagents: bool = False,
    tail: int | None = None,
) -> list[Turn]:
    """Read conversation turns from a Claude Code session file.

    Args:
        path: Path to the session JSONL file.
        verbose: If True, append brief tool call summaries to assistant turns.
        include_subagents: If True, include sidechain (sub-agent) turns.
        tail: If set, return only turns from the last N user messages onward.

    Returns:
        List of Turn namedtuples in conversation order.
    """
    turns: list[Turn] = []

    with path.open() as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
                t = rec.get("type", "")
                is_sidechain = rec.get("isSidechain", False)
                ts: str = rec.get("timestamp", "")

                if not include_subagents and is_sidechain:
                    continue

                if t == "user":
                    content = rec.get("message", {}).get("content", "")
                    text = _extract_user_text(content)
                    if text:
                        turns.append(Turn("USER", text, ts))

                elif t == "assistant":
                    blocks = rec.get("message", {}).get("content", [])
                    if not isinstance(blocks, list):
                        continue
                    text_parts = []
                    tool_parts = []
                    for block in blocks:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type", "")
                        if btype == "text":
                            t_text = block.get("text", "").strip()
                            if t_text:
                                text_parts.append(t_text)
                        elif btype == "tool_use" and verbose:
                            tool_parts.append(_format_tool_call(block))

                    full_text = "\n".join(text_parts).strip()
                    if tool_parts and verbose:
                        full_text = (full_text + "\n" + " ".join(tool_parts)).strip()
                    if full_text:
                        turns.append(Turn("CLAUDE", full_text, ts))

            except Exception:
                pass

    return _apply_tail(turns, tail)


def _format_tool_call(block: dict) -> str:  # type: ignore[type-arg]
    """Format a tool_use block as a short bracketed summary."""
    name = block.get("name", "?")
    inp = block.get("input", {})
    if name == "Bash":
        desc = inp.get("description", inp.get("command", ""))[:60]
    elif name in ("Read", "Edit", "Write"):
        desc = str(inp.get("file_path", ""))
    elif name == "Agent":
        desc = str(inp.get("description", ""))[:60]
    else:
        desc = str(next(iter(inp.values()), ""))[:60]
    return f"[{name}: {desc}]"


def _apply_tail(turns: list[Turn], tail: int | None) -> list[Turn]:
    """Trim to the last N user-message turns and everything after."""
    if tail is None:
        return turns
    user_indices = [i for i, t in enumerate(turns) if t.role == "USER"]
    if tail >= len(user_indices):
        return turns
    return turns[user_indices[-tail] :]

"""Output formatting helpers."""

from __future__ import annotations

import textwrap
from datetime import datetime

from agent_chat_reader.models import SessionMeta, Turn

_MAX_TURN_LEN = 3000
_WRAP_THRESHOLD = 400


def fmt_ts(ts_str: str) -> str:
    """Format an ISO timestamp to a short local time string."""
    if not ts_str:
        return ""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ts_str[:16]


def fmt_mtime(mtime: float) -> str:
    """Format a Unix mtime float to a short local time string."""
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")  # noqa: DTZ006


def print_turn(turn: Turn) -> None:
    """Print a single conversation turn with a header rule."""
    ts_str = f"  {fmt_ts(turn.timestamp)}" if turn.timestamp else ""
    print(f"\n{'─'*60}")
    print(f"[{turn.role}]{ts_str}")
    print("─" * 60)
    text = turn.text
    if len(text) <= _WRAP_THRESHOLD:
        print(textwrap.fill(text, width=100))
    else:
        print(text[:_MAX_TURN_LEN] + ("…" if len(text) > _MAX_TURN_LEN else ""))


def print_session_list(sessions: list[SessionMeta]) -> None:
    """Print a formatted table of sessions."""
    print(f"{'SRC':<6} {'DATE':<16} {'SIZE':>7}  {'ID':<36}  TITLE")
    print("─" * 100)
    for s in sessions:
        src = s.source
        date = fmt_mtime(s.mtime)
        size = f"{s.size_kb}KB"
        title = (s.title or "(no title)")[:55]
        print(f"{src:<6} {date:<16} {size:>7}  {s.id:<36}  {title}")


def print_find_result(
    session: SessionMeta,
    hits: list[tuple[str, str]],
    *,
    max_hits: int = 4,
) -> None:
    """Print a single session's find results."""
    print(f"\n{'='*70}")
    print(f"[{session.source.upper()}] {session.id}  {fmt_mtime(session.mtime)}")
    if session.title:
        print(f'  "{session.title[:60]}"')
    for role, snippet in hits[:max_hits]:
        print(f"  [{role}] {snippet!r}")
    if len(hits) > max_hits:
        print(f"  ... and {len(hits) - max_hits} more matches")

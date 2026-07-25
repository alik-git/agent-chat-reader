"""Output formatting helpers."""

from __future__ import annotations

import textwrap
from datetime import datetime, timedelta

from agent_chat_reader.models import SessionMeta, Turn

_MAX_TURN_LEN = 3000
_WRAP_THRESHOLD = 400

# Type alias: (role, snippet, timestamp)
FindHit = tuple[str, str, str]


def fmt_ts(ts_str: str) -> str:
    """Format an ISO timestamp to a short local time string."""
    if not ts_str:
        return ""
    dt = _parse_ts(ts_str)
    if dt is not None:
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    return ts_str[:16]


def _parse_ts(ts_str: str) -> datetime | None:
    """Parse an ISO timestamp, returning None for non-ISO placeholders."""
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


def fmt_elapsed(delta: timedelta) -> str:
    """Format an elapsed duration compactly."""
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    minutes %= 60
    if hours < 24:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    days = hours // 24
    hours %= 24
    return f"{days}d {hours}h" if hours else f"{days}d"


def elapsed_seconds_between(previous_turn: Turn, turn: Turn) -> int | None:
    """Return elapsed seconds between two turns, if both timestamps parse."""
    current_dt = _parse_ts(turn.timestamp)
    previous_dt = _parse_ts(previous_turn.timestamp)
    if current_dt is None or previous_dt is None:
        return None
    return max(0, int((current_dt - previous_dt).total_seconds()))


def _timestamp_suffix(turn: Turn, previous_turn: Turn | None) -> str:
    """Return the optional timestamp/gap suffix for a turn header."""
    timestamp = fmt_ts(turn.timestamp)
    if not timestamp:
        return ""

    if previous_turn is None:
        return f"  {timestamp}"

    elapsed_seconds = elapsed_seconds_between(previous_turn, turn)
    if elapsed_seconds is None:
        return f"  {timestamp}"

    elapsed = fmt_elapsed(timedelta(seconds=elapsed_seconds))
    role = turn.role.lower()
    return f"  {timestamp}  ({role} took {elapsed})"


def fmt_mtime(mtime: float) -> str:
    """Format a Unix mtime float to a short local time string."""
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")  # noqa: DTZ006


def print_turn(
    turn: Turn,
    *,
    show_timestamps: bool = True,
    previous_turn: Turn | None = None,
) -> None:
    """Print a single conversation turn with a header rule."""
    ts_str = _timestamp_suffix(turn, previous_turn) if show_timestamps else ""
    print(f"\n{'─' * 60}")
    print(f"[{turn.role}]{ts_str}")
    print("─" * 60)
    text = turn.text
    if len(text) <= _WRAP_THRESHOLD:
        print(textwrap.fill(text, width=100))
    else:
        print(text[:_MAX_TURN_LEN] + ("…" if len(text) > _MAX_TURN_LEN else ""))


def print_session_list(sessions: list[SessionMeta]) -> None:
    """Print a formatted table of sessions."""
    print(f"{'SRC':<10} {'DATE':<16} {'SIZE':>7}  {'ID':<36}  TITLE")
    print("─" * 104)
    for s in sessions:
        src = s.source
        date = fmt_mtime(s.mtime)
        size = f"{s.size_kb}KB"
        title = (s.title or "(no title)")[:55]
        print(f"{src:<10} {date:<16} {size:>7}  {s.id:<36}  {title}")


def print_find_result(
    session: SessionMeta,
    hits: list[FindHit],
    *,
    max_hits: int = 4,
) -> None:
    """Print a single session's find results."""
    print(f"\n{'=' * 70}")
    print(f"[{session.source.upper()}] {session.id}  {fmt_mtime(session.mtime)}")
    if session.title:
        print(f'  "{session.title[:60]}"')
    for role, snippet, ts in hits[:max_hits]:
        ts_str = f" {fmt_ts(ts)}" if ts else ""
        print(f"  [{role}{ts_str}] {snippet!r}")
    if len(hits) > max_hits:
        print(f"  ... and {len(hits) - max_hits} more matches")

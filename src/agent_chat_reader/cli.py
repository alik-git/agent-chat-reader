"""Command-line entry point for agent-chat-reader."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from agent_chat_reader import __version__, claude, codex, codex_side, search
from agent_chat_reader.models import SessionMeta, Turn
from agent_chat_reader.output import (
    elapsed_seconds_between,
    fmt_ts,
    print_find_result,
    print_session_list,
    print_turn,
)


def _find_session(
    session_id: str,
    *,
    include_subagents: bool = False,
) -> SessionMeta | None:
    """Locate a session by ID fragment."""
    p = Path(session_id)
    if p.exists():
        source = "codex" if ".codex" in str(p) else "claude"
        stat = p.stat()
        return SessionMeta(
            source=source,
            id=_session_id_for_path(p, source),
            path=p,
            mtime=stat.st_mtime,
            size_kb=stat.st_size // 1024,
            title=p.name,
        )

    codex_matches = list(codex.CODEX_SESSIONS.rglob(f"*{session_id}*.jsonl"))
    if codex_matches:
        path = sorted(codex_matches)[-1]
        stat = path.stat()
        return SessionMeta(
            source="codex",
            id=codex._session_id_from_path(path),
            path=path,
            mtime=stat.st_mtime,
            size_kb=stat.st_size // 1024,
            title=path.name,
        )

    claude_matches = list(claude.CLAUDE_PROJECTS.rglob(f"*{session_id}*.jsonl"))
    if claude_matches:
        path = sorted(claude_matches)[-1]
        stat = path.stat()
        return SessionMeta(
            source="claude",
            id=path.stem,
            path=path,
            mtime=stat.st_mtime,
            size_kb=stat.st_size // 1024,
            title=path.name,
        )

    side_chat = codex_side.find_session(
        session_id,
        include_subagents=include_subagents,
    )
    if side_chat is not None:
        return side_chat

    return None


def _include_codex_sessions(source_filter: str | None) -> bool:
    """Return True if normal Codex sessions should be included."""
    return source_filter in (None, "codex")


def _include_codex_side_chats(source_filter: str | None) -> bool:
    """Return True if Codex side chats should be included."""
    return source_filter in (None, "codex", "codex-side")


def _include_claude_sessions(source_filter: str | None) -> bool:
    """Return True if Claude sessions should be included."""
    return source_filter in (None, "claude")


def _search_sources(source_filter: str | None) -> tuple[str, ...]:
    """Return exact indexed sources for the CLI's source-filter semantics."""
    sources: list[str] = []
    if _include_codex_sessions(source_filter):
        sources.append("codex")
    if _include_codex_side_chats(source_filter):
        sources.append(codex_side.CODEX_SIDE_SOURCE)
    if _include_claude_sessions(source_filter):
        sources.append("claude")
    return tuple(sources)


def _positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parse_since(value: str) -> float:
    """Parse a local ISO date or an ISO timestamp for search filtering."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an ISO date or timestamp, for example 2026-07-01"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.timestamp()


def cmd_list(
    *,
    source_filter: str | None,
    include_subagents: bool,
    limit: int,
) -> int:
    """List recent sessions from both sources."""
    sessions: list[SessionMeta] = []
    if _include_codex_sessions(source_filter):
        sessions += codex.list_sessions(include_subagents=include_subagents)
    if _include_codex_side_chats(source_filter):
        sessions += codex_side.list_sessions(include_subagents=include_subagents)
    if _include_claude_sessions(source_filter):
        sessions += claude.list_sessions()

    sessions = sorted(sessions, key=lambda s: s.mtime, reverse=True)[:limit]
    print_session_list(sessions)
    return 0


def cmd_find(
    keywords: list[str],
    *,
    source_filter: str | None,
    include_subagents: bool,
    limit: int,
    since: float | None,
) -> int:
    """Search the incremental index with session-level AND semantics."""
    try:
        results = search.search_sessions(
            keywords,
            sources=_search_sources(source_filter),
            include_subagents=include_subagents,
            limit=limit,
            since=since,
        )
    except (OSError, sqlite3.Error) as exc:
        print(f"Search index error: {exc}", file=sys.stderr)
        return 1

    if not results:
        query = " AND ".join(f"{k!r}" for k in keywords)
        print(f"No sessions found containing {query}")
        return 0

    for result in results:
        print_find_result(result.session, result.hits)

    return 0


def cmd_read(
    session_id: str,
    *,
    verbose: bool,
    include_subagents: bool,
    tail: int | None,
    show_timestamps: bool,
    output_format: str,
) -> int:
    """Read a specific session as clean conversation."""
    session = _find_session(session_id, include_subagents=include_subagents)
    if session is None:
        print(f"Session not found: {session_id}", file=sys.stderr)
        return 1

    turns = _read_session_turns(
        session,
        verbose=verbose,
        include_subagents=include_subagents,
        tail=tail,
    )

    if output_format == "json":
        print(
            _json_session(
                source=session.source,
                session_id=session.id,
                path=session.path,
                size_kb=session.size_kb,
                turns=turns,
            )
        )
        return 0

    print(f"Source: {session.source.upper()}  |  {session.id}  |  {session.size_kb}KB")

    if not turns:
        print("(no conversation turns found)")
        return 0

    previous_turn = None
    for turn in turns:
        print_turn(
            turn,
            show_timestamps=show_timestamps,
            previous_turn=previous_turn,
        )
        previous_turn = turn

    print(f"\n{'─' * 60}")
    print(f"Total turns: {len(turns)}")
    return 0


def _read_session_turns(
    session: SessionMeta,
    *,
    verbose: bool,
    include_subagents: bool,
    tail: int | None = None,
) -> list[Turn]:
    """Read turns for any supported session source."""
    if session.source == "codex":
        return codex.read_turns(session.path, tail=tail)
    if session.source == codex_side.CODEX_SIDE_SOURCE:
        return codex_side.read_turns(session.id, tail=tail)
    return claude.read_turns(
        session.path,
        verbose=verbose,
        include_subagents=include_subagents,
        tail=tail,
    )


def _json_session(
    *,
    source: str,
    path: Path,
    size_kb: int,
    turns: list[Turn],
    session_id: str | None = None,
) -> str:
    """Serialize a read session as structured JSON."""
    previous_turn = None
    turn_records: list[dict[str, object]] = []
    for turn in turns:
        elapsed_seconds: int | None = (
            None
            if previous_turn is None
            else elapsed_seconds_between(previous_turn, turn)
        )
        turn_records.append(
            {
                "role": turn.role,
                "timestamp": turn.timestamp or None,
                "local_time": fmt_ts(turn.timestamp) or None,
                "elapsed_seconds": elapsed_seconds,
                "text": turn.text,
            }
        )
        previous_turn = turn

    payload = {
        "source": source,
        "session_id": session_id or _session_id_for_path(path, source),
        "path": str(path),
        "size_kb": size_kb,
        "total_turns": len(turns),
        "turns": turn_records,
    }
    return json.dumps(payload, indent=2)


def _session_id_for_path(path: Path, source: str) -> str:
    """Return the source-specific session id for a session file."""
    if source == "codex":
        return codex._session_id_from_path(path)
    if source == codex_side.CODEX_SIDE_SOURCE:
        return path.stem
    return path.stem


def main(argv: list[str] | None = None) -> int:
    """Run the agent-chat-reader command-line interface."""
    p = argparse.ArgumentParser(
        prog="agent-chat-reader",
        description="Read and search Codex CLI and Claude Code chat history.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"agent-chat-reader {__version__}",
    )
    p.add_argument("session", nargs="?", help="Session ID or file path to read")
    p.add_argument("--list", "-l", action="store_true", help="List recent sessions")
    p.add_argument(
        "--find",
        "-f",
        nargs="+",
        metavar="KEYWORD",
        help="Search sessions (multiple keywords = AND logic)",
    )
    p.add_argument(
        "--source",
        choices=["codex", "codex-side", "claude"],
        help="Filter to one source",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Include tool call summaries (Claude sessions)",
    )
    p.add_argument(
        "--tail",
        "-n",
        type=int,
        metavar="N",
        help="Show only the last N user turns",
    )
    p.add_argument(
        "--include-subagents",
        action="store_true",
        help="Include guardian/subagent sessions",
    )
    p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for session reads (default: text)",
    )
    p.add_argument(
        "--show-timestamps",
        action="store_true",
        dest="show_timestamps",
        default=True,
        help="Show message timestamps and elapsed gaps in session reads",
    )
    p.add_argument(
        "--hide-timestamps",
        action="store_false",
        dest="show_timestamps",
        help="Hide message timestamps and elapsed gaps in session reads",
    )
    p.add_argument(
        "--limit",
        type=_positive_int,
        default=40,
        help="Max sessions for --list or --find (default: 40)",
    )
    p.add_argument(
        "--since",
        type=_parse_since,
        metavar="DATE",
        help="Only search sessions active on or after an ISO date/timestamp",
    )

    args = p.parse_args(argv)

    if args.list:
        return cmd_list(
            source_filter=args.source,
            include_subagents=args.include_subagents,
            limit=args.limit,
        )
    if args.find:
        return cmd_find(
            args.find,  # list[str] from nargs="+"
            source_filter=args.source,
            include_subagents=args.include_subagents,
            limit=args.limit,
            since=args.since,
        )
    if args.session:
        return cmd_read(
            args.session,
            verbose=args.verbose,
            include_subagents=args.include_subagents,
            tail=args.tail,
            show_timestamps=args.show_timestamps,
            output_format=args.format,
        )

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

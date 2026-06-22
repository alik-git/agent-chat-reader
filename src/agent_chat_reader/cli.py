"""Command-line entry point for agent-chat-reader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_chat_reader import __version__, claude, codex
from agent_chat_reader.models import SessionMeta, Turn
from agent_chat_reader.output import (
    FindHit,
    elapsed_seconds_between,
    fmt_ts,
    gap_kind_for,
    print_find_result,
    print_session_list,
    print_turn,
)


def _find_session(session_id: str) -> tuple[Path, str] | None:
    """Locate a session file by ID fragment, returning (path, source)."""
    p = Path(session_id)
    if p.exists():
        source = "codex" if ".codex" in str(p) else "claude"
        return p, source

    codex_matches = list(codex.CODEX_SESSIONS.rglob(f"*{session_id}*.jsonl"))
    if codex_matches:
        return sorted(codex_matches)[-1], "codex"

    claude_matches = list(claude.CLAUDE_PROJECTS.rglob(f"*{session_id}*.jsonl"))
    if claude_matches:
        return sorted(claude_matches)[-1], "claude"

    return None


def cmd_list(
    *,
    source_filter: str | None,
    include_subagents: bool,
    limit: int,
) -> int:
    """List recent sessions from both sources."""
    sessions: list[SessionMeta] = []
    if source_filter != "claude":
        sessions += codex.list_sessions(include_subagents=include_subagents)
    if source_filter != "codex":
        sessions += claude.list_sessions()

    sessions = sorted(sessions, key=lambda s: s.mtime, reverse=True)[:limit]
    print_session_list(sessions)
    return 0


def _title_key(title: str) -> str:
    """Normalise a session title for dedup grouping."""
    return title.strip()[:80].lower()


def cmd_find(
    keywords: list[str],
    *,
    source_filter: str | None,
    include_subagents: bool,
) -> int:
    """Search sessions for keywords (AND logic), deduplicating continuation sessions."""
    sessions: list[SessionMeta] = []
    if source_filter != "claude":
        sessions += codex.list_sessions(include_subagents=include_subagents)
    if source_filter != "codex":
        sessions += claude.list_sessions()

    sessions = sorted(sessions, key=lambda s: s.mtime, reverse=True)
    keywords_lower = [k.lower() for k in keywords]

    def _turn_matches(text: str) -> bool:
        """Return True if the turn contains all keywords (AND)."""
        text_lower = text.lower()
        return all(k in text_lower for k in keywords_lower)

    # Collect hits per session, then group by title to deduplicate continuations.
    # Each group keeps the most-recent session's metadata as the representative.
    groups: dict[str, tuple[SessionMeta, list[FindHit], int]] = {}

    for s in sessions:
        try:
            if s.source == "codex":
                turns = codex.read_turns(s.path)
            else:
                turns = claude.read_turns(s.path)
        except Exception:
            continue

        hits: list[FindHit] = [
            (t.role, t.text[:120].replace("\n", " "), t.timestamp)
            for t in turns
            if _turn_matches(t.text)
        ]
        if not hits:
            continue

        key = _title_key(s.title)
        if key not in groups:
            groups[key] = (s, hits, 1)
        else:
            rep, existing_hits, count = groups[key]
            rep = s if s.mtime > rep.mtime else rep
            groups[key] = (rep, existing_hits + hits, count + 1)

    if not groups:
        query = " AND ".join(f"{k!r}" for k in keywords)
        print(f"No sessions found containing {query}")
        return 0

    for rep, hits, count in groups.values():
        print_find_result(rep, hits, merged_count=count)

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
    result = _find_session(session_id)
    if result is None:
        print(f"Session not found: {session_id}", file=sys.stderr)
        return 1

    path, source = result
    size_kb = path.stat().st_size // 1024

    if source == "codex":
        turns = codex.read_turns(path, tail=tail)
    else:
        turns = claude.read_turns(
            path, verbose=verbose, include_subagents=include_subagents, tail=tail
        )

    if output_format == "json":
        print(_json_session(source=source, path=path, size_kb=size_kb, turns=turns))
        return 0

    print(f"Source: {source.upper()}  |  {path.name}  |  {size_kb}KB")

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


def _json_session(*, source: str, path: Path, size_kb: int, turns: list[Turn]) -> str:
    """Serialize a read session as structured JSON."""
    previous_turn = None
    turn_records = []
    for turn in turns:
        elapsed_seconds = (
            None
            if previous_turn is None
            else elapsed_seconds_between(previous_turn, turn)
        )
        gap_kind = None if previous_turn is None else gap_kind_for(previous_turn, turn)
        turn_records.append(
            {
                "role": turn.role,
                "timestamp": turn.timestamp or None,
                "local_time": fmt_ts(turn.timestamp) or None,
                "elapsed_seconds": elapsed_seconds,
                "gap_kind": gap_kind,
                "text": turn.text,
            }
        )
        previous_turn = turn

    payload = {
        "source": source,
        "session_id": _session_id_for_path(path, source),
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
    p.add_argument("--source", choices=["codex", "claude"], help="Filter to one source")
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
        type=int,
        default=40,
        help="Max sessions for --list (default: 40)",
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

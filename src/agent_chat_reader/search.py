"""Session-level search over the incrementally refreshed local index."""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from agent_chat_reader.models import SessionMeta
from agent_chat_reader.output import FindHit
from agent_chat_reader.search_index import (
    default_index_path,
    open_index,
    refresh_index,
)

_SNIPPET_CONTEXT = 70


class SearchResult(NamedTuple):
    """One matching session and its most useful message snippets."""

    session: SessionMeta
    hits: list[FindHit]
    score: float


class _MessageMatch(NamedTuple):
    """One indexed message that contains a query term."""

    message_id: int
    session_rowid: int
    ordinal: int
    role: str
    timestamp: str
    text: str
    score: float


def _fts_literal(keyword: str) -> str:
    """Quote user text as one literal FTS phrase instead of query syntax."""
    return f'"{keyword.replace(chr(34), chr(34) * 2)}"'


def _centered_snippet(text: str, keywords: list[str]) -> str:
    """Return compact text centered on the earliest matching keyword."""
    matches = [
        match
        for keyword in keywords
        if (match := re.search(re.escape(keyword), text, flags=re.IGNORECASE))
        is not None
    ]
    if not matches:
        return " ".join(text[: 2 * _SNIPPET_CONTEXT].split())

    match = min(matches, key=lambda item: item.start())
    start = max(0, match.start() - _SNIPPET_CONTEXT)
    end = min(len(text), match.end() + _SNIPPET_CONTEXT)
    snippet = " ".join(text[start:end].split())
    if start:
        snippet = f"…{snippet}"
    if end < len(text):
        snippet = f"{snippet}…"
    return snippet


def _message_rows(
    conn: sqlite3.Connection,
    *,
    keyword: str,
    visibility: int,
    sources: tuple[str, ...],
    since: float | None,
) -> list[_MessageMatch]:
    """Return messages containing one literal keyword under the query filters."""
    placeholders = ", ".join("?" for _source in sources)
    filters = ["s.visibility = ?", f"s.source IN ({placeholders})"]
    parameters: list[object] = [visibility, *sources]
    if since is not None:
        filters.append("s.mtime >= ?")
        parameters.append(since)

    if len(keyword) >= 3:
        rows = conn.execute(
            f"""
            SELECT
                m.id AS message_id, m.session_rowid, m.ordinal,
                m.role, m.timestamp, m.text, bm25(messages_fts) AS score
            FROM messages_fts
            JOIN messages AS m ON m.id = messages_fts.rowid
            JOIN sessions AS s ON s.id = m.session_rowid
            WHERE messages_fts MATCH ? AND {" AND ".join(filters)}
            """,  # noqa: S608 - placeholders hold every user-provided value
            [_fts_literal(keyword), *parameters],
        )
        return [_MessageMatch(**dict(row)) for row in rows]

    rows = conn.execute(
        f"""
        SELECT
            m.id AS message_id, m.session_rowid, m.ordinal,
            m.role, m.timestamp, m.text, 0.0 AS score
        FROM messages AS m
        JOIN sessions AS s ON s.id = m.session_rowid
        WHERE {" AND ".join(filters)}
        """,  # noqa: S608 - only fixed column predicates are interpolated
        parameters,
    )
    folded = keyword.casefold()
    return [
        _MessageMatch(**dict(row))
        for row in rows
        if folded in str(row["text"]).casefold()
    ]


def _session_meta(conn: sqlite3.Connection, rowid: int) -> SessionMeta:
    """Load public session metadata from one indexed row."""
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (rowid,)).fetchone()
    if row is None:
        raise LookupError(f"Indexed session disappeared: {rowid}")
    return SessionMeta(
        source=str(row["source"]),
        id=str(row["session_id"]),
        path=Path(str(row["path"])),
        mtime=float(row["mtime"]),
        size_kb=int(row["size_kb"]),
        title=str(row["title"]),
    )


def _order_covering_keywords(
    messages: dict[int, _MessageMatch],
    matched_keywords: dict[int, list[str]],
    keywords: list[str],
) -> list[_MessageMatch]:
    """Order snippets so the visible prefix explains every matched term."""
    remaining = set(keywords)
    pool = list(messages.values())
    ordered: list[_MessageMatch] = []
    while remaining and pool:
        best = min(
            pool,
            key=lambda match: (
                -len(set(matched_keywords[match.message_id]) & remaining),
                -len(matched_keywords[match.message_id]),
                match.score,
                match.ordinal,
            ),
        )
        covered = set(matched_keywords[best.message_id]) & remaining
        if not covered:
            break
        ordered.append(best)
        pool.remove(best)
        remaining -= covered

    pool.sort(
        key=lambda match: (
            -len(matched_keywords[match.message_id]),
            match.score,
            match.ordinal,
        )
    )
    return ordered + pool


def _session_hits(
    keyword_matches: list[list[_MessageMatch]],
    keywords: list[str],
    session_rowid: int,
) -> list[FindHit]:
    """Choose match-centered snippets for one result session."""
    messages: dict[int, _MessageMatch] = {}
    matched_keywords: dict[int, list[str]] = defaultdict(list)
    for keyword, matches in zip(keywords, keyword_matches, strict=True):
        for match in matches:
            if match.session_rowid == session_rowid:
                messages[match.message_id] = match
                matched_keywords[match.message_id].append(keyword)

    ordered = _order_covering_keywords(messages, matched_keywords, keywords)
    return [
        (
            match.role,
            _centered_snippet(match.text, matched_keywords[match.message_id]),
            match.timestamp,
        )
        for match in ordered
    ]


def _query_index(
    conn: sqlite3.Connection,
    *,
    keywords: list[str],
    sources: tuple[str, ...],
    include_subagents: bool,
    limit: int,
    since: float | None,
) -> list[SearchResult]:
    """Search each term independently, then intersect at session scope."""
    keyword_matches = [
        _message_rows(
            conn,
            keyword=keyword,
            visibility=int(include_subagents),
            sources=sources,
            since=since,
        )
        for keyword in keywords
    ]
    session_sets = [
        {match.session_rowid for match in matches} for matches in keyword_matches
    ]
    if not session_sets or any(not session_ids for session_ids in session_sets):
        return []

    results: list[SearchResult] = []
    for session_rowid in set.intersection(*session_sets):
        score = sum(
            min(
                match.score for match in matches if match.session_rowid == session_rowid
            )
            for matches in keyword_matches
        )
        results.append(
            SearchResult(
                session=_session_meta(conn, session_rowid),
                hits=_session_hits(keyword_matches, keywords, session_rowid),
                score=score,
            )
        )
    results.sort(
        key=lambda result: (
            result.score,
            -result.session.mtime,
            result.session.source,
            result.session.id,
        )
    )
    return results[:limit]


def search_sessions(
    keywords: list[str],
    *,
    sources: tuple[str, ...],
    include_subagents: bool,
    limit: int,
    since: float | None,
    index_path: Path | None = None,
) -> list[SearchResult]:
    """Refresh the disposable index and return ranked matching sessions."""
    clean_keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
    if not clean_keywords or not sources or limit <= 0:
        return []
    with open_index(index_path or default_index_path()) as conn:
        with conn:
            refresh_index(
                conn,
                sources=sources,
                include_subagents=include_subagents,
            )
        return _query_index(
            conn,
            keywords=clean_keywords,
            sources=sources,
            include_subagents=include_subagents,
            limit=limit,
            since=since,
        )

"""Incremental local search index for clean chat turns."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from agent_chat_reader import claude, codex, codex_side
from agent_chat_reader.models import SessionMeta, Turn

_SCHEMA_VERSION = 3
_CACHE_ENV = "AGENT_CHAT_READER_CACHE_DIR"


class _FileState(NamedTuple):
    """Identity and append cursor inputs for one JSONL source file."""

    identity: str
    fingerprint: str
    size: int
    mtime: float


class _FileCandidate(NamedTuple):
    """One JSONL source discovered without opening its contents."""

    source: str
    session_id: str
    path: Path
    state: _FileState


class _FileMetadata(NamedTuple):
    """Cached content-derived metadata for one JSONL source file."""

    title: str
    is_subagent: bool


class _IndexedFile(NamedTuple):
    """A searchable session plus its current source-file state."""

    session: SessionMeta
    state: _FileState
    is_subagent: bool


def default_index_path() -> Path:
    """Return the disposable local search-index path."""
    configured = os.environ.get(_CACHE_ENV)
    cache_dir = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return cache_dir / "agent-chat-reader" / "search.sqlite3"


def _drop_schema(conn: sqlite3.Connection) -> None:
    """Drop all derived index objects before a schema rebuild."""
    conn.executescript(
        """
        DROP TRIGGER IF EXISTS messages_ai;
        DROP TRIGGER IF EXISTS messages_ad;
        DROP TRIGGER IF EXISTS messages_au;
        DROP TABLE IF EXISTS messages_fts;
        DROP TABLE IF EXISTS messages;
        DROP TABLE IF EXISTS sessions;
        DROP TABLE IF EXISTS source_files;
        DROP TABLE IF EXISTS metadata;
        """
    )


def _create_schema(conn: sqlite3.Connection) -> None:
    """Create the versioned session, message, and trigram FTS schema."""
    conn.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE source_files (
            source TEXT NOT NULL,
            session_id TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            title TEXT NOT NULL,
            is_subagent INTEGER NOT NULL,
            PRIMARY KEY (source, session_id)
        );

        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY,
            visibility INTEGER NOT NULL,
            source TEXT NOT NULL,
            session_id TEXT NOT NULL,
            path TEXT NOT NULL,
            mtime REAL NOT NULL,
            size_kb INTEGER NOT NULL,
            title TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            file_identity TEXT NOT NULL,
            indexed_bytes INTEGER NOT NULL,
            UNIQUE (visibility, source, session_id)
        );

        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            session_rowid INTEGER NOT NULL
                REFERENCES sessions(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            role TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            text TEXT NOT NULL
        );

        CREATE INDEX messages_session_idx
            ON messages(session_rowid, ordinal);
        CREATE INDEX sessions_scope_idx
            ON sessions(visibility, source, mtime DESC);

        CREATE VIRTUAL TABLE messages_fts USING fts5(
            text,
            content='messages',
            content_rowid='id',
            tokenize='trigram'
        );

        CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
        END;
        CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, text)
            VALUES ('delete', old.id, old.text);
        END;
        CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, text)
            VALUES ('delete', old.id, old.text);
            INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
        END;

        """
    )
    conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create or rebuild the disposable index for the current schema version."""
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version == _SCHEMA_VERSION:
        return
    _drop_schema(conn)
    _create_schema(conn)


def open_index(index_path: Path) -> sqlite3.Connection:
    """Open the local index with safe defaults for short CLI transactions."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(index_path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    _ensure_schema(conn)
    return conn


def _metadata_get(conn: sqlite3.Connection, key: str) -> str | None:
    """Read one incremental-source cursor from the index."""
    row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row is not None else None


def _metadata_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert one incremental-source cursor in the index."""
    conn.execute(
        """
        INSERT INTO metadata(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def _file_state(path: Path) -> _FileState | None:
    """Return replacement identity, mutation fingerprint, and append size."""
    try:
        stat = path.stat()
    except OSError:
        return None
    resolved = path.resolve()
    identity = f"{resolved}:{stat.st_dev}:{stat.st_ino}"
    return _FileState(
        identity=identity,
        fingerprint=f"{identity}:{stat.st_mtime_ns}:{stat.st_size}",
        size=stat.st_size,
        mtime=stat.st_mtime,
    )


def _source_paths(source: str) -> Iterator[Path]:
    """Yield JSONL paths for one file-backed source without reading them."""
    if source == "codex":
        if codex.CODEX_SESSIONS.exists():
            yield from codex.CODEX_SESSIONS.rglob("*.jsonl")
        return

    if not claude.CLAUDE_PROJECTS.exists():
        return
    for project_dir in claude.CLAUDE_PROJECTS.iterdir():
        if project_dir.is_dir():
            yield from project_dir.glob("*.jsonl")


def _discover_source_files(source: str) -> list[_FileCandidate]:
    """Discover file-backed sessions using one stat call per path."""
    candidates: list[_FileCandidate] = []
    for path in _source_paths(source):
        state = _file_state(path)
        if state is None:
            continue
        session_id = (
            codex._session_id_from_path(path) if source == "codex" else path.stem
        )
        candidates.append(
            _FileCandidate(
                source=source,
                session_id=session_id,
                path=path,
                state=state,
            )
        )
    return candidates


def _read_file_metadata(candidate: _FileCandidate) -> _FileMetadata:
    """Read title and visibility metadata from a new or changed source file."""
    if candidate.source == "codex":
        metadata = codex.session_metadata(candidate.path)
        return _FileMetadata(
            title=metadata.title,
            is_subagent=metadata.is_subagent,
        )
    return _FileMetadata(
        title=claude.session_title(candidate.path) or "(untitled)",
        is_subagent=False,
    )


def _cached_file_metadata(
    conn: sqlite3.Connection,
    *,
    source: str,
) -> dict[str, sqlite3.Row]:
    """Load cached source fingerprints and content-derived metadata."""
    return {
        str(row["session_id"]): row
        for row in conn.execute(
            """
            SELECT session_id, fingerprint, title, is_subagent
            FROM source_files
            WHERE source = ?
            """,
            (source,),
        )
    }


def _store_file_metadata(
    conn: sqlite3.Connection,
    *,
    candidate: _FileCandidate,
    metadata: _FileMetadata,
) -> None:
    """Store content-derived metadata under the file's current fingerprint."""
    conn.execute(
        """
        INSERT INTO source_files(
            source, session_id, fingerprint, title, is_subagent
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source, session_id) DO UPDATE SET
            fingerprint = excluded.fingerprint,
            title = excluded.title,
            is_subagent = excluded.is_subagent
        """,
        (
            candidate.source,
            candidate.session_id,
            candidate.state.fingerprint,
            metadata.title,
            int(metadata.is_subagent),
        ),
    )


def _prepare_source_file(
    conn: sqlite3.Connection,
    *,
    candidate: _FileCandidate,
    cached: sqlite3.Row | None,
) -> _IndexedFile | None:
    """Reuse cached metadata or inspect one new or changed source file."""
    if cached is not None and str(cached["fingerprint"]) == candidate.state.fingerprint:
        metadata = _FileMetadata(
            title=str(cached["title"]),
            is_subagent=bool(cached["is_subagent"]),
        )
    else:
        metadata = _read_file_metadata(candidate)
        refreshed_state = _file_state(candidate.path)
        if refreshed_state is None:
            return None
        candidate = candidate._replace(state=refreshed_state)
        _store_file_metadata(conn, candidate=candidate, metadata=metadata)

    return _IndexedFile(
        session=SessionMeta(
            source=candidate.source,
            id=candidate.session_id,
            path=candidate.path,
            mtime=candidate.state.mtime,
            size_kb=candidate.state.size // 1024,
            title=metadata.title,
        ),
        state=candidate.state,
        is_subagent=metadata.is_subagent,
    )


def _database_identity(path: Path) -> str | None:
    """Return a stable identity that changes when a source database is replaced."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return f"{path.resolve()}:{stat.st_dev}:{stat.st_ino}"


def _timestamp_value(timestamp: str) -> float | None:
    """Convert an ISO turn timestamp to epoch seconds when possible."""
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp).timestamp()
    except ValueError:
        return None


def _read_file_turns_from(
    session: SessionMeta,
    *,
    include_subagents: bool,
    offset: int,
    last_assistant_text: str | None,
) -> tuple[list[Turn], int]:
    """Read complete records after one file-backed session's byte cursor."""
    if session.source == "codex":
        return codex.read_turns_from(
            session.path,
            offset=offset,
            last_assistant_text=last_assistant_text,
        )
    return claude.read_turns_from(
        session.path,
        offset=offset,
        include_subagents=include_subagents,
    )


def _replace_session(
    conn: sqlite3.Connection,
    *,
    visibility: int,
    session: SessionMeta,
    fingerprint: str,
    file_identity: str,
    indexed_bytes: int,
    turns: list[Turn],
) -> None:
    """Atomically replace one indexed session and all of its messages."""
    conn.execute(
        "DELETE FROM sessions WHERE visibility = ? AND source = ? AND session_id = ?",
        (visibility, session.source, session.id),
    )
    cursor = conn.execute(
        """
        INSERT INTO sessions(
            visibility, source, session_id, path, mtime, size_kb, title,
            fingerprint, file_identity, indexed_bytes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            visibility,
            session.source,
            session.id,
            str(session.path),
            session.mtime,
            session.size_kb,
            session.title,
            fingerprint,
            file_identity,
            indexed_bytes,
        ),
    )
    if cursor.lastrowid is None:
        raise sqlite3.DatabaseError("SQLite did not return an inserted session id")
    session_rowid = cursor.lastrowid
    conn.executemany(
        """
        INSERT INTO messages(session_rowid, ordinal, role, timestamp, text)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (session_rowid, ordinal, turn.role, turn.timestamp, turn.text)
            for ordinal, turn in enumerate(turns)
        ],
    )


def _last_assistant_text(conn: sqlite3.Connection, session_rowid: int) -> str | None:
    """Return parser state needed to deduplicate an appended Codex record."""
    row = conn.execute(
        """
        SELECT role, text FROM messages
        WHERE session_rowid = ?
        ORDER BY ordinal DESC
        LIMIT 1
        """,
        (session_rowid,),
    ).fetchone()
    if row is None or str(row["role"]) == "USER":
        return None
    return str(row["text"])


def _append_session(
    conn: sqlite3.Connection,
    *,
    session_rowid: int,
    session: SessionMeta,
    state: _FileState,
    indexed_bytes: int,
    turns: list[Turn],
) -> None:
    """Append newly parsed turns and advance one file-backed byte cursor."""
    row = conn.execute(
        """
        SELECT COALESCE(MAX(ordinal), -1) AS ordinal
        FROM messages
        WHERE session_rowid = ?
        """,
        (session_rowid,),
    ).fetchone()
    first_ordinal = int(row["ordinal"]) + 1
    conn.executemany(
        """
        INSERT INTO messages(session_rowid, ordinal, role, timestamp, text)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                session_rowid,
                first_ordinal + index,
                turn.role,
                turn.timestamp,
                turn.text,
            )
            for index, turn in enumerate(turns)
        ],
    )
    conn.execute(
        """
        UPDATE sessions
        SET path = ?, mtime = ?, size_kb = ?, title = ?, fingerprint = ?,
            file_identity = ?, indexed_bytes = ?
        WHERE id = ?
        """,
        (
            str(session.path),
            session.mtime,
            session.size_kb,
            session.title,
            state.fingerprint,
            state.identity,
            indexed_bytes,
            session_rowid,
        ),
    )


def _remove_missing_sessions(
    conn: sqlite3.Connection,
    *,
    visibility: int,
    source: str,
    current_ids: set[str],
) -> None:
    """Delete cached sessions no longer present in a file-backed source."""
    rows = conn.execute(
        "SELECT session_id FROM sessions WHERE visibility = ? AND source = ?",
        (visibility, source),
    )
    for row in rows:
        session_id = str(row["session_id"])
        if session_id not in current_ids:
            conn.execute(
                """
                DELETE FROM sessions
                WHERE visibility = ? AND source = ? AND session_id = ?
                """,
                (visibility, source, session_id),
            )


def _remove_missing_source_files(
    conn: sqlite3.Connection,
    *,
    source: str,
    current_ids: set[str],
) -> None:
    """Delete metadata for source files that are no longer present."""
    rows = conn.execute(
        "SELECT session_id FROM source_files WHERE source = ?",
        (source,),
    )
    for row in rows:
        session_id = str(row["session_id"])
        if session_id not in current_ids:
            conn.execute(
                "DELETE FROM source_files WHERE source = ? AND session_id = ?",
                (source, session_id),
            )


def _refresh_file_source(
    conn: sqlite3.Connection,
    *,
    source: str,
    include_subagents: bool,
) -> None:
    """Refresh changed Codex or Claude JSONL sessions only."""
    visibility = int(include_subagents)
    candidates = _discover_source_files(source)
    cached_metadata = _cached_file_metadata(conn, source=source)
    existing = {
        str(row["session_id"]): row
        for row in conn.execute(
            """
            SELECT id, session_id, fingerprint, file_identity, indexed_bytes
            FROM sessions
            WHERE visibility = ? AND source = ?
            """,
            (visibility, source),
        )
    }

    current_file_ids: set[str] = set()
    current_session_ids: set[str] = set()
    for candidate in candidates:
        indexed_file = _prepare_source_file(
            conn,
            candidate=candidate,
            cached=cached_metadata.get(candidate.session_id),
        )
        if indexed_file is None:
            continue
        current_file_ids.add(indexed_file.session.id)
        if source == "codex" and indexed_file.is_subagent and not include_subagents:
            continue

        session = indexed_file.session
        state = indexed_file.state
        current_session_ids.add(session.id)
        indexed = existing.get(session.id)
        if indexed is not None and str(indexed["fingerprint"]) == state.fingerprint:
            continue
        if (
            indexed is not None
            and str(indexed["file_identity"]) == state.identity
            and state.size > int(indexed["indexed_bytes"])
        ):
            session_rowid = int(indexed["id"])
            turns, indexed_bytes = _read_file_turns_from(
                session,
                include_subagents=include_subagents,
                offset=int(indexed["indexed_bytes"]),
                last_assistant_text=_last_assistant_text(conn, session_rowid),
            )
            _append_session(
                conn,
                session_rowid=session_rowid,
                session=session,
                state=state,
                indexed_bytes=indexed_bytes,
                turns=turns,
            )
            continue

        turns, indexed_bytes = _read_file_turns_from(
            session,
            include_subagents=include_subagents,
            offset=0,
            last_assistant_text=None,
        )
        _replace_session(
            conn,
            visibility=visibility,
            session=session,
            fingerprint=state.fingerprint,
            file_identity=state.identity,
            indexed_bytes=indexed_bytes,
            turns=turns,
        )

    _remove_missing_sessions(
        conn,
        visibility=visibility,
        source=source,
        current_ids=current_session_ids,
    )
    _remove_missing_source_files(
        conn,
        source=source,
        current_ids=current_file_ids,
    )


def _side_session_meta(
    thread_id: str,
    turns: list[Turn],
    *,
    include_subagents: bool,
) -> SessionMeta | None:
    """Build side-chat metadata without rescanning the shared log database."""
    user_turns = [
        turn
        for turn in turns
        if turn.role == "USER"
        and (include_subagents or not codex_side._is_helper_prompt(turn.text))
    ]
    if not user_turns:
        return None
    mtime = _timestamp_value(user_turns[-1].timestamp)
    if mtime is None:
        mtime = codex_side.CODEX_LOGS_DB.stat().st_mtime
    size_kb = sum(len(turn.text.encode()) for turn in turns) // 1024
    return SessionMeta(
        source=codex_side.CODEX_SIDE_SOURCE,
        id=thread_id,
        path=codex_side.CODEX_LOGS_DB,
        mtime=mtime,
        size_kb=size_kb,
        title=user_turns[0].text.replace("\n", " ")[:80],
    )


def _delete_side_session(
    conn: sqlite3.Connection,
    *,
    visibility: int,
    thread_id: str,
) -> None:
    """Delete one side-chat session from the derived index."""
    conn.execute(
        """
        DELETE FROM sessions
        WHERE visibility = ? AND source = ? AND session_id = ?
        """,
        (visibility, codex_side.CODEX_SIDE_SOURCE, thread_id),
    )


def _index_side_thread(
    conn: sqlite3.Connection,
    *,
    thread_id: str,
    include_subagents: bool,
    fingerprint: str,
    normal_thread_ids: set[str],
) -> None:
    """Re-read and replace one changed side-chat thread."""
    visibility = int(include_subagents)
    _delete_side_session(conn, visibility=visibility, thread_id=thread_id)
    if thread_id in normal_thread_ids:
        return
    turns = codex_side.read_turns(thread_id, raise_errors=True)
    session = _side_session_meta(
        thread_id,
        turns,
        include_subagents=include_subagents,
    )
    if session is None:
        return
    _replace_session(
        conn,
        visibility=visibility,
        session=session,
        fingerprint=fingerprint,
        file_identity=fingerprint,
        indexed_bytes=0,
        turns=turns,
    )


def _rebuild_side_source(
    conn: sqlite3.Connection,
    *,
    include_subagents: bool,
    identity: str,
) -> None:
    """Build the side-chat scope once before switching to row-id updates."""
    visibility = int(include_subagents)
    conn.execute(
        "DELETE FROM sessions WHERE visibility = ? AND source = ?",
        (visibility, codex_side.CODEX_SIDE_SOURCE),
    )
    for session in codex_side.list_sessions(
        include_subagents=include_subagents,
        raise_errors=True,
    ):
        turns = codex_side.read_turns(session.id, raise_errors=True)
        clean_meta = _side_session_meta(
            session.id,
            turns,
            include_subagents=include_subagents,
        )
        if clean_meta is not None:
            _replace_session(
                conn,
                visibility=visibility,
                session=clean_meta,
                fingerprint=identity,
                file_identity=identity,
                indexed_bytes=0,
                turns=turns,
            )


def _remove_normal_side_sessions(
    conn: sqlite3.Connection,
    *,
    visibility: int,
    normal_thread_ids: set[str],
) -> None:
    """Remove threads now represented by canonical Codex rollout sessions."""
    for thread_id in normal_thread_ids:
        _delete_side_session(conn, visibility=visibility, thread_id=thread_id)


def _refresh_side_source(
    conn: sqlite3.Connection,
    *,
    include_subagents: bool,
) -> None:
    """Refresh side chats by processing only log rows after the saved cursor."""
    visibility = int(include_subagents)
    identity_key = f"side:{visibility}:identity"
    row_key = f"side:{visibility}:last_row_id"
    identity = _database_identity(codex_side.CODEX_LOGS_DB)
    if identity is None:
        conn.execute(
            "DELETE FROM sessions WHERE visibility = ? AND source = ?",
            (visibility, codex_side.CODEX_SIDE_SOURCE),
        )
        return

    if _metadata_get(conn, identity_key) != identity:
        snapshot_row_id = codex_side.latest_log_row_id(raise_errors=True)
        _rebuild_side_source(
            conn,
            include_subagents=include_subagents,
            identity=identity,
        )
        _metadata_set(conn, identity_key, identity)
        _metadata_set(conn, row_key, str(snapshot_row_id))
        return

    previous_row_id = int(_metadata_get(conn, row_key) or 0)
    latest_row_id, changed_threads = codex_side.changed_thread_ids(
        previous_row_id,
        raise_errors=True,
    )
    if latest_row_id < previous_row_id:
        snapshot_row_id = latest_row_id
        _rebuild_side_source(
            conn,
            include_subagents=include_subagents,
            identity=identity,
        )
        _metadata_set(conn, row_key, str(snapshot_row_id))
        return

    normal_thread_ids = codex_side.normal_codex_thread_ids(raise_errors=True)
    _remove_normal_side_sessions(
        conn,
        visibility=visibility,
        normal_thread_ids=normal_thread_ids,
    )
    for thread_id in sorted(changed_threads):
        _index_side_thread(
            conn,
            thread_id=thread_id,
            include_subagents=include_subagents,
            fingerprint=f"{identity}:{latest_row_id}",
            normal_thread_ids=normal_thread_ids,
        )
    _metadata_set(conn, row_key, str(latest_row_id))


def refresh_index(
    conn: sqlite3.Connection,
    *,
    sources: tuple[str, ...],
    include_subagents: bool,
) -> None:
    """Refresh only the sources included by the current query."""
    if "codex" in sources:
        _refresh_file_source(
            conn,
            source="codex",
            include_subagents=include_subagents,
        )
    if codex_side.CODEX_SIDE_SOURCE in sources:
        _refresh_side_source(conn, include_subagents=include_subagents)
    if "claude" in sources:
        _refresh_file_source(
            conn,
            source="claude",
            include_subagents=include_subagents,
        )

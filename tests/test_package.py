"""Tests for agent-chat-reader CLI and parsing logic."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

import agent_chat_reader
from agent_chat_reader import claude, codex, codex_side, search
from agent_chat_reader.claude import _extract_user_text
from agent_chat_reader.cli import _json_session, main
from agent_chat_reader.codex import _apply_tail, _session_id_from_path, read_turns
from agent_chat_reader.models import Turn
from agent_chat_reader.output import elapsed_seconds_between, fmt_elapsed, print_turn


@pytest.fixture(autouse=True)
def _isolate_search_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every test's derived search index inside its temporary directory."""
    monkeypatch.setenv("AGENT_CHAT_READER_CACHE_DIR", str(tmp_path / "cache"))


# ── Version ───────────────────────────────────────────────────────────────────


def test_version() -> None:
    """Package version is a non-empty string."""
    assert isinstance(agent_chat_reader.__version__, str)
    assert agent_chat_reader.__version__


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    """--version prints the package version."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert agent_chat_reader.__version__ in capsys.readouterr().out


# ── Codex parsing ─────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as JSONL."""
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


_SESSION_FILE = "rollout-2026-01-01T00-00-00-019eae7d-da44-7413-8eb7-52d87219b1d3.jsonl"


def test_codex_read_turns_basic(tmp_path: Path) -> None:
    """Extracts user and agent turns from a minimal Codex session."""
    session = tmp_path / _SESSION_FILE
    _write_jsonl(
        session,
        [
            {"type": "session_meta", "payload": {"thread_source": "user"}},
            {
                "type": "event_msg",
                "timestamp": "2026-01-01T00:00:00Z",
                "payload": {"type": "user_message", "message": "hello"},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-01-01T00:00:01Z",
                "payload": {"type": "agent_message", "message": "hi there"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-01-01T00:00:02Z",
                "payload": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                },
            },
        ],
    )
    turns = read_turns(session)
    assert turns[0] == Turn("USER", "hello", "2026-01-01T00:00:00Z")
    assert turns[1] == Turn("AGENT", "hi there", "2026-01-01T00:00:01Z")
    assert turns[2] == Turn("AGENT", "ok", "2026-01-01T00:00:02Z")


def test_codex_deduplicates_agent_and_response_item(tmp_path: Path) -> None:
    """agent_message and response_item with same text are deduplicated."""
    session = tmp_path / _SESSION_FILE
    _write_jsonl(
        session,
        [
            {
                "type": "event_msg",
                "timestamp": "t1",
                "payload": {"type": "user_message", "message": "hi"},
            },
            {
                "type": "event_msg",
                "timestamp": "t2",
                "payload": {"type": "agent_message", "message": "same text"},
            },
            {
                "type": "response_item",
                "timestamp": "t3",
                "payload": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "same text"}],
                },
            },
        ],
    )
    turns = read_turns(session)
    agent_turns = [t for t in turns if t.role == "AGENT"]
    assert len(agent_turns) == 1
    assert agent_turns[0].text == "same text"


def test_codex_skips_empty_messages(tmp_path: Path) -> None:
    """Empty user/agent messages are not emitted as turns."""
    session = tmp_path / _SESSION_FILE
    _write_jsonl(
        session,
        [
            {"type": "event_msg", "payload": {"type": "user_message", "message": "  "}},
            {"type": "event_msg", "payload": {"type": "agent_message", "message": ""}},
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "real"},
            },
        ],
    )
    turns = read_turns(session)
    assert len(turns) == 1
    assert turns[0].text == "real"


def test_codex_session_id_from_path() -> None:
    """UUID is correctly extracted from rollout filename."""
    p = Path("rollout-2026-01-01T00-00-00-019eae7d-da44-7413-8eb7-52d87219b1d3.jsonl")
    assert _session_id_from_path(p) == "019eae7d-da44-7413-8eb7-52d87219b1d3"


def test_apply_tail_returns_last_n_user_turns() -> None:
    """tail=2 returns turns starting from the second-to-last USER turn."""
    turns = [
        Turn("USER", "msg1", ""),
        Turn("AGENT", "resp1", ""),
        Turn("USER", "msg2", ""),
        Turn("AGENT", "resp2", ""),
        Turn("USER", "msg3", ""),
        Turn("AGENT", "resp3", ""),
    ]
    result = _apply_tail(turns, tail=2)
    user_texts = [t.text for t in result if t.role == "USER"]
    assert user_texts == ["msg2", "msg3"]


def test_apply_tail_none_returns_all() -> None:
    """tail=None returns all turns unchanged."""
    turns = [Turn("USER", "x", ""), Turn("AGENT", "y", "")]
    assert _apply_tail(turns, tail=None) == turns


def test_codex_byte_cursor_waits_for_complete_appended_record(tmp_path: Path) -> None:
    """Incremental reads never advance past a partially written JSONL record."""
    session = tmp_path / _SESSION_FILE
    _write_search_session(session, user_text="first", agent_text="reply")
    _turns, offset = codex.read_turns_from(session, offset=0)
    partial = '{"type":"event_msg","payload":{"type":"user_message","message":"second"}'
    with session.open("a") as fh:
        fh.write(partial)

    turns, unchanged_offset = codex.read_turns_from(session, offset=offset)
    assert turns == []
    assert unchanged_offset == offset

    with session.open("a") as fh:
        fh.write("}\n")
    turns, final_offset = codex.read_turns_from(session, offset=offset)
    assert [turn.text for turn in turns] == ["second"]
    assert final_offset == session.stat().st_size


# ── Codex side-chat parsing ──────────────────────────────────────────────────


def _create_logs_db(path: Path) -> None:
    """Create a minimal Codex logs database fixture."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                ts_nanos INTEGER NOT NULL,
                level TEXT NOT NULL,
                target TEXT NOT NULL,
                feedback_log_body TEXT,
                module_path TEXT,
                file TEXT,
                line INTEGER,
                thread_id TEXT,
                process_uuid TEXT,
                estimated_bytes INTEGER NOT NULL DEFAULT 0
            )
            """
        )


def _create_state_db(path: Path, thread_ids: list[str]) -> None:
    """Create a minimal Codex state database fixture."""
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE threads (id TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO threads (id) VALUES (?)", [(i,) for i in thread_ids]
        )


def _debug_escape(text: str) -> str:
    """Escape text for the Rust-debug log snippets used in fixtures."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def _submission_body(text: str, *, submission_id: str = "sub-1") -> str:
    """Return a minimal user-submission log body."""
    return (
        "session_loop{thread_id=side-thread}: Submission sub=Submission "
        f'{{ id: "{submission_id}", op: UserInput {{ items: '
        f'[Text {{ text: "{_debug_escape(text)}", text_elements: [] }}] }} }}'
    )


def _assistant_body(text: str, *, message_id: str = "msg-1") -> str:
    """Return a minimal assistant-message log body."""
    return (
        "session_loop{thread_id=side-thread}:"
        "receiving_stream:handle_responses:"
        f'handle_output_item_done: Output item item=Message {{ id: Some("{message_id}"), '
        f'role: "assistant", content: [OutputText {{ text: "{_debug_escape(text)}" }}] }}'
    )


def _insert_log(
    path: Path,
    *,
    row_id: int,
    thread_id: str,
    target: str,
    body: str,
    ts: int = 1_767_225_600,
    ts_nanos: int = 0,
) -> None:
    """Insert one fixture log row."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO logs (
                id, ts, ts_nanos, level, target, feedback_log_body,
                thread_id, estimated_bytes
            )
            VALUES (?, ?, ?, 'DEBUG', ?, ?, ?, 0)
            """,
            (row_id, ts, ts_nanos, target, body, thread_id),
        )


def _patch_codex_side_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    normal_thread_ids: list[str] | None = None,
) -> tuple[Path, Path]:
    """Point codex_side at temp SQLite fixtures."""
    logs_db = tmp_path / "logs_2.sqlite"
    state_db = tmp_path / "state_5.sqlite"
    _create_logs_db(logs_db)
    _create_state_db(state_db, normal_thread_ids or [])
    monkeypatch.setattr(codex_side, "CODEX_LOGS_DB", logs_db)
    monkeypatch.setattr(codex_side, "CODEX_STATE_DB", state_db)
    return logs_db, state_db


def _patch_all_search_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Point every searchable source at empty temporary fixtures."""
    codex_sessions = tmp_path / "codex_sessions"
    claude_projects = tmp_path / "claude_projects"
    codex_sessions.mkdir()
    claude_projects.mkdir()
    monkeypatch.setattr(codex, "CODEX_SESSIONS", codex_sessions)
    monkeypatch.setattr(claude, "CLAUDE_PROJECTS", claude_projects)
    logs_db, _state_db = _patch_codex_side_paths(monkeypatch, tmp_path)
    return codex_sessions, logs_db


def _codex_session_path(root: Path, session_id: str) -> Path:
    """Return a valid rollout filename for one test session id."""
    return root / f"rollout-2026-01-01T00-00-00-{session_id}.jsonl"


def _write_search_session(
    path: Path,
    *,
    user_text: str,
    agent_text: str,
) -> None:
    """Write one minimal searchable Codex conversation."""
    _write_jsonl(
        path,
        [
            {"type": "session_meta", "payload": {"thread_source": "user"}},
            {
                "type": "event_msg",
                "timestamp": "2026-01-01T00:00:00Z",
                "payload": {"type": "user_message", "message": user_text},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-01-01T00:00:01Z",
                "payload": {"type": "agent_message", "message": agent_text},
            },
        ],
    )


def test_codex_side_lists_human_side_chats_and_filters_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Side-chat listing excludes normal threads and helper prompts by default."""
    logs_db, _state_db = _patch_codex_side_paths(
        monkeypatch,
        tmp_path,
        normal_thread_ids=["normal-thread"],
    )
    _insert_log(
        logs_db,
        row_id=1,
        thread_id="side-thread",
        target="codex_core::session::handlers",
        body=_submission_body("real side chat"),
    )
    _insert_log(
        logs_db,
        row_id=2,
        thread_id="normal-thread",
        target="codex_core::session::handlers",
        body=_submission_body("normal Codex session"),
    )
    _insert_log(
        logs_db,
        row_id=3,
        thread_id="helper-thread",
        target="codex_core::session::handlers",
        body=_submission_body(
            "You are a helpful assistant. You will be presented with a user prompt"
        ),
    )
    _insert_log(
        logs_db,
        row_id=4,
        thread_id="suggestion-helper-thread",
        target="codex_core::session::handlers",
        body=_submission_body(
            "# Overview\n\nGenerate 0 to 3 hyperpersonalized suggestions"
        ),
    )

    sessions = codex_side.list_sessions()
    assert [s.id for s in sessions] == ["side-thread"]
    assert sessions[0].source == "codex-side"
    assert sessions[0].title == "real side chat"

    sessions_with_helpers = codex_side.list_sessions(include_subagents=True)
    assert {s.id for s in sessions_with_helpers} == {
        "side-thread",
        "helper-thread",
        "suggestion-helper-thread",
    }


def test_codex_side_read_turns_reconstructs_user_and_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Side-chat reads reconstruct clean user and assistant turns."""
    logs_db, _state_db = _patch_codex_side_paths(monkeypatch, tmp_path)
    _insert_log(
        logs_db,
        row_id=1,
        thread_id="side-thread",
        target="codex_core::session::handlers",
        body=_submission_body("hello\nside chat"),
    )
    _insert_log(
        logs_db,
        row_id=2,
        thread_id="side-thread",
        target="codex_core::stream_events_utils",
        body=_assistant_body('hi "there"', message_id="msg-1"),
        ts_nanos=1,
    )
    _insert_log(
        logs_db,
        row_id=3,
        thread_id="side-thread",
        target="codex_core::stream_events_utils",
        body=_assistant_body('hi "there"', message_id="msg-1"),
        ts_nanos=2,
    )

    turns = codex_side.read_turns("side-thread")
    assert turns == [
        Turn("USER", "hello\nside chat", "2026-01-01T00:00:00Z"),
        Turn("AGENT", 'hi "there"', "2026-01-01T00:00:00Z"),
    ]


def test_cli_find_can_search_codex_side_chats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI search includes side chats when filtered to codex-side."""
    logs_db, _state_db = _patch_codex_side_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(codex, "CODEX_SESSIONS", tmp_path / "codex_sessions")
    _insert_log(
        logs_db,
        row_id=1,
        thread_id="side-thread",
        target="codex_core::session::handlers",
        body=_submission_body("blue12345red12345blue"),
    )

    assert main(["--find", "blue12345red12345blue", "--source", "codex-side"]) == 0
    out = capsys.readouterr().out
    assert "[CODEX-SIDE] side-thread" in out
    assert "blue12345red12345blue" in out


def test_cli_read_can_emit_codex_side_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI read locates side-chat ids and emits structured JSON."""
    logs_db, _state_db = _patch_codex_side_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(codex, "CODEX_SESSIONS", tmp_path / "codex_sessions")
    _insert_log(
        logs_db,
        row_id=1,
        thread_id="side-thread",
        target="codex_core::session::handlers",
        body=_submission_body("read me"),
    )

    assert main(["side-thread", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "codex-side"
    assert payload["session_id"] == "side-thread"
    assert payload["turns"][0]["text"] == "read me"


# ── Indexed search ────────────────────────────────────────────────────────────


def test_search_matches_terms_across_turns_and_centers_snippets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AND applies to a session and snippets expose late message matches."""
    codex_sessions, _logs_db = _patch_all_search_sources(monkeypatch, tmp_path)
    session_id = "019eae7d-da44-7413-8eb7-52d87219b1d3"
    path = _codex_session_path(codex_sessions, session_id)
    _write_search_session(
        path,
        user_text="collision geometry is the first topic",
        agent_text=f"{'prefix ' * 40}friction metadata is the second topic",
    )
    with path.open("a") as fh:
        for index in range(5):
            fh.write(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": f"2026-01-01T00:00:0{index + 2}Z",
                        "payload": {
                            "type": "agent_message",
                            "message": f"another friction match {index}",
                        },
                    }
                )
                + "\n"
            )

    results = search.search_sessions(
        ["collision", "friction"],
        sources=("codex",),
        include_subagents=False,
        limit=40,
        since=None,
    )
    visible_snippets = " ".join(hit[1] for hit in results[0].hits[:4])
    assert "collision" in visible_snippets
    assert "friction" in visible_snippets

    assert main(["--find", "collision", "friction", "--source", "codex"]) == 0
    out = capsys.readouterr().out
    assert session_id in out
    assert "collision" in out
    assert "friction" in out


def test_search_keeps_distinct_sessions_with_equal_titles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Equal title text never merges distinct session identities."""
    codex_sessions, _logs_db = _patch_all_search_sources(monkeypatch, tmp_path)
    first_id = "019eae7d-da44-7413-8eb7-52d87219b1d3"
    second_id = "029eae7d-da44-7413-8eb7-52d87219b1d4"
    for session_id in (first_id, second_id):
        _write_search_session(
            _codex_session_path(codex_sessions, session_id),
            user_text="identical opening title",
            agent_text="a separate conversation",
        )

    assert main(["--find", "identical opening", "--source", "codex"]) == 0
    out = capsys.readouterr().out
    assert first_id in out
    assert second_id in out


def test_search_preserves_case_insensitive_literal_substrings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The FTS path preserves legacy substring behavior without query syntax."""
    codex_sessions, _logs_db = _patch_all_search_sources(monkeypatch, tmp_path)
    _write_search_session(
        _codex_session_path(
            codex_sessions,
            "019eae7d-da44-7413-8eb7-52d87219b1d3",
        ),
        user_text='Policy_Interface says "Hello"',
        agent_text="reply",
    )
    results = search.search_sessions(
        ["LICY_INT", '"Hello"'],
        sources=("codex",),
        include_subagents=False,
        limit=40,
        since=None,
    )
    assert len(results) == 1


def test_search_reuses_unchanged_file_index_and_refreshes_changed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchanged JSONL sessions are not reparsed, while appended text is indexed."""
    codex_sessions, _logs_db = _patch_all_search_sources(monkeypatch, tmp_path)
    path = _codex_session_path(
        codex_sessions,
        "019eae7d-da44-7413-8eb7-52d87219b1d3",
    )
    _write_search_session(path, user_text="alpha marker", agent_text="initial reply")
    real_read_turns_from = codex.read_turns_from
    read_count = 0

    def counting_read_turns_from(
        path: Path,
        *,
        offset: int,
        last_assistant_text: str | None = None,
    ) -> tuple[list[Turn], int]:
        """Count source parses while preserving the production parser."""
        nonlocal read_count
        read_count += 1
        return real_read_turns_from(
            path,
            offset=offset,
            last_assistant_text=last_assistant_text,
        )

    monkeypatch.setattr(codex, "read_turns_from", counting_read_turns_from)
    options = {
        "sources": ("codex",),
        "include_subagents": False,
        "limit": 40,
        "since": None,
    }
    assert search.search_sessions(["alpha"], **options)
    assert search.search_sessions(["alpha"], **options)
    assert read_count == 1

    with path.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": "2026-01-01T00:00:02Z",
                    "payload": {"type": "user_message", "message": "beta marker"},
                }
            )
            + "\n"
        )
    assert search.search_sessions(["beta"], **options)
    assert read_count == 2


def test_side_search_uses_log_cursor_after_initial_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Side-chat refreshes rebuild changed threads without relisting all logs."""
    _codex_sessions, logs_db = _patch_all_search_sources(monkeypatch, tmp_path)
    _insert_log(
        logs_db,
        row_id=1,
        thread_id="side-thread",
        target="codex_core::session::handlers",
        body=_submission_body("alpha side marker"),
    )
    real_list_sessions = codex_side.list_sessions
    list_count = 0

    def counting_list_sessions(
        *,
        include_subagents: bool = False,
        raise_errors: bool = False,
    ) -> list:
        """Count expensive full side-chat discovery calls."""
        nonlocal list_count
        list_count += 1
        return real_list_sessions(
            include_subagents=include_subagents,
            raise_errors=raise_errors,
        )

    monkeypatch.setattr(codex_side, "list_sessions", counting_list_sessions)
    options = {
        "sources": ("codex-side",),
        "include_subagents": False,
        "limit": 40,
        "since": None,
    }
    assert search.search_sessions(["alpha"], **options)
    _insert_log(
        logs_db,
        row_id=2,
        thread_id="side-thread",
        target="codex_core::stream_events_utils",
        body=_assistant_body("beta side marker"),
        ts_nanos=1,
    )
    assert search.search_sessions(["beta"], **options)
    assert list_count == 1


def test_side_build_does_not_skip_rows_appended_during_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The saved cursor predates discovery so concurrent rows remain pending."""
    _codex_sessions, logs_db = _patch_all_search_sources(monkeypatch, tmp_path)
    _insert_log(
        logs_db,
        row_id=1,
        thread_id="first-thread",
        target="codex_core::session::handlers",
        body=_submission_body("first marker", submission_id="sub-1"),
    )
    real_list_sessions = codex_side.list_sessions
    inserted = False

    def list_then_append(
        *,
        include_subagents: bool = False,
        raise_errors: bool = False,
    ) -> list:
        """Simulate a new side chat arriving after the discovery snapshot."""
        nonlocal inserted
        sessions = real_list_sessions(
            include_subagents=include_subagents,
            raise_errors=raise_errors,
        )
        if not inserted:
            inserted = True
            _insert_log(
                logs_db,
                row_id=2,
                thread_id="second-thread",
                target="codex_core::session::handlers",
                body=_submission_body("second marker", submission_id="sub-2"),
                ts_nanos=1,
            )
        return sessions

    monkeypatch.setattr(codex_side, "list_sessions", list_then_append)
    options = {
        "sources": ("codex-side",),
        "include_subagents": False,
        "limit": 40,
        "since": None,
    }
    assert search.search_sessions(["first"], **options)
    results = search.search_sessions(["second"], **options)
    assert [result.session.id for result in results] == ["second-thread"]


def test_search_limit_and_since_filter_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Search honors the shared result limit and explicit activity date."""
    codex_sessions, _logs_db = _patch_all_search_sources(monkeypatch, tmp_path)
    old_id = "019eae7d-da44-7413-8eb7-52d87219b1d3"
    new_id = "029eae7d-da44-7413-8eb7-52d87219b1d4"
    old_path = _codex_session_path(codex_sessions, old_id)
    new_path = _codex_session_path(codex_sessions, new_id)
    for path in (old_path, new_path):
        _write_search_session(path, user_text="dated marker", agent_text="reply")
    os.utime(old_path, (1_767_225_600, 1_767_225_600))
    os.utime(new_path, (1_767_398_400, 1_767_398_400))

    assert (
        main(
            [
                "--find",
                "dated marker",
                "--source",
                "codex",
                "--since",
                "2026-01-02",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert new_id in out
    assert old_id not in out


def test_deleted_search_cache_rebuilds_automatically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The derived index remains safe to delete and recreate from sources."""
    codex_sessions, _logs_db = _patch_all_search_sources(monkeypatch, tmp_path)
    _write_search_session(
        _codex_session_path(
            codex_sessions,
            "019eae7d-da44-7413-8eb7-52d87219b1d3",
        ),
        user_text="rebuild marker",
        agent_text="reply",
    )
    options = {
        "sources": ("codex",),
        "include_subagents": False,
        "limit": 40,
        "since": None,
    }
    assert search.search_sessions(["rebuild"], **options)
    index_path = tmp_path / "cache" / "agent-chat-reader" / "search.sqlite3"
    index_path.unlink()
    assert search.search_sessions(["rebuild"], **options)


# ── Output formatting ────────────────────────────────────────────────────────


def test_print_turn_shows_timestamp_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default transcript view includes timing context."""
    print_turn(Turn("USER", "hello", "2026-01-01T12:00:00Z"))
    out = capsys.readouterr().out
    assert "[USER]" in out
    assert "2026-01" in out


def test_print_turn_can_hide_timestamp(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Timestamp display can be disabled for quieter transcript reads."""
    print_turn(
        Turn("USER", "hello", "2026-01-01T00:00:00Z"),
        show_timestamps=False,
    )
    out = capsys.readouterr().out
    assert "[USER]" in out
    assert "2026-01" not in out


def test_print_turn_can_show_agent_elapsed_time(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Timestamp mode shows elapsed time for the current speaker."""
    previous_turn = Turn("USER", "hello", "2026-01-01T00:00:00Z")
    print_turn(
        Turn("AGENT", "hi", "2026-01-01T00:02:05Z"),
        show_timestamps=True,
        previous_turn=previous_turn,
    )
    out = capsys.readouterr().out
    assert "[AGENT]" in out
    assert "(agent took 2m)" in out


def test_print_turn_can_show_user_elapsed_time(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """User messages show elapsed time without inferring idle state."""
    previous_turn = Turn("AGENT", "question?", "2026-01-01T00:00:00Z")
    print_turn(
        Turn("USER", "answer", "2026-01-01T00:30:00Z"),
        show_timestamps=True,
        previous_turn=previous_turn,
    )
    assert "(user took 30m)" in capsys.readouterr().out


def test_fmt_elapsed_handles_long_gaps() -> None:
    """Elapsed durations stay compact for long sessions."""
    from datetime import timedelta

    assert fmt_elapsed(timedelta(days=1, hours=2, minutes=3)) == "1d 2h"


def test_elapsed_seconds_between_returns_machine_readable_gap() -> None:
    """Elapsed gap calculations are available without parsing display text."""
    previous_turn = Turn("USER", "hello", "2026-01-01T00:00:00Z")
    turn = Turn("AGENT", "hi", "2026-01-01T00:02:05Z")
    assert elapsed_seconds_between(previous_turn, turn) == 125


def test_json_session_includes_structured_timing_fields(tmp_path: Path) -> None:
    """JSON output exposes stable fields for agents and scripts."""
    session = tmp_path / _SESSION_FILE
    turns = [
        Turn("USER", "hello", "2026-01-01T00:00:00Z"),
        Turn("AGENT", "hi", "2026-01-01T00:02:05Z"),
    ]
    payload = json.loads(
        _json_session(source="codex", path=session, size_kb=12, turns=turns)
    )
    assert payload["source"] == "codex"
    assert payload["size_kb"] == 12
    assert payload["total_turns"] == 2
    assert payload["turns"][0]["elapsed_seconds"] is None
    assert payload["turns"][1]["elapsed_seconds"] == 125
    assert payload["turns"][1]["text"] == "hi"


# ── Claude parsing ────────────────────────────────────────────────────────────


def test_claude_extract_user_text_string() -> None:
    """Plain string content is returned as-is."""
    assert _extract_user_text("hello") == "hello"


def test_claude_extract_user_text_empty_string() -> None:
    """Empty string returns None."""
    assert _extract_user_text("") is None


def test_claude_extract_user_text_tool_result_returns_none() -> None:
    """Content that is a tool-result carrier returns None."""
    content = [{"type": "tool_result", "tool_use_id": "x", "content": "output"}]
    assert _extract_user_text(content) is None


def test_claude_extract_user_text_text_block() -> None:
    """Content with a text block returns the text."""
    content = [{"type": "text", "text": "real question"}]
    assert _extract_user_text(content) == "real question"


# ── CLI smoke ─────────────────────────────────────────────────────────────────


def test_cli_no_args_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """Calling main() with no args prints help and exits 0."""
    assert main([]) == 0
    assert "agent-chat-reader" in capsys.readouterr().out

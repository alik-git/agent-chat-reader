"""Tests for agent-chat-reader CLI and parsing logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_chat_reader
from agent_chat_reader.claude import _extract_user_text
from agent_chat_reader.cli import _json_session, main
from agent_chat_reader.codex import _apply_tail, _session_id_from_path, read_turns
from agent_chat_reader.models import Turn
from agent_chat_reader.output import elapsed_seconds_between, fmt_elapsed, print_turn

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

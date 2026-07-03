"""Codex side-chat reader backed by the local runtime log database."""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, cast

from agent_chat_reader.models import SessionMeta, Turn

CODEX_HOME = Path.home() / ".codex"
CODEX_LOGS_DB = CODEX_HOME / "logs_2.sqlite"
CODEX_STATE_DB = CODEX_HOME / "state_5.sqlite"
CODEX_SIDE_SOURCE = "codex-side"

_USER_SUBMISSION_TARGET = "codex_core::session::handlers"
_ASSISTANT_MESSAGE_TARGET = "codex_core::stream_events_utils"
_USER_SUBMISSION_MARKER = "Submission sub=Submission"
_USER_INPUT_MARKER = "op: UserInput"
_ASSISTANT_MESSAGE_MARKER = ":handle_output_item_done: Output item item=Message"

_SUBMISSION_ID_RE = re.compile(r'Submission \{ id: "([^"]+)"')
_MESSAGE_ID_RE = re.compile(r'Output item item=Message \{ id: Some\("([^"]+)"\)')

_HELPER_PROMPT_PREFIXES = (
    "You are a helpful assistant. You will be presented with a user prompt",
    "The following is the Codex agent history whose request action you are assessing",
    (
        "The following is the Codex agent history added since your last "
        "approval assessment"
    ),
    "You are an expert at upholding safety",
    "## Memory Writing Agent",
    "# Overview\n\nGenerate 0 to 3 hyperpersonalized suggestions",
)


class _LogRow(NamedTuple):
    """The log row fields needed to reconstruct visible side-chat turns."""

    row_id: int
    ts: int
    ts_nanos: int
    target: str
    body: str


def list_sessions(*, include_subagents: bool = False) -> list[SessionMeta]:
    """List Codex side chats found in the runtime log database.

    Side chats are not normal rollout JSONL sessions. The desktop app currently
    records them in `logs_2.sqlite`, so this reader reconstructs a clean session
    list from user-submission log rows and filters out known internal helper
    sessions by default.
    """
    if not CODEX_LOGS_DB.exists():
        return []

    normal_thread_ids = _normal_codex_thread_ids()
    latest_mtimes: dict[str, float] = {}
    titles: dict[str, str] = {}

    try:
        with _connect_readonly(CODEX_LOGS_DB) as conn:
            for row in conn.execute(
                """
                SELECT id, ts, ts_nanos, thread_id, feedback_log_body
                FROM logs
                WHERE thread_id IS NOT NULL
                  AND thread_id != ''
                  AND target = ?
                  AND feedback_log_body LIKE ?
                  AND feedback_log_body LIKE ?
                ORDER BY ts DESC, ts_nanos DESC, id DESC
                """,
                (
                    _USER_SUBMISSION_TARGET,
                    f"%{_USER_SUBMISSION_MARKER}%",
                    f"%{_USER_INPUT_MARKER}%",
                ),
            ):
                thread_id = cast(str, row["thread_id"])
                if thread_id in normal_thread_ids:
                    continue

                text = _extract_user_submission_text(
                    cast(str, row["feedback_log_body"])
                )
                if not text:
                    continue
                if not include_subagents and _is_helper_prompt(text):
                    continue

                if thread_id not in latest_mtimes:
                    latest_mtimes[thread_id] = _mtime(
                        cast(int, row["ts"]),
                        cast(int, row["ts_nanos"]),
                    )
                # Rows are newest-first, so the last title assigned for a thread
                # is the earliest visible prompt we saw for that side chat.
                titles[thread_id] = text.replace("\n", " ")[:80]
    except sqlite3.Error:
        return []

    size_kb = CODEX_LOGS_DB.stat().st_size // 1024
    sessions = [
        SessionMeta(
            source=CODEX_SIDE_SOURCE,
            id=thread_id,
            path=CODEX_LOGS_DB,
            mtime=mtime,
            size_kb=size_kb,
            title=titles.get(thread_id, ""),
        )
        for thread_id, mtime in latest_mtimes.items()
    ]
    return sorted(sessions, key=lambda s: s.mtime, reverse=True)


def find_session(
    session_id: str,
    *,
    include_subagents: bool = False,
) -> SessionMeta | None:
    """Find a side-chat session by thread-id fragment."""
    matches = [
        session
        for session in list_sessions(include_subagents=include_subagents)
        if session_id in session.id
    ]
    return matches[0] if matches else None


def read_turns(thread_id: str, *, tail: int | None = None) -> list[Turn]:
    """Read visible turns from a Codex side chat thread."""
    if not CODEX_LOGS_DB.exists():
        return []

    turns: list[Turn] = []
    seen_user_submissions: set[str] = set()
    seen_assistant_messages: set[str] = set()
    last_assistant_text: str | None = None

    try:
        with _connect_readonly(CODEX_LOGS_DB) as conn:
            for row in conn.execute(
                """
                SELECT id, ts, ts_nanos, target, feedback_log_body
                FROM logs
                WHERE thread_id = ?
                ORDER BY ts ASC, ts_nanos ASC, id ASC
                """,
                (thread_id,),
            ):
                log_row = _log_row(row)

                if log_row.target == _USER_SUBMISSION_TARGET:
                    submission_id = _extract_submission_id(log_row.body)
                    if submission_id in seen_user_submissions:
                        continue
                    text = _extract_user_submission_text(log_row.body)
                    if not text:
                        continue
                    seen_user_submissions.add(submission_id)
                    turns.append(Turn("USER", text, _timestamp(log_row)))
                    last_assistant_text = None

                elif log_row.target == _ASSISTANT_MESSAGE_TARGET:
                    message_id = _extract_message_id(log_row.body)
                    if message_id in seen_assistant_messages:
                        continue
                    text = _extract_assistant_message_text(log_row.body)
                    if not text or text == last_assistant_text:
                        continue
                    seen_assistant_messages.add(message_id)
                    turns.append(Turn("AGENT", text, _timestamp(log_row)))
                    last_assistant_text = text
    except sqlite3.Error:
        return []

    return _apply_tail(turns, tail)


def _connect_readonly(path: Path) -> sqlite3.Connection:
    """Open a SQLite database without creating missing files."""
    uri = f"{path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _normal_codex_thread_ids() -> set[str]:
    """Return thread ids already represented by normal Codex rollout sessions."""
    if not CODEX_STATE_DB.exists():
        return set()
    try:
        with _connect_readonly(CODEX_STATE_DB) as conn:
            return {
                cast(str, row["id"]) for row in conn.execute("SELECT id FROM threads")
            }
    except sqlite3.Error:
        return set()


def _log_row(row: sqlite3.Row) -> _LogRow:
    """Convert a SQLite row into typed fields."""
    return _LogRow(
        row_id=cast(int, row["id"]),
        ts=cast(int, row["ts"]),
        ts_nanos=cast(int, row["ts_nanos"]),
        target=cast(str, row["target"]),
        body=cast(str, row["feedback_log_body"] or ""),
    )


def _extract_submission_id(body: str) -> str:
    """Extract a user-submission id, falling back to the body hash."""
    match = _SUBMISSION_ID_RE.search(body)
    return match.group(1) if match else str(hash(body))


def _extract_message_id(body: str) -> str:
    """Extract an assistant-message id, falling back to the body hash."""
    match = _MESSAGE_ID_RE.search(body)
    return match.group(1) if match else str(hash(body))


def _extract_user_submission_text(body: str) -> str | None:
    """Extract user text chunks from a Codex `Submission` debug log row."""
    if _USER_SUBMISSION_MARKER not in body or _USER_INPUT_MARKER not in body:
        return None
    parts = _extract_debug_strings(body, "Text { text:")
    text = "".join(parts).strip()
    return text or None


def _extract_assistant_message_text(body: str) -> str | None:
    """Extract assistant text from a Codex `Output item item=Message` log row."""
    if _ASSISTANT_MESSAGE_MARKER not in body or 'role: "assistant"' not in body:
        return None
    parts = _extract_debug_strings(body, "OutputText { text:")
    text = "\n".join(part.strip() for part in parts if part.strip()).strip()
    return text or None


def _extract_debug_strings(body: str, marker: str) -> list[str]:
    """Extract Rust-debug quoted strings that follow a repeated marker."""
    values: list[str] = []
    start = 0
    while True:
        marker_index = body.find(marker, start)
        if marker_index == -1:
            return values
        quote_index = body.find('"', marker_index + len(marker))
        if quote_index == -1:
            return values
        value, next_index = _parse_debug_string(body, quote_index)
        if value is not None:
            values.append(value)
        start = max(next_index, quote_index + 1)


def _parse_debug_string(body: str, quote_index: int) -> tuple[str | None, int]:
    """Parse one Rust-debug quoted string and common escapes."""
    if quote_index >= len(body) or body[quote_index] != '"':
        return None, quote_index

    chars: list[str] = []
    i = quote_index + 1
    while i < len(body):
        char = body[i]
        if char == '"':
            return "".join(chars), i + 1
        if char == "\\":
            decoded, i = _decode_escape(body, i)
            chars.append(decoded)
            continue
        chars.append(char)
        i += 1

    return None, i


def _decode_escape(body: str, slash_index: int) -> tuple[str, int]:
    """Decode a single escape sequence in a Rust-debug string."""
    if slash_index + 1 >= len(body):
        return "\\", slash_index + 1

    escaped = body[slash_index + 1]
    if escaped == "n":
        return "\n", slash_index + 2
    if escaped == "r":
        return "\r", slash_index + 2
    if escaped == "t":
        return "\t", slash_index + 2
    if escaped in {'"', "\\"}:
        return escaped, slash_index + 2
    if escaped == "u" and body.startswith("\\u{", slash_index):
        end = body.find("}", slash_index + 3)
        if end != -1:
            try:
                return chr(int(body[slash_index + 3 : end], 16)), end + 1
            except ValueError:
                pass

    return "\\" + escaped, slash_index + 2


def _is_helper_prompt(text: str) -> bool:
    """Return True for Codex-internal prompts hidden from normal chat history."""
    stripped = text.lstrip()
    return any(stripped.startswith(prefix) for prefix in _HELPER_PROMPT_PREFIXES)


def _timestamp(row: _LogRow) -> str:
    """Return an ISO timestamp for a log row."""
    return (
        datetime.fromtimestamp(_mtime(row.ts, row.ts_nanos), UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _mtime(ts: int, ts_nanos: int) -> float:
    """Return a Unix timestamp with nanosecond fraction."""
    return ts + ts_nanos / 1_000_000_000


def _apply_tail(turns: list[Turn], tail: int | None) -> list[Turn]:
    """Trim to the last N user-message turns and everything after."""
    if tail is None:
        return turns
    user_indices = [i for i, turn in enumerate(turns) if turn.role == "USER"]
    if tail >= len(user_indices):
        return turns
    return turns[user_indices[-tail] :]

# agent-chat-reader

Read and search [Codex CLI](https://github.com/openai/codex) and [Claude Code](https://claude.ai/code) chat history from the terminal.

Useful for AI agents that need to recall what was discussed or decided in past sessions, without manually parsing noisy JSONL files.

## Install

```bash
uv tool install agent-chat-reader
```

Or for development:

```bash
git clone https://github.com/alik-git/agent-chat-reader
cd agent-chat-reader
uv sync --extra dev
uv run agent-chat-reader --help
```

## Usage

**List recent sessions** (both Codex and Claude Code, sorted by recency):

```bash
agent-chat-reader --list
```

**Search across all sessions** for a keyword:

```bash
agent-chat-reader --find "sim2sim"
agent-chat-reader --find "policy_interface" --source codex
agent-chat-reader --find "side chat phrase" --source codex-side
agent-chat-reader --find collision friction --since 2026-07-01 --limit 10
```

Separate search arguments use AND logic across the whole session, so the words
may occur in different messages. A quoted shell argument remains one literal
phrase. Results show text around each match and keep distinct session IDs even
when titles are identical.

The first search builds a disposable local SQLite index from the clean turns.
Later searches refresh only appended JSONL records and new side-chat log rows.
Delete `~/.cache/agent-chat-reader/search.sqlite3` at any time to rebuild it from
the source histories.

**Read a specific session** by UUID prefix:

```bash
agent-chat-reader 019eaecb
agent-chat-reader 1bfc739b --verbose     # include tool call summaries
agent-chat-reader 019eaecb --tail 5      # last 5 user turns only
agent-chat-reader 019eaecb --hide-timestamps
agent-chat-reader 019eaecb --format json
```

## What it filters out

The raw JSONL files are very noisy. This tool extracts only:

- **Codex**: `user_message` events, `agent_message` events, and full `response_item` assistant text. Guardian/subagent sessions (auto-approval bots) are hidden by default.
- **Codex side chats**: user submissions and assistant text reconstructed from the local Codex runtime log database. Known internal helper sessions are hidden by default.
- **Claude Code**: real user turns (not tool-result carriers), and assistant text blocks (not thinking blocks or tool calls). Sidechain sub-agent turns are hidden by default.

Use `--include-subagents` to see everything.

## Options

| Flag | Description |
|------|-------------|
| `--list` / `-l` | List recent sessions from both sources |
| `--find KEYWORD...` / `-f` | Search sessions; separate arguments use session-wide AND |
| `--source codex\|codex-side\|claude` | Filter to one source. `codex` includes normal Codex sessions and side chats |
| `--verbose` / `-v` | Include brief tool call summaries (Claude sessions) |
| `--tail N` / `-n N` | Show only the last N user turns of a session |
| `--include-subagents` | Include guardian/subagent sessions |
| `--format text\|json` | Output format for session reads |
| `--hide-timestamps` | Hide message timestamps and elapsed-gap labels |
| `--limit N` | Max sessions shown by `--list` or `--find` (default: 40) |
| `--since DATE` | Search sessions active on or after an ISO date/timestamp |

Session reads show timestamps by default. Elapsed labels use the current
speaker's role, such as `(agent took 8s)` or `(user took 4m)`, and only report
the time since the previous visible message.

Use `--format json` when another agent or script needs stable structured
fields instead of terminal separators. JSON turns include `timestamp`,
`local_time`, `elapsed_seconds`, `role`, and `text`.

## Session storage locations

| Agent | Path |
|-------|------|
| Codex CLI | `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` |
| Codex side chats | `~/.codex/logs_2.sqlite` |
| Claude Code | `~/.claude/projects/*/*.jsonl` |

The derived search index defaults to
`~/.cache/agent-chat-reader/search.sqlite3`. Set
`AGENT_CHAT_READER_CACHE_DIR` to place it under a different cache root.

Codex side chats are log-backed rather than normal rollout transcripts, so the
tool labels them as `codex-side` in list and search output.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

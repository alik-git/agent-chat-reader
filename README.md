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
```

**Read a specific session** by UUID prefix:

```bash
agent-chat-reader 019eaecb
agent-chat-reader 1bfc739b --verbose     # include tool call summaries
agent-chat-reader 019eaecb --tail 5      # last 5 user turns only
```

## What it filters out

The raw JSONL files are very noisy. This tool extracts only:

- **Codex**: `user_message` events, `agent_message` events, and full `response_item` assistant text. Guardian/subagent sessions (auto-approval bots) are hidden by default.
- **Claude Code**: real user turns (not tool-result carriers), and assistant text blocks (not thinking blocks or tool calls). Sidechain sub-agent turns are hidden by default.

Use `--include-subagents` to see everything.

## Options

| Flag | Description |
|------|-------------|
| `--list` / `-l` | List recent sessions from both sources |
| `--find KEYWORD` / `-f` | Search all sessions for a keyword |
| `--source codex\|claude` | Filter to one source |
| `--verbose` / `-v` | Include brief tool call summaries (Claude sessions) |
| `--tail N` / `-n N` | Show only the last N user turns of a session |
| `--include-subagents` | Include guardian/subagent sessions |
| `--limit N` | Max sessions shown by `--list` (default: 40) |

## Session storage locations

| Agent | Path |
|-------|------|
| Codex CLI | `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` |
| Claude Code | `~/.claude/projects/*/*.jsonl` |

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

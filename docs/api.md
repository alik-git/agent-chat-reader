# API

`agent-chat-reader` currently treats its command-line interface as the public
API. The Python modules under `agent_chat_reader` are implementation details and
may evolve with the local Codex and Claude history formats.

Use `agent-chat-reader --format json <session>` when another agent or script
needs stable structured transcript data.

# Letta agent adapter (agent-managed memory)

`letta-agent` subclasses [`letta`](letta.md) but lets the agent decide what to
remember. Its cost profile is different: every write is an LLM step.

```bash
pip install membench[letta]
membench run --backend letta-agent --pack mini --backend-config configs/default/letta-agent.yaml
```

| Contract | Letta call |
|---|---|
| `write` | `agents.messages.create(agent_id, messages=[{"role": "user", "content": "[<ts>] <role>: <content>"}])`; the agent edits its core blocks / archival memory with its own memory tools |
| `recall` | same as `letta`: core blocks + archival search by `at` |
| `forget` | a user message asking the agent to erase the target event's content from core and archival memory |
| `reset` | deletes the agent |

- Recalled passages are the agent's own notes, so items carry no `Event.id`:
  `evidence_recall_at_k` is not meaningful for this backend (it reads 0).
- Forget compliance depends on the agent following the instruction; that is
  what the metric measures here.
- `usage()` sums `prompt_tokens` / `completion_tokens` from each response's
  `usage`, priced against `model`. `uses_llm` is true, so runs default to 3
  repeats.

Config keys are the same as `letta`. Live check:
`MEMBENCH_LIVE=1 pytest "tests/test_live_backends.py::test_live_mini_runs_clean[letta-agent]"`
or `gh workflow run live.yml -f backend=letta-agent`.

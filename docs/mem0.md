# Mem0 adapter

Install the optional extra:

```bash
pip install membench[mem0]
```

Run a pack against Mem0 OSS:

```bash
membench run --backend mem0 --pack customer-support --out report.json
```

Config is read from `--backend-config <file>` (a JSON or YAML mapping) and
passed to `Mem0Adapter`. It supports:

- `user_id` — Mem0 user/session scope (default: `"membench"`).
- `top_k` — number of memories returned per `recall()` (default: `3`).
- `mem0_config` — optional dict forwarded to `Memory.from_config(...)`
  (vector store / LLM / embedder settings); omit to use Mem0's defaults.

Each run starts clean: `membench run` calls `delete_all(user_id=...)` before
replaying the pack, so use a dedicated `user_id` for evals. Pass `--no-reset`
to keep existing memories.

Example `mem0.yaml` (Mem0's default LLM/embedder are OpenAI and read
`OPENAI_API_KEY`):

```yaml
user_id: membench-eval
top_k: 3
mem0_config:
  llm:
    provider: openai
    config:
      model: gpt-4o-mini
  embedder:
    provider: openai
    config:
      model: text-embedding-3-small
```

Run with an OpenAI judge:

```bash
export OPENAI_API_KEY=sk-...               # used by Mem0
export MEMBENCH_JUDGE_API_KEY=$OPENAI_API_KEY
membench run --backend mem0 --pack customer-support \
  --backend-config mem0.yaml \
  --judge-base-url https://api.openai.com/v1 --judge-model gpt-4o-mini \
  --out report.json
```

If `mem0ai` isn't installed, `membench run --backend mem0 ...` fails with
a clear error asking you to `pip install membench[mem0]` instead of
crashing with an `ImportError` traceback.

# Mem0 adapter

Install the optional extra:

```bash
pip install membench[mem0]
```

Run a pack against Mem0 OSS:

```bash
membench run --backend mem0 --pack customer-support --out report.json
```

Config (passed to `Mem0Adapter` via `--backend mem0`) supports:

- `user_id` — Mem0 user/session scope (default: `"membench"`).
- `top_k` — number of memories returned per `recall()` (default: `3`).
- `mem0_config` — optional dict forwarded to `Memory.from_config(...)`
  (vector store / LLM / embedder settings); omit to use Mem0's defaults.

If `mem0ai` isn't installed, `membench run --backend mem0 ...` fails with
a clear error asking you to `pip install membench[mem0]` instead of
crashing with an `ImportError` traceback.

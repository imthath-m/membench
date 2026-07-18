# Transcript scenario generator

Turn a transcript into a schema-valid scenario pack:

```bash
membench generate --transcripts transcript.txt --out pack.yaml
```

`--transcripts` accepts a single file or a directory of files (concatenated
in sorted order). It reuses the judge's provider-agnostic client config
(`--judge-model`, `--judge-base-url`, `MEMBENCH_JUDGE_API_KEY`) from
`core.scoring.judge` — no separate generator client or API key scheme.

**Generation needs an API key** (`MEMBENCH_JUDGE_API_KEY`); without one,
`membench generate` fails with an actionable error instead of a crash.

**Running the generated pack does not need a key.** The output is validated
against `packs/schema.py` before it's written, so it loads and runs as-is:

```bash
membench run --backend full-context --pack pack.yaml --no-judge --out report.json
```

`--no-judge` skips the semantic-recall/contradiction-rate metrics (they
score as `null`); the four deterministic metrics still run fully offline.

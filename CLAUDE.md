# Agent instructions

- Start by reading [docs/HANDOFF.md](docs/HANDOFF.md): state, PR stack, keys & setup, open-items runbook.
- Then read [docs/progress.md](docs/progress.md) ("Overall status" and the latest section) and resume from the first unchecked item.
- Follow the conventions in HANDOFF.md "Instructions for the next agent": stacked PRs, progress logging, checks before every push (`ruff check .`, `pytest`, `uv lock --check`, results rescore + leaderboard diff).
- Never commit dataset files; never fake results, labels or costs.

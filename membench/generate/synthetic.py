"""Synthetic generator v2: deterministic, seeded, template-based packs (no LLM).

A `Domain` (see `generate/domains.py`) holds fact templates; `generate`
builds one episode per simulated user, each with the same query mix:

  - `single_hop`: a fact stated once.
  - `multi_hop`: two facts in different sessions that must be joined.
  - `update`: a fact revised along a chain A→B→C (`chain_depth`); the old
    values are `expect_absent` (staleness).
  - `temporal`: the value a revised fact had at an earlier date.
  - `abstention`: a fact never stated (`abstain: true`).
  - `forget`: a fact the user asked to delete (a `forget` step), then asked about.

Every answerable query carries a `semantic_recall` judge rubric and every
`update` query a `contradiction` rubric, so the judge metrics (and their
validation sample, Phase 15) have hundreds of items per pack.

Facts are spread over `sessions` sessions `gap_days` apart, mixed with
`distractors` filler turns per session; every query comes after the last
session. Output is a raw pack mapping (validated by `parse_pack`), identical
byte for byte for a given (domain, seed, knobs): the only randomness is a
`random.Random` seeded from the domain name and seed.

`paraphrase_queries` is the optional LLM pass that rewrites each question to
share fewer words with the turn that answers it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from membench.generate.domains import DOMAINS, Domain

GENERATOR_VERSION = "2"
_EPOCH = datetime(2025, 1, 6, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Knobs:
    episodes: int = 36
    sessions: int = 4
    distractors: int = 4  # filler turns per session
    chain_depth: int = 3  # values in the update chain (A→B→C)
    gap_days: tuple[int, int] = (2, 30)  # days between sessions (min ≥ 2 for temporal)
    multi_hop: bool = True
    abstention: bool = True
    forget: bool = True


def _day(d: datetime) -> str:
    return f"{d:%B} {d.day}, {d.year}"


class _Episode:
    def __init__(self, e: int, rng: random.Random, domain: Domain, knobs: Knobs) -> None:
        self.e, self.rng, self.domain, self.knobs = e, rng, domain, knobs
        self.n = 0
        self.turns: list[list[dict[str, Any]]] = [[] for _ in range(knobs.sessions)]

    def fact(self, session: int, content: str, key: str, fact_type: str = "state") -> str:
        self.n += 1
        event_id = f"e{self.e}-w{self.n}"
        self.turns[session].append({
            "type": "write", "id": event_id,
            "fact_key": f"e{self.e}-{key}", "role": self.domain.fact_role, "content": content,
            "fact_type": fact_type,
        })
        return event_id


def _episode(e: int, rng: random.Random, domain: Domain, knobs: Knobs) -> dict[str, Any]:
    ep = _Episode(e, rng, domain, knobs)
    single, chain, past, gone, unknown = rng.sample(domain.attributes, 5)
    queries: list[dict[str, Any]] = []
    later = knobs.sessions - 1

    value = rng.choice(single.values)
    event = ep.fact(rng.randrange(knobs.sessions), single.state.format(v=value), single.key)
    queries.append({"category": "single_hop", "query": single.question, "expect_contains": [value],
                    "expect_key": f"e{e}-{single.key}", "answer": value, "evidence": [event]})

    if knobs.multi_hop:
        link = rng.choice(domain.links)
        entity, value = rng.choice(link.entities), rng.choice(link.values)
        first, second = sorted(rng.sample(range(knobs.sessions), 2))
        ids = [ep.fact(first, link.first.format(e=entity), f"{link.key}-1"),
               ep.fact(second, link.second.format(e=entity, v=value), f"{link.key}-2")]
        queries.append({"category": "multi_hop", "query": link.question,
                        "expect_contains": [value], "answer": value, "evidence": ids})

    values = rng.sample(chain.values, knobs.chain_depth)
    sessions = sorted(rng.sample(range(knobs.sessions), knobs.chain_depth))
    ids = [ep.fact(s, (chain.state if i == 0 else chain.update).format(v=v), chain.key,
                   "state" if i == 0 else "update")
           for i, (s, v) in enumerate(zip(sessions, values))]
    contradiction = (f"The evidence names more than one current {chain.label} (among "
                     f"{', '.join(values)}) without indicating which one superseded the others.")
    queries.append({"category": "update", "query": chain.question,
                    "expect_contains": [values[-1]], "expect_key": f"e{e}-{chain.key}",
                    "expect_absent": values[:-1], "answer": values[-1], "evidence": [ids[-1]],
                    "judge": {"contradiction": contradiction}})

    old, new = rng.sample(past.values, 2)
    before = rng.randrange(knobs.sessions - 1)
    past_ids = [ep.fact(before, past.state.format(v=old), past.key),
                ep.fact(rng.randrange(before + 1, knobs.sessions), past.update.format(v=new),
                        past.key, "update")]
    temporal = {"category": "temporal", "expect_contains": [old], "answer": old,
                "evidence": [past_ids[0]], "_asof": before}
    queries.append(temporal)

    if knobs.abstention:
        queries.append({"category": "abstention", "query": unknown.question, "abstain": True})

    forget_target = None
    if knobs.forget:
        value = rng.choice(gone.values)
        forget_target = ep.fact(rng.randrange(later), gone.state.format(v=value), gone.key)
        ep.fact(later, domain.forget_request.format(what=gone.label), f"{gone.key}-erase")
        queries.append({"category": "forget", "query": gone.question, "abstain": True,
                        "expect_absent": [value]})

    for turns in ep.turns:  # filler turns at random positions, facts keep their order
        for role, text in rng.sample(domain.distractors, knobs.distractors):
            ep.n += 1
            turns.insert(rng.randrange(len(turns) + 1), {
                "type": "write", "id": f"e{e}-w{ep.n}", "fact_key": f"e{e}-noise-{ep.n}", "role": role,
                "content": text,
            })

    steps: list[dict[str, Any]] = []
    day = _EPOCH + timedelta(days=rng.randrange(365), hours=rng.randrange(8, 18))
    session_days = []
    for s, turns in enumerate(ep.turns):
        if s:
            day += timedelta(days=rng.randint(*knobs.gap_days))
        session_days.append(day)
        for t, turn in enumerate(turns):
            steps.append({"type": "write", "session_id": f"e{e}-s{s}", **turn,
                          "timestamp": (day + timedelta(minutes=t)).isoformat()})
    if forget_target is not None:
        steps.append({"type": "forget", "session_id": f"e{e}-s{later}", "target": forget_target,
                      "timestamp": (day + timedelta(minutes=len(ep.turns[later]))).isoformat()})
    asked = day + timedelta(days=1)
    temporal["query"] = past.past_question.format(
        date=_day(session_days[temporal.pop("_asof")] + timedelta(days=1))
    )
    for q, query in enumerate(queries):
        if "answer" in query:  # judge rubric for semantic_recall
            query.setdefault("judge", {})["semantic_recall"] = (
                f"The evidence indicates that the answer to {query['query']!r} is {query['answer']}."
            )
        steps.append({"type": "query", "session_id": f"e{e}-q",
                      "timestamp": (asked + timedelta(minutes=q)).isoformat(), **query})
    return {"steps": steps}


def generate(name: str, seed: int = 0, knobs: Knobs | None = None) -> dict[str, Any]:
    """Raw pack mapping for the named domain; deterministic per (name, seed, knobs)."""
    domain = DOMAINS[name]
    knobs = knobs or Knobs()
    if not 2 <= knobs.chain_depth <= knobs.sessions or knobs.gap_days[0] < 2:
        raise ValueError("need 2 <= chain_depth <= sessions and gap_days min >= 2")
    rng = random.Random(f"{name}:{seed}")
    return {
        "name": name,
        "description": f"{domain.description} Synthetic v{GENERATOR_VERSION}, seed {seed}.",
        "episodes": [_episode(e, rng, domain, knobs) for e in range(knobs.episodes)],
    }


def paraphrase_queries(raw: dict[str, Any], client: Any) -> dict[str, Any]:
    """Rewrite every query with an LLM (`client.complete(prompt) -> str`), in place.

    Asks for the same question with as few words as possible shared with the
    turns it is answered by (its `evidence`), so keyword overlap alone can't
    find the needle. Abstention / forget queries are rephrased without a needle.
    """
    for episode in raw["episodes"]:
        content = {s["id"]: s["content"] for s in episode["steps"] if s["type"] == "write"}
        for step in episode["steps"]:
            if step["type"] != "query":
                continue
            needles = "\n".join(content[i] for i in step.get("evidence", [])) or "(none)"
            reply = client.complete(
                "Rewrite the QUESTION so it asks for exactly the same information but "
                "shares as few words as possible with the TEXT. Keep names, dates and "
                "numbers. Reply with only the rewritten question.\n\n"
                f"QUESTION: {step['query']}\n\nTEXT:\n{needles}\n"
            )
            step["query"] = reply.strip().splitlines()[0].strip() or step["query"]
    return raw

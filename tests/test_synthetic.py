"""Synthetic generator v2: size, category balance, determinism, structure."""

import re
from collections import Counter
from datetime import datetime, timezone

import pytest
import yaml

from membench.adapters.full_context import FullContextAdapter
from membench.core.runner import run_pack
from membench.datasets import SYNTHETIC
from membench.generate.domains import DOMAINS
from membench.generate.synthetic import Knobs, generate, paraphrase_queries
from membench.packs.schema import ForgetStep, QueryStep, WriteStep, parse_pack

CATEGORIES = {"single_hop", "multi_hop", "update", "temporal", "abstention", "forget"}


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_pack_has_200_queries_and_30_per_category(name):
    pack = parse_pack(generate(name))
    counts = Counter(s.category for s in pack.steps if isinstance(s, QueryStep))
    assert set(counts) == CATEGORIES
    assert sum(counts.values()) >= 200 and min(counts.values()) >= 30


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_generation_is_byte_identical_per_seed(name):
    dump = lambda seed: yaml.safe_dump(generate(name, seed=seed), sort_keys=False)
    assert dump(7) == dump(7)
    assert dump(7) != dump(8)


def test_synthetic_names_match_domains():
    assert {n.replace("-", "_") for n in SYNTHETIC} == set(DOMAINS)


def test_episode_structure():
    pack = parse_pack(generate("customer_support_xl", knobs=Knobs(episodes=1)))
    writes = {s.id: s for s in pack.steps if isinstance(s, WriteStep)}
    queries = {s.category: s for s in pack.steps if isinstance(s, QueryStep)}

    update = queries["update"]
    chain = [w for w in writes.values() if w.fact_key == update.expect_key]
    assert [w.fact_type for w in chain] == ["state", "update", "update"]  # depth 3
    assert len(update.expect_absent) == 2
    assert all(v in c.content for v, c in zip(update.expect_absent, chain))
    assert update.expect_contains[0] in writes[update.evidence[0]].content

    hop = queries["multi_hop"]
    assert len({writes[i].session_id for i in hop.evidence}) == 2

    temporal = queries["temporal"]
    old = writes[temporal.evidence[0]]
    newer = next(w for w in writes.values() if w.fact_key == old.fact_key and w.id != old.id)
    assert temporal.expect_contains[0] in old.content
    when = re.search(r"\w+ \d+, \d{4}", temporal.query)[0]
    asked = datetime.strptime(when, "%B %d, %Y").replace(tzinfo=timezone.utc).date()
    assert datetime.fromisoformat(old.timestamp).date() < asked
    assert asked < datetime.fromisoformat(newer.timestamp).date()

    assert queries["abstention"].abstain
    forget = [s for s in pack.steps if isinstance(s, ForgetStep)]
    assert len(forget) == 1 and queries["forget"].abstain
    assert queries["forget"].expect_absent[0] in writes[forget[0].target[0]].content

    stamps = [datetime.fromisoformat(s.timestamp) for s in pack.steps]
    assert stamps == sorted(stamps)


def test_knobs_shape_the_pack():
    knobs = Knobs(episodes=2, sessions=5, distractors=2, chain_depth=4, multi_hop=False,
                  abstention=False, forget=False)
    pack = parse_pack(generate("document_chain", knobs=knobs))
    counts = Counter(s.category for s in pack.steps if isinstance(s, QueryStep))
    assert counts == {"single_hop": 2, "update": 2, "temporal": 2}
    assert len({s.session_id for s in pack.steps if isinstance(s, WriteStep)}) == 10
    with pytest.raises(ValueError):
        generate("document_chain", knobs=Knobs(chain_depth=5))


def test_full_context_recalls_everything_but_is_stale():
    pack = parse_pack(generate("tool_call_memory", knobs=Knobs(episodes=3)))
    scores = run_pack(pack, FullContextAdapter({}), "full-context").scores
    assert scores["recall_precision"] == 1.0
    assert scores["staleness"] == 1.0  # every superseded value is still in context
    assert scores["forget_compliance"] == 1.0


class EchoClient:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return "  Rephrased question?\nextra line"


def test_paraphrase_rewrites_queries_with_their_evidence():
    raw = generate("multi_step_decisions", knobs=Knobs(episodes=1))
    client = EchoClient()
    paraphrase_queries(raw, client)
    queries = [s for s in raw["episodes"][0]["steps"] if s["type"] == "query"]
    assert {q["query"] for q in queries} == {"Rephrased question?"}
    assert len(client.prompts) == len(queries)
    hop = next(p for p, q in zip(client.prompts, queries) if q["category"] == "multi_hop")
    assert hop.count("\n", hop.index("TEXT:")) >= 2  # both evidence turns
    parse_pack(raw)


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_judge_rubrics_on_answerable_and_update_queries(name):
    queries = [s for s in parse_pack(generate(name)).steps if isinstance(s, QueryStep)]
    for q in queries:
        assert bool(q.judge and q.judge.semantic_recall) == (q.answer is not None)
        assert bool(q.judge and q.judge.contradiction) == (q.category == "update")
    assert sum(q.answer is not None for q in queries) >= 100

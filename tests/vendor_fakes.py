"""In-memory stand-ins for the hosted / LLM-backed vendor SDKs (no network).

Each fake mirrors the SDK surface its adapter calls (checked against the real
packages: mem0ai 2.2, zep-cloud 4.1, letta-client 1.12, langmem 0.0.30,
cognee 1.6) and keeps the semantics the contract suite relies on: writes are
scoped to a user / graph / agent / dataset, asynchronous backends only expose
a write once it has been processed (so a missing `flush()` fails), and
deletes remove data. Search ranks stored texts by word overlap with the query.
"""

from __future__ import annotations

import itertools
import re
import uuid
from types import SimpleNamespace

_ids = itertools.count()


def _rank(query: str, texts: list[tuple[object, str]], k: int) -> list[tuple[object, str, float]]:
    words = set(re.findall(r"\w+", query.lower()))
    scored = [(obj, text, float(len(words & set(re.findall(r"\w+", text.lower())))))
              for obj, text in texts]
    return sorted(scored, key=lambda s: -s[2])[:k]


# --- Mem0 Platform (mem0.MemoryClient) ---------------------------------------

class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> _Response:
        return self

    def json(self) -> dict:
        return self._payload


class FakeMemoryClient:
    """Async adds: a memory exists only after its event has been polled to SUCCEEDED."""

    def __init__(self, api_key=None, host=None) -> None:
        self.host = host or "https://api.mem0.ai"
        self.memories: list[dict] = []
        self.events: dict[str, dict] = {}
        self.client = SimpleNamespace(get=self._get)

    def add(self, messages, *, user_id=None, metadata=None, timestamp=None):
        assert isinstance(timestamp, int) and user_id
        event_id = f"evt-{next(_ids)}"
        memory = {"id": f"mem-{next(_ids)}", "memory": messages[-1]["content"],
                  "user_id": user_id, "metadata": metadata}
        self.events[event_id] = {"polls_left": 1, "memory": memory}
        return {"event_id": event_id, "status": "PENDING"}

    def _get(self, path: str) -> _Response:
        event = self.events[path.strip("/").split("/")[-1]]
        if event["polls_left"]:
            event["polls_left"] -= 1
            return _Response({"status": "RUNNING"})
        memory = event.pop("memory", None)
        if memory is not None:
            self.memories.append(memory)
        return _Response({"status": "SUCCEEDED"})

    def search(self, query, *, filters, top_k):
        mine = [(m, m["memory"]) for m in self.memories if m["user_id"] == filters["user_id"]]
        return {"results": [{**m, "score": s} for m, _, s in _rank(query, mine, top_k)]}

    def get_all(self, *, filters, page=1, page_size=100):
        mine = [m for m in self.memories if m["user_id"] == filters["user_id"]]
        chunk = mine[(page - 1) * page_size: page * page_size]
        return {"results": chunk, "next": "more" if len(mine) > page * page_size else None}

    def delete(self, memory_id):
        self.memories = [m for m in self.memories if m["id"] != memory_id]

    def delete_all(self, *, user_id):
        self.memories = [m for m in self.memories if m["user_id"] != user_id]


# --- Zep Cloud (zep_cloud.client.Zep) ----------------------------------------

class FakeZep:
    """Episodes become facts (edges) only once their task has been polled."""

    def __init__(self, api_key=None) -> None:
        self.users: set[str] = set()
        self.edges: list[SimpleNamespace] = []
        self.tasks: dict[str, dict] = {}
        self.user = SimpleNamespace(create=self._create_user, delete=self._delete_user)
        self.task = SimpleNamespace(get=self._get_task)
        self.graph = SimpleNamespace(
            episode=SimpleNamespace(add=self._add_episode, delete=self._delete_episode),
            search_edges=self._search_edges,
        )

    def _create_user(self, *, metadata=None):
        user = f"user-{next(_ids)}"
        self.users.add(user)
        return SimpleNamespace(uuid_=user, graph_uuid=f"graph-of-{user}")

    def _delete_user(self, user_uuid):
        self.users.discard(user_uuid)
        self.edges = [e for e in self.edges if e.graph != user_uuid]

    def _task(self, apply) -> SimpleNamespace:
        task_id = f"task-{next(_ids)}"
        self.tasks[task_id] = {"polls_left": 1, "apply": apply}
        return SimpleNamespace(uuid_=task_id)

    def _add_episode(self, graph_uuid, *, data, type, reference_time):
        assert graph_uuid in self.users and type == "message"
        episode = f"ep-{next(_ids)}"
        edge = SimpleNamespace(graph=graph_uuid, fact=data.split(": ", 1)[1],
                               episode_uuids=[episode], valid_at=reference_time,
                               invalid_at=None)
        task = self._task(lambda: self.edges.append(edge))
        return SimpleNamespace(episode=SimpleNamespace(uuid_=episode), task=task)

    def _delete_episode(self, graph_uuid, episode_uuid):
        def apply():
            self.edges = [e for e in self.edges if episode_uuid not in e.episode_uuids]
        return SimpleNamespace(task=self._task(apply))

    def _get_task(self, task_uuid):
        task = self.tasks[task_uuid]
        if task["polls_left"]:
            task["polls_left"] -= 1
            return SimpleNamespace(status="processing", error=None)
        apply = task.pop("apply", None)
        if apply is not None:
            apply()
        return SimpleNamespace(status="succeeded", error=None)

    def _search_edges(self, graph_uuid, *, query, page_size, filters):
        at = filters["date_filters"]["any_of"][0]["all_of"][0]["value"]
        mine = [(e, e.fact) for e in self.edges if e.graph == graph_uuid and e.valid_at <= at]
        items = [SimpleNamespace(**vars(e), score=s) for e, _, s in _rank(query, mine, page_size)]
        return SimpleNamespace(items=items)


# zep_cloud module stand-in: the adapter only needs EdgeSearchFilters.model_validate.
FAKE_ZEP_MODULE = SimpleNamespace(EdgeSearchFilters=SimpleNamespace(model_validate=lambda d: d))


# --- Letta (letta_client.Letta) ----------------------------------------------

class FakeLetta:
    """Agents with archival passages and core blocks. `messages.create` plays a
    tiny agent: it files each message as a passage and honours forget prompts."""

    def __init__(self, api_key=None, base_url=None) -> None:
        self.agents_state: dict[str, dict] = {}
        passages = SimpleNamespace(create=self._create_passage, search=self._search,
                                   delete=self._delete_passage)
        self.agents = SimpleNamespace(
            create=self._create_agent, delete=self._delete_agent, passages=passages,
            blocks=SimpleNamespace(list=lambda agent_id: self.agents_state[agent_id]["blocks"]),
            messages=SimpleNamespace(create=self._message),
        )

    def _create_agent(self, *, name, model, embedding, tags, memory_blocks):
        agent_id = f"agent-{next(_ids)}"
        blocks = [SimpleNamespace(**b) for b in memory_blocks]
        self.agents_state[agent_id] = {"name": name, "passages": [], "blocks": blocks}
        return SimpleNamespace(id=agent_id)

    def _delete_agent(self, agent_id):
        del self.agents_state[agent_id]

    def _create_passage(self, agent_id, *, text, created_at=None):
        passage = SimpleNamespace(id=f"passage-{next(_ids)}", content=text,
                                  created_at=created_at or "9999")
        self.agents_state[agent_id]["passages"].append(passage)
        return [passage]

    def _search(self, agent_id, *, query, top_k, end_datetime):
        mine = [(p, p.content) for p in self.agents_state[agent_id]["passages"]
                if p.created_at <= end_datetime]
        return SimpleNamespace(results=[p for p, _, _ in _rank(query, mine, top_k)])

    def _delete_passage(self, passage_id, *, agent_id):
        state = self.agents_state[agent_id]
        state["passages"] = [p for p in state["passages"] if p.id != passage_id]

    def _message(self, agent_id, *, messages):
        text = messages[-1]["content"]
        state = self.agents_state[agent_id]
        if text.startswith("Permanently forget"):
            content = text.split("\n", 1)[1]
            state["passages"] = [p for p in state["passages"] if content not in p.content]
        else:  # the "agent" archives the message under its own (untraceable) passage id
            stamp, note = text[1:].split("] ", 1)
            self._create_passage(agent_id, text=note, created_at=stamp)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))


# --- LangMem (langmem.create_memory_store_manager + langgraph store) ---------

class FakeStore:
    """`BaseStore` subset: namespaced items, query-ranked `search`, `delete`."""

    def __init__(self, *, index=None) -> None:
        self.index = index
        self.items: dict[tuple, dict[str, dict]] = {}

    def put(self, namespace, key, value):
        self.items.setdefault(tuple(namespace), {})[key] = value

    def search(self, namespace, *, query=None, limit=10):
        mine = [((k, v), v["content"]["content"]) for k, v in self.items.get(namespace, {}).items()]
        ranked = _rank(query or "", mine, limit)
        return [SimpleNamespace(key=k, value=v, score=s) for (k, v), _, s in ranked]

    def delete(self, namespace, key):
        self.items.get(tuple(namespace), {}).pop(key, None)


class FakeUsageHandler:
    def __init__(self) -> None:
        self.usage_metadata: dict[str, dict] = {}


def fake_memory_store_manager(model, *, namespace, store):
    """Stores each message verbatim as one memory (no LLM)."""

    def invoke(inputs, config):
        user = config["configurable"]["langgraph_user_id"]
        ns = tuple(part.replace("{langgraph_user_id}", user) for part in namespace)
        key = str(uuid.uuid4())
        value = {"kind": "Memory", "content": {"content": inputs["messages"][-1]["content"]}}
        store.put(ns, key, value)
        for handler in config["callbacks"]:
            meta = handler.usage_metadata.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
            meta["input_tokens"] += 50
            meta["output_tokens"] += 5
        return [{"namespace": ns, "key": key, "value": value}]

    return SimpleNamespace(invoke=invoke)


# --- Cognee (module-level async API) -----------------------------------------

class FakeCognee:
    """Datasets of data items; only cognified items are searchable (as chunks)."""

    def __init__(self) -> None:
        self.SearchType = {"CHUNKS": "CHUNKS", "GRAPH_COMPLETION": "GRAPH_COMPLETION"}
        self.data: dict[str, dict[str, dict]] = {}  # dataset -> data id -> item
        self.prune = SimpleNamespace(prune_data=self._prune_data, prune_system=self._prune_system)

    async def add(self, text, *, dataset_name):
        items = self.data.setdefault(dataset_name, {})
        for item in items.values():  # identical content dedups
            if item["text"] == text:
                return SimpleNamespace(data_ids=[item["uuid"]])
        data_uuid = uuid.uuid4()
        items[str(data_uuid)] = {"uuid": data_uuid, "text": text, "cognified": False}
        return SimpleNamespace(data_ids=[data_uuid])

    async def cognify(self, *, datasets):
        for name in datasets:
            for item in self.data.get(name, {}).values():
                item["cognified"] = True

    async def search(self, query, *, query_type, datasets, top_k, only_context):
        results = []
        for name in datasets:
            mine = [(key, item["text"]) for key, item in self.data.get(name, {}).items()
                    if item["cognified"]]
            ranked = _rank(query, mine, top_k)
            if query_type == "CHUNKS":
                found = [{"text": text, "document_id": key, "score": s} for key, text, s in ranked]
            else:
                found = "\n".join(text for _, text, _ in ranked)
            results.append({"dataset_name": name, "search_result": found})
        return results

    async def forget(self, *, data_id, dataset):
        self.data.get(dataset, {}).pop(str(data_id), None)

    async def _prune_data(self):
        self.data = {}

    async def _prune_system(self, metadata=False):
        assert metadata is True

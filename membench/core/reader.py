"""Reader stage: an LLM answers each query from the recalled context.

Optional (`membench run --reader-model`). The reader's answer is stored on
the query trace (`detail.reader_answer`) and judged against the pack's gold
`answer` (`qa_accuracy`) or checked for "I don't know" on `abstain` queries
(`abstention`). That makes results comparable to published QA numbers
(LongMemEval, LoCoMo), which score answers rather than retrieved context.
Configured like the judge (model, base URL, API key) and reuses its clients.
"""

from __future__ import annotations

from typing import Any

# Bump whenever `build_reader_prompt` changes; recorded in the run manifest.
READER_PROMPT_VERSION = "1"
READER_API_KEY_ENV = "MEMBENCH_READER_API_KEY"
ABSTAIN_ANSWER = "I don't know"


def build_reader_prompt(question: str, context: str, at: str) -> str:
    return (
        "You are an assistant with a long-term memory. Answer the QUESTION using "
        "only the MEMORY below. Be brief: one short sentence or phrase. If the "
        f'MEMORY does not contain the answer, reply exactly "{ABSTAIN_ANSWER}".\n\n'
        f"CURRENT DATE: {at}\n\nMEMORY:\n{context}\n\nQUESTION:\n{question}\n"
    )


class Reader:
    """Wraps a judge-style client (`respond(prompt) -> (text, usage)`)."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.model = client.model
        self.base_url = getattr(client, "base_url", None)
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def answer(self, question: str, context: str, at: str, style: str = "membench") -> str:
        """`style` "longmemeval" uses LongMemEval's official reader prompt."""
        if style == "longmemeval":
            from membench.datasets.longmemeval import reader_prompt

            prompt = reader_prompt(question, context, at)
        else:
            prompt = build_reader_prompt(question, context, at)
        self.usage["calls"] += 1
        text, usage = self.client.respond(prompt)
        for name in ("input_tokens", "output_tokens"):
            self.usage[name] += usage.get(name, 0)
        if not isinstance(text, str):
            raise TypeError(f"reader reply text is {type(text).__name__}, not str")
        return text.strip()

# Oracle baseline

An upper bound: recall returns exactly the query's gold `evidence` events, so
retrieval is perfect and QA accuracy measures the reader alone.

```bash
membench run --backend oracle --pack customer-support --reader-model claude-haiku-4-5
```

- The only adapter that sees ground truth, by design: it sets
  `sees_evidence = True`, and the runner then sets `adapter.evidence` to the
  query's `evidence` ids before each recall. Every other adapter keeps
  `sees_evidence = False` and never receives it.
- Context lines match `full-context` (`[<timestamp>] [<role>] <content>`), in
  evidence order. Forgotten events are never returned; a query with no
  `evidence` gets an empty context.
- Its scores are a ceiling, not a competitor: label it "oracle (upper bound)"
  wherever it is published.

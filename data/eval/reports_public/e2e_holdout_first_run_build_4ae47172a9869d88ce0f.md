> Public copy: local absolute paths were converted to repository-relative sources; metrics and per-question decisions are unchanged.

> Frozen holdout: this is the first formal run and must not be replaced after tuning against its failures.

# Engineering RAG evaluation

- Dataset: `engineering_holdout_v2.jsonl`
- Questions: 30
- Top-K: 5
- Baseline: `hybrid_live`
- Suite: `e2e`

| Strategy | Route acc. | Route macro-F1 | Primary Hit@K | Primary Recall@K | nDCG@K | Symbol Recall@K | MRR | Refusal F1 | False refusal | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid_live | 0.700 | 0.626 | 0.870 | 0.775 | 0.712 | 0.321 | 0.759 | 0.636 | 0.348 | 178.54 | 1425.26 |

## Interpretation boundaries

- Source metrics use explicit v2 primary/supporting labels and de-duplicate repeated chunks from the same source.
- Index ablation uses oracle routes, answerable questions only, and disables live rg/AST/Git plus answer generation.
- End-to-end evaluation fixes the hybrid index and evaluates routing, live verification, citations, and refusal separately.
- Symbol Recall@K uses explicit `relevant_symbols`; inspect the eligible sample count.
- Refusal is an explicit predictor decision. An empty retrieval result is not silently counted as a safe refusal.
- File/citation metrics are source-level, not passage entailment or answer correctness metrics.
- Latency is measured after one discarded warm-up call per strategy; model/index loading and report serialization are excluded.

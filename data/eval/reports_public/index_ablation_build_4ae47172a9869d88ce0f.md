> Public copy: local absolute paths were converted to repository-relative sources; metrics and per-question decisions are unchanged.

# Engineering RAG evaluation

- Dataset: `mini_nanobot_internal.jsonl + official_engineering_specs.jsonl`
- Questions: 70
- Top-K: 5
- Baseline: `bm25`
- Suite: `index`

| Strategy | Route acc. | Route macro-F1 | Primary Hit@K | Primary Recall@K | nDCG@K | Symbol Recall@K | MRR | Refusal F1 | False refusal | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | N/A | N/A | 0.629 | 0.629 | 0.488 | 0.338 | 0.472 | 0.000 | 0.000 | 3.81 | 5.53 |
| dense | N/A | N/A | 0.400 | 0.400 | 0.350 | 0.206 | 0.301 | 0.000 | 0.000 | 110.70 | 138.99 |
| hybrid | N/A | N/A | 0.457 | 0.457 | 0.385 | 0.333 | 0.323 | 0.000 | 0.000 | 114.87 | 159.17 |

## Interpretation boundaries

- Source metrics use explicit v2 primary/supporting labels and de-duplicate repeated chunks from the same source.
- Index ablation uses oracle routes, answerable questions only, and disables live rg/AST/Git plus answer generation.
- End-to-end evaluation fixes the hybrid index and evaluates routing, live verification, citations, and refusal separately.
- Symbol Recall@K uses explicit `relevant_symbols`; inspect the eligible sample count.
- Refusal is an explicit predictor decision. An empty retrieval result is not silently counted as a safe refusal.
- File/citation metrics are source-level, not passage entailment or answer correctness metrics.
- Latency is measured after one discarded warm-up call per strategy; model/index loading and report serialization are excluded.

# Experiment 1 -- retrieval comparison

Scored questions: **29** (answerable, with evidence). One question = 3.4 percentage points, so gaps of a few points are within noise.

UNSCORED (no evidence supplied): ids [12]

Unanswerable questions (no retrieval metric): 1

Sorted by any_hit@6, then MRR. **Sort order is not a significance test.** @6 is the number of chunks the LLM is shown.

| Model | Chunking | any@6 (95% CI) | full@6 | coverage@6 | source@6 | MRR | chunks | index s | ms/query |
|---|---|---|---|---|---|---|---|---|---|
| minilm | sliding_window | 100% (100%-100%) | 97% | 99% | 100% | 0.77 | 216 | 10.5 | 73 |
| bge-small | sliding_window | 97% (90%-100%) | 93% | 96% | 100% | 0.83 | 216 | 11.4 | 79 |
| nomic | sliding_window | 93% (83%-100%) | 86% | 91% | 100% | 0.84 | 216 | 85.8 | 236 |
| bge-small | paragraph | 93% (83%-100%) | 83% | 89% | 100% | 0.80 | 233 | 8.0 | 55 |
| minilm | paragraph | 93% (83%-100%) | 86% | 91% | 100% | 0.76 | 233 | 27.9 | 52 |
| nomic | fixed | 93% (83%-100%) | 90% | 91% | 100% | 0.75 | 176 | 71.6 | 218 |
| minilm | fixed | 93% (83%-100%) | 90% | 91% | 100% | 0.74 | 176 | 9.5 | 123 |
| nomic | paragraph | 90% (76%-100%) | 79% | 85% | 100% | 0.81 | 233 | 99.2 | 219 |
| bge-small | fixed | 90% (76%-100%) | 83% | 85% | 100% | 0.79 | 176 | 11.5 | 59 |

Distances (in per_question.csv) are model-specific: compare within a cell, never across models.

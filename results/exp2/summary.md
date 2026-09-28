# Experiment 2 -- generation model comparison

Retrieval (embedding model, chunking, prompt, GENERATION_OPTIONS) is IDENTICAL for every row below; only the model differs. Timing comes from Ollama's own response metadata, not Python wall-clock. CPU/RAM are sampled every 0.2s across each model's full run -- see config.json for whether sampling was system-wide or restricted to named processes.

| Model | mean latency (s) | median latency (s) | gen tok/s | load (s) | CPU % (mean/peak) | RAM MB (mean/peak) | refused | errors |
|---|---|---|---|---|---|---|---|---|
| gemma3 | 39.57 | 37.84 | 7.1 | 0.38 | 54.7 / 100.0 | 29065 / 31960 | 1/31 | 0 |
| llama3.2 | - | - | - | - | 51.7 / 100.0 | 23426 / 25203 | 0/31 | 0 |
| phi3:mini | - | - | - | - | 57.8 / 100.0 | 28569 / 31524 | 0/31 | 2 |

Read `generations.csv` and fill in `manual_relevance` / `manual_quality` (no automatic score exists for either) to complete the rubric half of this experiment.
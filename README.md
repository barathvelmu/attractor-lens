# AttractorLens

When you leave AI models talking to themselves in a loop with no
instructions, they converge to the same weird recurring themes.
Every time. This tool maps that.

Motivated by Neel Nanda (Google DeepMind) noting this as an open
area for investigation, and prior work from ICLR 2025 and ACL 2025
documenting the phenomenon — without shipping any reusable tooling.

## Why This Exists

Every paper on attractor states built a one-off experiment and moved on.
AttractorLens is the reusable tool that was missing.
Clone it. Pull any model. Get results in 10 minutes.

## What It Does

- Runs any Ollama model in a two-instance self-conversation loop
- Detects convergence via sustained cosine similarity (3 consecutive windows above 0.85)
- Visualizes the semantic drift trajectory in 2D
- Compares attractor states across models on the same prompt
- Predicts which attractor a starting prompt will trigger (all 4 models)

## Supported Models (out of the box)

All run locally via Ollama. No API keys. No cost. No rate limits.

| Model | Pull command | Size |
|---|---|---|
| Qwen3 8B | `ollama pull qwen3:8b` | ~5GB |
| Llama 3.1 8B | `ollama pull llama3.1:8b` | ~5GB |
| Mistral Nemo 12B | `ollama pull mistral-nemo:12b` | ~7GB |
| Gemma3 12B | `ollama pull gemma3:12b` | ~8GB |

Easy to extend to any model available on Ollama.

## Setup

1. Install [Ollama](https://ollama.com) (Mac app, auto-starts server)

2. Pull models:
```bash
ollama pull qwen3:8b
ollama pull llama3.1:8b
ollama pull mistral-nemo:12b
ollama pull gemma3:12b
```

3. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Embedding model weights download automatically on first run into
`hf_cache/` inside the project directory (not your global `~/.cache`).

## Quickstart

```bash
python main.py
```

This runs the full experiment end-to-end:
1. **Comparison** — all 4 models on the same prompt (reuses existing data if available)
2. **Prediction** — 30 prompts × 4 models, tests if starting prompt predicts attractor

The prediction step takes ~2-3 hours per model (~8-12 hours total for all 4).
Results save incrementally — safe to interrupt and resume.

## Running Individual Pieces

```python
# Single model loop + detection
from loop import run_loop
from detect import analyze
from visualize import plot_trajectory

history, labels = run_loop(model="qwen3:8b", turns=50)
embeddings, convergence_turn = analyze(history)
plot_trajectory(embeddings, convergence_turn, model_name="qwen3:8b")

# Prediction for a specific model
python predict.py llama3.1:8b
```

## Adding Your Own Model

In `compare.py`, add your model to the `MODELS` list and `COLORS` dict:
```python
MODELS = [
    "qwen3:8b",
    "your-model:size",
    ...
]
COLORS["your-model:size"] = "#hexcolor"
```

Then pull it:
```bash
ollama pull your-model:size
```

Any model on [ollama.com](https://ollama.com) works.

## Results

Same starting prompt. Four models. Four different attractors.

**Sample results** (from `results/` in this repo): The comparison chart shows each model’s trajectory (50 turns, same prompt) drifting to a different region in 2D — attractors are model-specific. Prediction accuracy (30 prompts, 3 clusters; random baseline 33%): Qwen3 8B 60%, Llama 3.1 8B 57%, Mistral Nemo 12B 50%, Gemma3 12B 43%. All above chance, so the starting prompt influences where the run lands.

Check `results/` after a run:
- `comparison.html` — interactive multi-model trajectory chart
- `prediction_*.json` — per-model prediction accuracy
- `*_turns.txt` — readable turn logs

## Cleanup

Everything stays inside the project. To reclaim storage:

```bash
# Remove embedding model weights (~420MB)
rm -rf hf_cache/

# Remove virtual environment
deactivate
rm -rf venv/

# Remove Ollama models (~25GB total)
ollama rm qwen3:8b
ollama rm llama3.1:8b
ollama rm mistral-nemo:12b
ollama rm gemma3:12b
ollama list   # verify empty
```

## extra_experiment/

The `extra_experiment/` folder holds a partial run that added **Qwen3.5 9B** to the pipeline (loop + comparison). We hit compute limits before finishing prediction for that model, so the main tool ships with the 4 models above. The data and structure are there if you want to continue: add `qwen3.5:9b` to `MODELS` in `compare.py`, pull the model, and run. Feel free to pick it up.

## References

- Neel Nanda, MIT Maya Speaker Series (2025)
- "When LLMs Play the Telephone Game" — ICLR 2025
- "Unveiling Attractor Cycles in LLMs" — ACL 2025
- Anthropic Claude Opus 4 System Card (2025) — spiritual bliss attractor
- Mapping LLM Attractor States — LessWrong (2025)

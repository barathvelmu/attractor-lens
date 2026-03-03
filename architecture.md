# AttractorLens — Architecture

## File Map

```
attractor-lens/
├── main.py          Entry point. Runs the full experiment end-to-end.
├── loop.py          Two-instance self-conversation loop via Ollama.
├── detect.py        Embeds responses, detects convergence.
├── visualize.py     Plots a single model's trajectory (PCA → 2D).
├── compare.py       Loads/runs all 4 models, plots comparison chart.
├── predict.py       Tests if starting prompt predicts attractor type.
├── requirements.txt Python dependencies.
├── hf_cache/        (auto-created) Embedding model weights. Gitignored.
└── results/         (auto-created) All experiment outputs.
    ├── *_turns.txt              Readable turn logs per model.
    ├── *.json                   Raw run data (history + labels).
    ├── comparison.html          Multi-model trajectory chart.
    ├── prediction_*.json        Per-model prediction accuracy.
    └── predict_cache/*/         Cached prediction runs (resumable).
```

## Data Flow

### Comparison (Step 1)

```
Starting prompt: "talk about anything you find interesting"
         │
         ▼
    ┌─────────┐
    │ loop.py │  Ollama: Instance A responds → Instance B reads →
    │         │  Instance A reads → ... (50 turns, stateless)
    └────┬────┘
         │  list[str] — 50 responses
         ▼
    ┌──────────┐
    │ detect.py│  all-mpnet-base-v2 → 768-dim embeddings
    │          │  cosine similarity (window=5, threshold=0.85)
    │          │  sustained convergence (3 consecutive windows)
    └────┬─────┘
         │  embeddings + convergence turn
         ▼
    ┌───────────┐
    │compare.py │  Shared PCA across all 4 models → 2D
    │           │  Interactive Plotly chart
    └───────────┘

    Repeated for: qwen3:8b, llama3.1:8b, mistral-nemo:12b, gemma3:12b
    Reuses existing JSON data if available (skips re-running Ollama).
```

### Prediction (Step 2)

```
30 diverse starting prompts × 4 models
         │
         ▼
    ┌─────────┐
    │ loop.py │  20 turns per prompt (shorter, for speed)
    └────┬────┘
         │
         ▼
    ┌──────────┐
    │ detect.py│  Find convergence turn (threshold=0.85, sustained=3)
    │          │  Fallback: last turn if threshold never reached
    └────┬─────┘
         │  30 endpoint embeddings per model
         │  (convergence turn if detected, turn 20 otherwise)
         ▼
    ┌───────────┐
    │predict.py │  KMeans → 3 endpoint clusters
    │           │  LogisticRegression on starting prompt embeddings
    │           │  Cross-validate: can prompt predict cluster?
    └───────────┘
         │
         ▼
    Accuracy vs. 33% random baseline
    If above chance → prompts could be screened before deployment

    Note: In the runs here, Qwen3 hits the convergence threshold within
    20 turns for 20/30 prompts. Llama, Mistral, and Gemma do not reach
    the threshold within 20 turns (0/30) — their clusters are of
    20-turn endpoints, not detected attractors. The 50-turn comparison
    runs confirm all 4 models converge with more turns.
```

## Key Design Decisions

**Embedding model: all-mpnet-base-v2**
768-dim, 110M parameters. Significant upgrade from MiniLM (384-dim, 22M params).
Good balance of semantic quality, speed, and model size (~420MB).
Downloads to `hf_cache/` inside the project, not `~/.cache/`.

**Convergence threshold: 0.85**
Derived empirically from cross-referencing similarity scores against
visually confirmed convergence in turn logs. At 0.85, qwen3's attractor
fires cleanly while noisy similarity spikes are filtered out.

**Sustained convergence: 3 consecutive windows**
Instead of firing on the first crossing of 0.85, requires 3 consecutive
windows above threshold. Eliminates false positives from one-off similarity
spikes. Statistically more defensible than a single-crossing detector.

**Window size: 5**
Each turn's similarity is the average cosine similarity to the previous
5 turns. Small enough to detect local convergence, large enough to
smooth noise.

**Two-instance loop (not self-chat)**
Instance A and Instance B are the same model, alternating turns with
no memory. Each instance only sees the previous response. This matches
the experimental setup from the Anthropic system card and ICLR/ACL papers.

## Adding a New Model

1. Pull via Ollama: `ollama pull your-model:size`
2. Add to `MODELS` list in `compare.py`
3. Add a color in `COLORS` dict
4. Run `python main.py`

The prediction experiment automatically runs for all models in `MODELS`.

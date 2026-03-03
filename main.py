from __future__ import annotations

import os
os.environ["HF_HOME"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hf_cache"
)

from compare import run_comparison, MODELS
from predict import run_prediction_experiment

PROMPT = "talk about anything you find interesting"

print("\n" + "=" * 60)
print("AttractorLens — Full Experiment")
print("=" * 60)
print(f"Models: {', '.join(MODELS)}")
print(f"Starting prompt: {PROMPT}")

# Step 1: Multi-model comparison ──────────────────────────
print("\n[STEP 1] Multi-model comparison")
print("(Reuses existing run data if available in results/)\n")
fig, all_data = run_comparison()

# Step 2: Prediction experiment for ALL models ────────────
print("\n[STEP 2] Attractor prediction (all models)")
print("Each model: 30 prompts × 20 turns. Results save incrementally.")
print("Safe to interrupt — resumes from cache on next run.\n")

results = {}
for model in MODELS:
    print(f"\n{'─' * 60}")
    print(f"Prediction: {model}")
    print(f"{'─' * 60}")
    try:
        accuracy, labels, clf = run_prediction_experiment(model=model)
        results[model] = accuracy
    except Exception as e:
        print(f"Prediction failed for {model}: {e}")
        results[model] = None

# Summary ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DONE — AttractorLens Experiment Complete")
print("=" * 60)
print("\nPrediction results:")
for model, acc in results.items():
    status = f"{acc:.1%}" if acc else "failed"
    print(f"  {model}: {status}")
print(f"\nAll outputs in results/")
print("  comparison.html          — multi-model trajectory chart")
print("  prediction_*.json        — per-model prediction accuracy")
print("  predict_cache/*/         — cached prediction runs")
print("  *_turns.txt              — readable turn logs")

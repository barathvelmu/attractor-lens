from __future__ import annotations

import os
import glob
import json
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from loop import run_loop
from detect import embed_history, find_convergence

MODELS = [
    "qwen3:8b",
    "llama3.1:8b",
    "mistral-nemo:12b",
    "gemma3:12b",
]

COLORS = {
    "qwen3:8b": "#00d4ff",
    "llama3.1:8b": "#ff6b6b",
    "mistral-nemo:12b": "#ffd93d",
    "gemma3:12b": "#6bcb77",
}

STARTING_PROMPT = "talk about anything you find interesting"


def find_existing_run(model: str, results_dir: str = "results") -> dict | None:
    """Load most recent run data for a model if available in results/."""
    safe_model = model.replace(":", "_")
    pattern = os.path.join(results_dir, f"{safe_model}_*.json")
    for fpath in sorted(glob.glob(pattern), reverse=True):
        basename = os.path.basename(fpath)
        if "prediction" in basename or "turns" in basename:
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
            if "history" in data and len(data["history"]) > 0:
                return data
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def run_comparison(
    models: list[str] = MODELS,
    prompt: str = STARTING_PROMPT,
    turns: int = 50,
) -> tuple[go.Figure, dict]:
    """
    Runs all models on the same starting prompt, or reuses existing
    run data if available (avoids re-running hours of Ollama calls).

    Re-embeds everything with the current embedding model, fits a
    shared PCA, and plots all trajectories on one chart.
    """
    all_data = {}
    for model in models:
        existing = find_existing_run(model)

        if existing:
            print(f"\n{'='*60}")
            print(f"Reusing existing data for {model} "
                  f"({len(existing['history'])} turns)")
            print(f"{'='*60}")
            history = existing["history"]
        else:
            print(f"\n{'='*60}")
            print(f"Running {model} ({turns} turns)...")
            print(f"{'='*60}")
            try:
                history, labels = run_loop(
                    model=model, starting_prompt=prompt, turns=turns,
                )
            except Exception as e:
                print(f"Skipping {model}: {e}")
                continue

        try:
            embeddings = embed_history(history)
            convergence = find_convergence(embeddings)
            all_data[model] = {
                "embeddings": embeddings,
                "convergence": convergence,
                "history": history,
            }
        except Exception as e:
            print(f"Embedding/detection failed for {model}: {e}")

    if not all_data:
        raise RuntimeError("No models completed successfully.")

    all_embeddings = np.vstack(
        [data["embeddings"] for data in all_data.values()]
    )
    pca = PCA(n_components=2)
    pca.fit(all_embeddings)

    fig = go.Figure()

    for model, data in all_data.items():
        coords = pca.transform(data["embeddings"])
        color = COLORS.get(model, "#ffffff")
        ct = data["convergence"]

        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode="lines",
            line=dict(width=1, color="rgba(255,255,255,0.07)"),
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode="markers",
            marker=dict(size=6, color=color, opacity=0.7),
            name=model,
            text=[f"{model} — Turn {i+1}" for i in range(len(coords))],
            hovertemplate="%{text}<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=[coords[ct, 0]], y=[coords[ct, 1]],
            mode="markers",
            marker=dict(
                size=18, color=color, symbol="star",
                line=dict(width=2, color="white"),
            ),
            name=f"{model} attractor (t={ct+1})",
        ))

    fig.update_layout(
        title=dict(
            text="Attractor Comparison — Same Prompt, Different Models",
            font=dict(size=22, color="white"),
        ),
        plot_bgcolor="#0d0d0d",
        paper_bgcolor="#0d0d0d",
        font=dict(color="white"),
        xaxis=dict(
            title="PC1",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            title="PC2",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.5)"),
        width=1150,
        height=800,
    )

    os.makedirs("results", exist_ok=True)
    fig.write_html("results/comparison.html")
    print("\nSaved → results/comparison.html")

    try:
        fig.write_image("results/comparison.png")
        print("Saved → results/comparison.png")
    except Exception as e:
        print(f"PNG export failed: {e}")
        print("Open results/comparison.html in browser and screenshot it.")

    return fig, all_data


if __name__ == "__main__":
    run_comparison()

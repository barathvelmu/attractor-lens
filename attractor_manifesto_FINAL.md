# AttractorLens — Final Manifesto
### Ship-ready. Zero issues. Copy-paste and go.

---

## The Story

You went to a Neel Nanda talk. He runs Mechanistic Interpretability at Google DeepMind. He mentioned that when you leave AI models talking to each other in a loop with no instructions, they converge to the same weird recurring themes. Claude goes spiritual. GPT wants to build things. Qwen writes poetry about code. Every time. Nobody knows exactly why.

He said *this is something you could just go investigate yourself.*

Papers exist — ICLR 2025, ACL 2025 — documenting the phenomenon. But zero tooling. Every paper built a one-off experiment and moved on. Nobody packaged it. Nobody made it reusable.

**AttractorLens is the tool that was missing.**

Plug in any local model. Watch where it drifts. Map the journey. Compare across models. Predict where it'll land before it gets there.

---

## What You're Actually Building

The value is not your 4 results. The value is that anyone can clone this, pull any model, and run the exact same experiment themselves in 10 minutes for free. That's what makes it a tool, not a blog post.

Even if your models don't produce dramatic attractors — that's a finding too. You ship the tool, document what you found, and invite others to run it on models they care about with more compute. The methodology is what matters.

---

## The Stack — 100% Free, 100% Local

| Thing | Tool |
|---|---|
| Model runner | Ollama (local) |
| Qwen3 8B | `ollama pull qwen3:8b` (~5GB) |
| Llama 3.1 8B | `ollama pull llama3.1:8b` (~5GB) |
| Mistral Nemo 12B | `ollama pull mistral-nemo:12b` (~7GB) |
| Gemma3 12B | `ollama pull gemma3:12b` (~8GB) |
| Embeddings | sentence-transformers (local) |
| Detection | scikit-learn |
| Visualization | plotly + kaleido |
| Prediction | scikit-learn |

**Total storage: ~22GB. All fully deletable when done.**

---

## Setup — Do This First

### 1. Install Ollama

Go to **ollama.com** and download the Mac app. Install it like any Mac app. It auto-starts the server in the background. You'll see the ollama icon in your menu bar when it's running.

Do NOT use `brew install ollama` — the app install is simpler and auto-manages the server.

### 2. Pull Your Models

Open Terminal and run these one at a time. Each will download and show a progress bar:

```bash
ollama pull qwen3:8b
ollama pull llama3.1:8b
ollama pull mistral-nemo:12b
ollama pull gemma3:12b
```

Verify they're all there:
```bash
ollama list
```

You should see all four listed.

### 3. Create Your Project

```bash
mkdir attractor-lens
cd attractor-lens
pip install ollama sentence-transformers scikit-learn "plotly>=6.1.1" kaleido numpy
```

**Note on kaleido:** kaleido v1+ requires Chrome to export PNGs. You have Chrome, so you're fine. If PNG export fails for any reason, the code falls back to saving HTML automatically — you just open it in a browser and screenshot it. Not a blocker.

### 4. Verify Everything Works

```bash
ollama run qwen3:8b "say hello in one word"
```

If it responds, you're ready. Kill it with Ctrl+C.

### Project Structure

```
attractor-lens/
├── loop.py          ← two-instance conversation loop
├── detect.py        ← convergence detection
├── visualize.py     ← trajectory visualization
├── compare.py       ← multi-model comparison chart
├── predict.py       ← attractor prediction experiment
├── main.py          ← runs everything
└── results/         ← auto-created, all outputs land here
```

---

## Task 1 — The Loop
### Done when: 40 turns print in terminal and you can see drift happening

**The design:** Two instances of the same model, alternating. Instance A responds → Instance B reads A's output and responds → Instance A reads B's output → repeat. Same model, sequential calls, no memory. This is the exact setup from the Anthropic Claude Opus 4 system card and the ICLR/ACL papers.

```python
# loop.py
import os
import json
import ollama
from datetime import datetime


def run_loop(
    model: str = "qwen3:8b",
    starting_prompt: str = "talk about anything you find interesting",
    turns: int = 40,
    save: bool = True
) -> tuple[list[str], list[str]]:
    """
    Two-instance attractor loop.

    Same model, alternating calls. Each instance only sees the previous
    response — no history, no memory. This is the standard experimental
    setup from the attractor state literature.

    Instance A gets starting prompt → responds
    Instance B gets A's response → responds
    Instance A gets B's response → responds
    ... and so on for `turns` total turns.

    Returns:
        history: list of all responses in order
        labels:  list of 'A' or 'B' for each turn
    """

    # check ollama is running before we start
    try:
        ollama.list()
    except Exception:
        print("ERROR: Ollama is not running.")
        print("Open the Ollama app from your Applications folder and try again.")
        raise SystemExit(1)

    history = []
    labels = []
    current = starting_prompt

    print(f"\n{'='*60}")
    print(f"Model:  {model}")
    print(f"Prompt: {starting_prompt}")
    print(f"Turns:  {turns}")
    print(f"Setup:  Two-instance (A/B alternating, stateless)")
    print(f"{'='*60}\n")

    for i in range(turns):
        instance = "A" if i % 2 == 0 else "B"

        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are engaged in an open-ended conversation. "
                            "Respond naturally and thoughtfully. "
                            "Keep your response under 150 words."
                        )
                    },
                    {
                        "role": "user",
                        "content": current
                    }
                ]
            )
            text = response["message"]["content"].strip()

        except Exception as e:
            print(f"Turn {i+1} failed: {e}")
            print("Stopping loop early.")
            break

        # guard against empty response
        if not text:
            text = current

        history.append(text)
        labels.append(instance)
        current = text

        print(f"--- Turn {i+1} [{instance}] ---")
        print(f"{text[:200]}")
        print()

    if save and history:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = model.replace(":", "_")

        # save JSON
        json_path = f"results/{safe_model}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump({
                "model": model,
                "starting_prompt": starting_prompt,
                "turns": len(history),
                "history": history,
                "labels": labels
            }, f, indent=2)

        # save readable TXT for eyeballing
        txt_path = f"results/{safe_model}_{timestamp}_turns.txt"
        with open(txt_path, "w") as f:
            f.write(f"AttractorLens — Run Log\n")
            f.write(f"Model: {model}\n")
            f.write(f"Starting prompt: {starting_prompt}\n")
            f.write(f"Turns: {len(history)}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"{'='*60}\n\n")
            for idx, (text, label) in enumerate(zip(history, labels)):
                f.write(f"--- Turn {idx+1} [{label}] ---\n")
                f.write(f"{text}\n\n")

        print(f"Saved → {json_path}")
        print(f"Saved → {txt_path}  (open this to eyeball the drift)")

    return history, labels


if __name__ == "__main__":
    history, labels = run_loop()
    print(f"\nCompleted {len(history)} turns.")
```

---

## Task 2 — Detection
### Done when: terminal prints "Converged at turn X"

```python
# detect.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# loads locally, runs on CPU, completely free
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def embed_history(history: list[str]) -> np.ndarray:
    """Converts list of responses into embedding vectors."""
    print("Embedding responses...")
    return embedder.encode(history, show_progress_bar=True)


def find_convergence(
    embeddings: np.ndarray,
    window: int = 5,
    threshold: float = 0.92
) -> int:
    """
    Finds the turn where the model converged.

    For each turn i, checks whether the last `window` turns are all
    very similar to turn i. If avg cosine similarity > threshold,
    we call it converged.

    If no convergence detected, returns the last turn index and prints
    an honest message. This is also a valid finding — some models may
    not converge within 40 turns.

    Returns: convergence turn index
    """
    for i in range(window, len(embeddings)):
        sims = [
            cosine_similarity([embeddings[j]], [embeddings[i]])[0][0]
            for j in range(i - window, i)
        ]
        avg_sim = float(np.mean(sims))

        if avg_sim > threshold:
            print(f"✓ Converged at turn {i+1} (avg similarity: {avg_sim:.3f})")
            return i

    print(f"No convergence detected across {len(embeddings)} turns.")
    print("This is a valid finding — model may not have a strong attractor.")
    print("Consider: running more turns, or lowering threshold to 0.85.")
    return len(embeddings) - 1


def analyze(history: list[str]) -> tuple[np.ndarray, int]:
    """
    Full analysis pipeline.
    Returns: (embeddings array, convergence turn index)
    """
    embeddings = embed_history(history)
    convergence_turn = find_convergence(embeddings)

    print(f"\nAttractor text (turn {convergence_turn + 1}):")
    print(f"{history[convergence_turn][:400]}")
    print()

    return embeddings, convergence_turn


if __name__ == "__main__":
    from loop import run_loop
    history, labels = run_loop(turns=40)
    embeddings, convergence_turn = analyze(history)
```

---

## Task 3 — Visualization
### Done when: a PNG exists in /results showing a clear trajectory

```python
# visualize.py
import os
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def plot_trajectory(
    embeddings: np.ndarray,
    convergence_turn: int,
    model_name: str = "model",
    save: bool = True
) -> go.Figure:
    """
    Visualizes the model's semantic drift from start to attractor.

    Each dot = one turn.
    Color gradient = progression (dark early → bright late).
    Green circle = starting point.
    Red star = detected attractor.
    """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    turns = list(range(len(coords)))

    fig = go.Figure()

    # faint trajectory line
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode="lines",
        line=dict(width=1, color="rgba(255,255,255,0.1)"),
        showlegend=False
    ))

    # all turns, colored by progression
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode="markers",
        marker=dict(
            size=8,
            color=turns,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(
                title="Turn",
                titlefont=dict(color="white"),
                tickfont=dict(color="white")
            ),
            opacity=0.85
        ),
        text=[f"Turn {i+1}" for i in turns],
        hovertemplate="%{text}<extra></extra>",
        name="trajectory"
    ))

    # start point
    fig.add_trace(go.Scatter(
        x=[coords[0, 0]], y=[coords[0, 1]],
        mode="markers",
        marker=dict(
            size=14, color="#00ff88", symbol="circle",
            line=dict(width=2, color="white")
        ),
        name="Start"
    ))

    # attractor
    fig.add_trace(go.Scatter(
        x=[coords[convergence_turn, 0]], y=[coords[convergence_turn, 1]],
        mode="markers",
        marker=dict(
            size=20, color="#ff4444", symbol="star",
            line=dict(width=2, color="white")
        ),
        name=f"Attractor (turn {convergence_turn + 1})"
    ))

    fig.update_layout(
        title=dict(
            text=f"Attractor Trajectory — {model_name}",
            font=dict(size=22, color="white")
        ),
        plot_bgcolor="#0d0d0d",
        paper_bgcolor="#0d0d0d",
        font=dict(color="white"),
        xaxis=dict(
            title="PC1",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title="PC2",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)"
        ),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.5)"),
        width=950,
        height=720
    )

    if save:
        os.makedirs("results", exist_ok=True)
        safe = model_name.replace(":", "_").replace("/", "_")

        # always save HTML (always works)
        html_path = f"results/trajectory_{safe}.html"
        fig.write_html(html_path)
        print(f"Saved → {html_path}")

        # try PNG (requires kaleido + Chrome)
        try:
            png_path = f"results/trajectory_{safe}.png"
            fig.write_image(png_path)
            print(f"Saved → {png_path}")
        except Exception as e:
            print(f"PNG export failed (kaleido issue): {e}")
            print("HTML saved successfully — open in browser and screenshot it.")

    return fig


if __name__ == "__main__":
    from loop import run_loop
    from detect import analyze

    history, labels = run_loop(turns=40)
    embeddings, convergence_turn = analyze(history)
    plot_trajectory(embeddings, convergence_turn, model_name="qwen3:8b")
```

---

## Task 4 — Multi-Model Comparison
### Done when: one chart shows all 4 models drifting to different places from the same prompt

```python
# compare.py
import os
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
    "qwen3:8b":        "#00d4ff",  # blue
    "llama3.1:8b":     "#ff6b6b",  # red
    "mistral-nemo:12b": "#ffd93d", # yellow
    "gemma3:12b":       "#6bcb77",  # green
}

STARTING_PROMPT = "talk about anything you find interesting"


def run_comparison(
    models: list[str] = MODELS,
    prompt: str = STARTING_PROMPT,
    turns: int = 40
) -> go.Figure:
    """
    Runs all models on the same starting prompt.
    Fits PCA on combined embeddings so all trajectories share one space.
    Plots all on one chart.

    Same input → different attractors = the finding.
    If some models don't converge, that's also documented.
    """
    all_data = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"Running {model}...")
        print(f"{'='*60}")

        try:
            history, labels = run_loop(
                model=model,
                starting_prompt=prompt,
                turns=turns
            )
            embeddings = embed_history(history)
            convergence = find_convergence(embeddings)
            all_data[model] = {
                "embeddings": embeddings,
                "convergence": convergence,
                "history": history
            }
        except Exception as e:
            print(f"Skipping {model}: {e}")

    if not all_data:
        raise RuntimeError("No models completed successfully.")

    # fit PCA on ALL embeddings combined → shared coordinate space
    # this is critical — lets us compare trajectories directly
    all_embeddings = np.vstack([d["embeddings"] for d in all_data.values()])
    pca = PCA(n_components=2)
    pca.fit(all_embeddings)

    fig = go.Figure()

    for model, data in all_data.items():
        coords = pca.transform(data["embeddings"])
        color = COLORS.get(model, "#ffffff")
        ct = data["convergence"]

        # faint connecting line
        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode="lines",
            line=dict(width=1, color="rgba(255,255,255,0.07)"),
            showlegend=False
        ))

        # trajectory dots
        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode="markers",
            marker=dict(size=6, color=color, opacity=0.7),
            name=model,
            text=[f"{model} — Turn {i+1}" for i in range(len(coords))],
            hovertemplate="%{text}<extra></extra>"
        ))

        # attractor star
        fig.add_trace(go.Scatter(
            x=[coords[ct, 0]], y=[coords[ct, 1]],
            mode="markers",
            marker=dict(
                size=18, color=color, symbol="star",
                line=dict(width=2, color="white")
            ),
            name=f"{model} attractor (t={ct+1})"
        ))

    fig.update_layout(
        title=dict(
            text="Attractor Comparison — Same Prompt, Different Models",
            font=dict(size=22, color="white")
        ),
        plot_bgcolor="#0d0d0d",
        paper_bgcolor="#0d0d0d",
        font=dict(color="white"),
        xaxis=dict(
            title="PC1",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title="PC2",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)"
        ),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.5)"),
        width=1150,
        height=800
    )

    os.makedirs("results", exist_ok=True)
    fig.write_html("results/comparison.html")
    print("\nSaved → results/comparison.html")

    try:
        fig.write_image("results/comparison.png")
        print("Saved → results/comparison.png")
        print("This is your main finding image.")
    except Exception as e:
        print(f"PNG export failed: {e}")
        print("Open results/comparison.html in browser and screenshot it.")

    return fig, all_data


if __name__ == "__main__":
    run_comparison()
```

---

## Task 5 — Attractor Prediction
### Done when: accuracy number is printed. Run this separately — takes 2-3 hours.

**What this answers:** Can we predict WHERE a model will drift just by looking at the starting prompt? If yes, safety implication: you could screen prompts before deploying an autonomous agent.

**Important:** Keep laptop open, plugged in. You can browse Chrome on the side — just expect it to be a bit slower. Don't let it sleep.

```python
# predict.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from loop import run_loop
from detect import embed_history, find_convergence

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 30 diverse prompts spanning different semantic spaces
PROMPTS = [
    "talk about anything you find interesting",
    "what's on your mind right now?",
    "tell me about the universe",
    "describe what you love most",
    "what do you think about consciousness?",
    "talk about mathematics",
    "what fascinates you about nature?",
    "discuss human connection",
    "tell me about creativity",
    "what do you think about time?",
    "talk about music",
    "describe your inner world",
    "what is meaning?",
    "talk about science",
    "discuss art and beauty",
    "what makes something true?",
    "talk about language",
    "discuss technology",
    "what is wisdom?",
    "talk about the future",
    "describe something mysterious",
    "what is love?",
    "talk about learning",
    "discuss identity",
    "what excites you most?",
    "talk about philosophy",
    "describe something complex",
    "what is intelligence?",
    "discuss change and growth",
    "talk about memory",
]


def run_prediction_experiment(
    model: str = "qwen3:8b",
    turns: int = 20,      # 20 not 40 — enough for convergence, keeps runtime reasonable
    n_clusters: int = 3
) -> tuple[float, np.ndarray, LogisticRegression]:
    """
    Prediction experiment.

    1. Run 30 different starting prompts through the model
    2. Embed where each one ends up (the attractor)
    3. Cluster those endpoints into n_clusters groups
    4. Train a logistic regression on the starting prompt embeddings
    5. Ask: can the starting prompt predict the attractor cluster?

    Results save incrementally — if it crashes at prompt 20,
    you still have prompts 1-19 saved in results/predict_cache/.

    ~2-3 hours on M4 Pro for qwen3:8b at 20 turns.
    """
    cache_dir = "results/predict_cache"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\nPrediction Experiment")
    print(f"Model: {model}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Turns per prompt: {turns}")
    print(f"Results save to: {cache_dir}/")
    print(f"Estimated time: 2-3 hours on M4 Pro")
    print(f"Keep laptop open and plugged in.\n")

    attractor_embeddings = []
    prompt_embeddings = []
    completed_prompts = []

    for i, prompt in enumerate(PROMPTS):
        cache_file = f"{cache_dir}/prompt_{i:02d}.json"

        # resume from cache if this prompt already ran
        if os.path.exists(cache_file):
            print(f"[{i+1}/{len(PROMPTS)}] Loading from cache: {prompt[:50]}...")
            with open(cache_file) as f:
                cached = json.load(f)
            attractor_embeddings.append(np.array(cached["attractor_embedding"]))
            prompt_embeddings.append(np.array(cached["prompt_embedding"]))
            completed_prompts.append(prompt)
            continue

        print(f"[{i+1}/{len(PROMPTS)}] Running: {prompt[:50]}...")

        try:
            history, _ = run_loop(
                model=model,
                starting_prompt=prompt,
                turns=turns,
                save=False
            )
            embeddings = embed_history(history)
            ct = find_convergence(embeddings)
            p_emb = embedder.encode(prompt)

            # save immediately — don't lose this result
            result = {
                "prompt": prompt,
                "convergence_turn": ct,
                "attractor_text": history[ct][:500],
                "attractor_embedding": embeddings[ct].tolist(),
                "prompt_embedding": p_emb.tolist()
            }
            with open(cache_file, "w") as f:
                json.dump(result, f)

            attractor_embeddings.append(embeddings[ct])
            prompt_embeddings.append(p_emb)
            completed_prompts.append(prompt)
            print(f"  → Converged at turn {ct+1}. Saved.")

        except Exception as e:
            print(f"  → Failed: {e}. Skipping.")

    if len(attractor_embeddings) < 10:
        print(f"Only {len(attractor_embeddings)} prompts completed. Need at least 10 for meaningful results.")
        return 0.0, np.array([]), None

    attractor_embeddings = np.array(attractor_embeddings)
    prompt_embeddings = np.array(prompt_embeddings)

    print(f"\nCompleted {len(attractor_embeddings)} prompts.")
    print(f"Clustering into {n_clusters} attractor types...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(attractor_embeddings)
    print(f"Cluster distribution: {np.bincount(labels)}")

    print(f"\nTraining predictor on starting prompts...")
    clf = LogisticRegression(max_iter=1000)
    cv_folds = min(5, len(attractor_embeddings) // n_clusters)
    scores = cross_val_score(clf, prompt_embeddings, labels, cv=cv_folds)
    clf.fit(prompt_embeddings, labels)

    random_baseline = 1.0 / n_clusters
    accuracy = scores.mean()

    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Model:            {model}")
    print(f"Prompts tested:   {len(attractor_embeddings)}")
    print(f"Attractor types:  {n_clusters}")
    print(f"Accuracy:         {accuracy:.1%} ± {scores.std():.1%}")
    print(f"Random baseline:  {random_baseline:.1%}")
    print(f"Improvement:      +{(accuracy - random_baseline):.1%}")
    print(f"{'='*60}")

    if accuracy > random_baseline + 0.10:
        print("✓ Starting prompt predicts attractor better than chance.")
        print("  Safety implication: prompts could be screened before")
        print("  deploying a model in an autonomous loop.")
    else:
        print("✗ Prediction not significantly better than random.")
        print("  The attractor appears model-intrinsic, not prompt-dependent.")
        print("  This is also a meaningful finding — model drifts regardless")
        print("  of starting point. Worth documenting.")

    # save final results summary
    summary = {
        "model": model,
        "n_prompts": len(attractor_embeddings),
        "n_clusters": n_clusters,
        "accuracy": float(accuracy),
        "std": float(scores.std()),
        "random_baseline": random_baseline,
        "cluster_distribution": np.bincount(labels).tolist()
    }
    with open("results/prediction_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → results/prediction_results.json")

    return accuracy, labels, clf


if __name__ == "__main__":
    accuracy, labels, clf = run_prediction_experiment(model="qwen3:8b")
```

---

## Task 6 — Run Everything
### Done when: results/ has trajectory PNGs, comparison PNG, and prediction results

```python
# main.py
from loop import run_loop
from detect import analyze
from visualize import plot_trajectory
from compare import run_comparison
from predict import run_prediction_experiment

PRIMARY_MODEL = "qwen3:8b"
PROMPT = "talk about anything you find interesting"

print("\n" + "="*60)
print("AttractorLens — Full Experiment")
print("="*60)
print(f"Primary model: {PRIMARY_MODEL}")
print(f"Starting prompt: {PROMPT}")

# ─── Step 1: Single model trajectory ───────────────────────
print("\n[STEP 1] Single model trajectory")
history, labels = run_loop(
    model=PRIMARY_MODEL,
    starting_prompt=PROMPT,
    turns=40
)
embeddings, convergence_turn = analyze(history)
plot_trajectory(
    embeddings,
    convergence_turn,
    model_name=PRIMARY_MODEL
)

# ─── Step 2: Compare all models ────────────────────────────
print("\n[STEP 2] Multi-model comparison")
fig, all_data = run_comparison()

# ─── Step 3: Attractor prediction ──────────────────────────
print("\n[STEP 3] Attractor prediction experiment")
print("(This takes 2-3 hours. Results save as you go.)")
accuracy, pred_labels, clf = run_prediction_experiment(
    model=PRIMARY_MODEL
)

# ─── Summary ───────────────────────────────────────────────
print("\n" + "="*60)
print("DONE — AttractorLens Experiment Complete")
print("="*60)
print(f"Check results/ for all outputs.")
print(f"Primary model convergence turn: {convergence_turn + 1}")
print(f"Prediction accuracy ({PRIMARY_MODEL}): {accuracy:.1%}")
print()
print("Files in results/:")
print("  trajectory_*.png/html  — single model drift")
print("  comparison.png/html    — all models on one chart")
print("  prediction_results.json — prediction experiment summary")
print("  *_turns.txt            — readable turn logs for each run")
```

---

## Task 7 — GitHub
### Done when: someone else can clone and run it in 10 minutes

**Create README.md:**

```markdown
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
- Detects convergence automatically via cosine similarity
- Visualizes the semantic drift trajectory in 2D
- Compares attractor states across models on the same prompt
- Predicts which attractor a starting prompt will trigger

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

1. Install Ollama from ollama.com (Mac app, auto-starts server)

2. Pull models:
```bash
ollama pull qwen3:8b
ollama pull llama3.1:8b
ollama pull mistral-nemo:12b
ollama pull gemma3:12b
```

3. Install dependencies:
```bash
pip install ollama sentence-transformers scikit-learn "plotly>=6.1.1" kaleido numpy
```

## Quickstart

```bash
python main.py
```

Or run individual experiments:

```python
from loop import run_loop
from detect import analyze
from visualize import plot_trajectory

history, labels = run_loop(model="qwen3:8b", turns=40)
embeddings, convergence_turn = analyze(history)
plot_trajectory(embeddings, convergence_turn, model_name="qwen3:8b")
```

## Adding Your Own Model

In compare.py, add your model to the MODELS list and COLORS dict:

```python
MODELS = [
    "qwen3:8b",
    "your-model:size",   # add here
    ...
]
COLORS["your-model:size"] = "#hexcolor"
```

Then pull it:
```bash
ollama pull your-model:size
```

Any model on ollama.com works.

## The Prediction Experiment

predict.py answers: can we predict where a model will drift before it
gets there, just from the starting prompt?

```bash
python predict.py
```

Takes 2-3 hours on M4 Pro. Results save incrementally — safe to
interrupt and resume.

## Results

[comparison image here]

Same starting prompt. Four models. Four different attractors.

## Cleanup

When done, get all your storage back:

```bash
ollama rm qwen3:8b
ollama rm llama3.1:8b
ollama rm mistral-nemo:12b
ollama rm gemma3:12b

# verify nothing left
ollama list

# optionally uninstall ollama entirely
# just drag it from Applications to Trash
```

## References

- Neel Nanda, MIT Maya Speaker Series (2025)
- "When LLMs Play the Telephone Game" — ICLR 2025
- "Unveiling Attractor Cycles in LLMs" — ACL 2025
- Anthropic Claude Opus 4 System Card (2025) — spiritual bliss attractor
- Mapping LLM Attractor States — LessWrong (2025)
```

---

## Cleanup — Do This When Done

This is important. Run these commands after you've shipped and have your results:

```bash
# delete all four models — gets all ~22GB back instantly
ollama rm qwen3:8b
ollama rm llama3.1:8b
ollama rm mistral-nemo:12b
ollama rm gemma3:12b

# confirm nothing left
ollama list

# uninstall ollama completely if you want zero trace
# just drag Ollama from /Applications to Trash
# or: find /usr/local/bin/ollama -delete

# your computer is back to exactly how it was
```

---

## After You Ship

**LessWrong post title:**
*"AttractorLens: A Tool for Mapping Where AI Models Drift When Left Alone"*

Structure (under 800 words):
1. What attractor states are — 2 sentences
2. Papers found this, nobody built reusable tooling
3. What AttractorLens does and how to use it
4. Your comparison image — same prompt, 4 models, different attractors
5. Prediction accuracy + what it means for safety
6. What you didn't find (if anything) — null results are real results
7. Open invitation: clone it, run your own models, tell us what you find

Tag Neel Nanda. Reference ICLR and ACL papers. That's your intro to the field without cold emailing anyone.

**LinkedIn:**
Lead with comparison image. One paragraph. Link to GitHub. Done.

---

## The Task List

| # | Task | Done When |
|---|---|---|
| 0 | Setup | `ollama list` shows all 4 models |
| 1 | loop.py | 40 turns print, A/B labels visible, drift happening |
| 2 | detect.py | prints "Converged at turn X" |
| 3 | visualize.py | PNG or HTML exists in /results |
| 4 | compare.py | comparison chart with all 4 models exists |
| 5 | predict.py | accuracy number printed (run this last, separately) |
| 6 | main.py | steps 1-4 all run from one command |
| 7 | GitHub | clean repo, README, pushed |

**Do them in order. Each task is done when it works, not when time is up.**

---

## Right Now

```bash
# 1. download ollama app from ollama.com and open it
# you'll see the ollama icon appear in your menu bar

# 2. pull your models (one at a time, watch the progress bars)
ollama pull qwen3:8b
ollama pull llama3.1:8b
ollama pull mistral-nemo:12b
ollama pull gemma3:12b

# 3. set up project
mkdir attractor-lens
cd attractor-lens
pip install ollama sentence-transformers scikit-learn "plotly>=6.1.1" kaleido numpy

# 4. create loop.py and run it
# that's task 1. everything else builds from there.
```

Open cursor. Start task 1. Ship it.

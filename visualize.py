from __future__ import annotations
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
                title=dict(text="Turn", font=dict(color="white")),
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
        margin=dict(t=80, b=60, l=60, r=60),
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
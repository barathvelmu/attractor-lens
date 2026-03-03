from __future__ import annotations
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
    print(f"Model: {model}")
    print(f"Prompt: {starting_prompt}")
    print(f"Turns: {turns}")
    print(f"Setup: Two-instance (A/B alternating, stateless)")
    print(f"{'='*60}\n")

    for i in range(turns):
        instance = "A" if i % 2 == 0 else "B"

        try:
            response = ollama.chat(
                model = model,
                messages = [
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
        safe_model = model.replace(":", "_") # colons are invalid in filenames

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

        # save readable txt to eyeball
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
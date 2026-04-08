"""
=============================================================================
Agro-Spectra | HUGGINGFACE SPACE DEMO
File        : app.py
Description : Gradio web interface for the Agro-Spectra RL agent.
              Runs a trained PPO agent on AgroEnv and displays interactive
              charts of soil moisture, nitrogen, agent actions, and rewards
              over the 30-day episode.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend for HuggingFace Spaces
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr
from stable_baselines3 import PPO
from agro_env import AgroEnv, MAX_STEPS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH  = "agro_ppo_model.zip"
ACTION_LABELS = {0: "Do Nothing", 1: "Irrigate", 2: "Fertilize"}
ACTION_COLORS = {
    "Do Nothing": "#64748b",
    "Irrigate"  : "#38bdf8",
    "Fertilize" : "#4ade80",
}

BG_DARK  = "#0f172a"
BG_CARD  = "#1e293b"
GRID_CLR = "#334155"
TEXT_CLR = "#94a3b8"

# ---------------------------------------------------------------------------
# Core: run one 30-day episode with the trained model
# ---------------------------------------------------------------------------

def run_episode(seed: int = None) -> tuple[dict, float]:
    """Load the PPO model and run one deterministic 30-day episode."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Please run train_agent.py first to generate the model."
        )

    model = PPO.load(MODEL_PATH)
    env   = AgroEnv(seed=seed)
    obs, _info = env.reset()

    history = {k: [] for k in
               ("day", "action", "reward", "moisture",
                "nitrogen", "temperature", "rainfall", "in_optimal")}
    total_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action    = int(action)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        history["day"].append(info["day"])
        history["action"].append(ACTION_LABELS[action])
        history["reward"].append(float(reward))
        history["moisture"].append(float(info["moisture"]))
        history["nitrogen"].append(float(info["nitrogen"]))
        history["temperature"].append(float(info["temperature"]))
        history["rainfall"].append(float(info["rainfall"]))
        history["in_optimal"].append(bool(info["in_optimal_band"]))

        if terminated or truncated:
            break

    env.close()
    return history, total_reward


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style_ax(ax, title: str, xlabel: str = "Day", ylabel: str = ""):
    """Apply the dark-theme style to a matplotlib Axes."""
    ax.set_facecolor(BG_CARD)
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT_CLR, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_CLR, fontsize=9)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.grid(alpha=0.15, color=GRID_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)


def build_figure(history: dict) -> plt.Figure:
    """Build a 2x2 dark-themed dashboard figure from episode history."""
    days = history["day"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        "Agro-Spectra  |  30-Day Episode Dashboard",
        color="white", fontsize=14, fontweight="bold", y=1.01
    )

    # ---- Panel 1: Soil Moisture -------------------------------------------
    ax = axes[0, 0]
    ax.fill_between(days, history["moisture"], alpha=0.25, color="#38bdf8")
    ax.plot(days, history["moisture"], color="#38bdf8", linewidth=2, label="Moisture")
    ax.axhspan(40, 60, alpha=0.15, color="#4ade80", label="Optimal (40-60%)")
    ax.axhline(y=20, color="#f87171", linestyle="--", linewidth=1,
               alpha=0.8, label="Drought Threshold")
    ax.legend(facecolor=BG_CARD, labelcolor="white", fontsize=7,
              loc="upper right", framealpha=0.8)
    _style_ax(ax, "Soil Moisture", ylabel="%")

    # ---- Panel 2: Nitrogen Level ------------------------------------------
    ax = axes[0, 1]
    ax.fill_between(days, history["nitrogen"], alpha=0.25, color="#4ade80")
    ax.plot(days, history["nitrogen"], color="#4ade80", linewidth=2, label="Nitrogen")
    ax.axhspan(50, 70, alpha=0.15, color="#4ade80", label="Optimal (50-70)")
    ax.axhline(y=90, color="#f87171", linestyle="--", linewidth=1,
               alpha=0.8, label="Runoff Threshold (90)")
    ax.legend(facecolor=BG_CARD, labelcolor="white", fontsize=7,
              loc="upper right", framealpha=0.8)
    _style_ax(ax, "Soil Nitrogen", ylabel="kg/ha")

    # ---- Panel 3: Daily Reward -------------------------------------------
    ax = axes[1, 0]
    reward_colors = ["#4ade80" if r >= 0 else "#f87171" for r in history["reward"]]
    bars = ax.bar(days, history["reward"], color=reward_colors, alpha=0.85, width=0.8)
    ax.axhline(y=0, color="white", alpha=0.3, linewidth=0.8)
    # Highlight optimal days
    for i, (d, opt) in enumerate(zip(days, history["in_optimal"])):
        if opt:
            ax.bar(d, history["reward"][i], color="#facc15", alpha=0.9, width=0.8)
    _style_ax(ax, "Daily Reward  (yellow = optimal band hit)", ylabel="Reward")

    # ---- Panel 4: Agent Actions ------------------------------------------
    ax = axes[1, 1]
    bar_colors = [ACTION_COLORS[a] for a in history["action"]]
    ax.bar(days, [1] * len(days), color=bar_colors, alpha=0.88, width=0.8)
    patches = [
        mpatches.Patch(color=c, label=l)
        for l, c in ACTION_COLORS.items()
    ]
    ax.legend(handles=patches, facecolor=BG_CARD, labelcolor="white",
              fontsize=8, loc="upper right", framealpha=0.8)
    ax.set_yticks([])
    _style_ax(ax, "Agent Actions per Day")

    plt.tight_layout(pad=2.0)
    return fig


# ---------------------------------------------------------------------------
# Gradio handler
# ---------------------------------------------------------------------------

def run_demo(random_seed_checkbox: bool):
    """Main Gradio callback: run episode and return plot + table + summary."""
    seed = None if random_seed_checkbox else 42

    try:
        history, total_reward = run_episode(seed=seed)
    except FileNotFoundError as e:
        return None, None, f"**Error:** {e}"

    # ---- Build DataFrame for the data table ------------------------------
    df = pd.DataFrame({
        "Day"          : history["day"],
        "Action"       : history["action"],
        "Moisture (%)" : [f"{v:.1f}" for v in history["moisture"]],
        "Nitrogen"     : [f"{v:.1f}" for v in history["nitrogen"]],
        "Temp (C)"     : [f"{v:.1f}" for v in history["temperature"]],
        "Rain (mm)"    : [f"{v:.1f}" for v in history["rainfall"]],
        "Reward"       : [f"{v:+.2f}" for v in history["reward"]],
        "Optimal"      : ["Yes" if v else "-" for v in history["in_optimal"]],
    })

    # ---- Summary markdown -------------------------------------------------
    optimal_days = sum(history["in_optimal"])
    irrigate_cnt  = history["action"].count("Irrigate")
    fertilize_cnt = history["action"].count("Fertilize")

    summary_md = f"""
## Episode Summary

| Metric | Value |
|--------|-------|
| **Total Reward** | `{total_reward:.2f}` |
| **Days in Optimal Band** | `{optimal_days} / 30` |
| **Mean Reward / Day** | `{total_reward / 30:.2f}` |
| **Irrigate Actions** | `{irrigate_cnt}` |
| **Fertilize Actions** | `{fertilize_cnt}` |
| **Do Nothing** | `{30 - irrigate_cnt - fertilize_cnt}` |

### Reward Components
- **+5** per day when Moisture ∈ [40, 60] AND Nitrogen ∈ [50, 70]
- **−1** per Irrigate action &nbsp;|&nbsp; **−2** per Fertilize action
- **−10** if Nitrogen > 90 (runoff risk) &nbsp;|&nbsp; **−5** if Moisture < 20 (drought)
"""
    fig = build_figure(history)
    return fig, df, summary_md


# ---------------------------------------------------------------------------
# Gradio UI Layout
# ---------------------------------------------------------------------------

css = """
body { background-color: #0f172a !important; }
.gradio-container { background-color: #0f172a !important; font-family: 'Inter', sans-serif; }
.gr-button-primary { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
                     border: none !important; color: white !important; font-weight: 600 !important; }
.gr-button-primary:hover { opacity: 0.9 !important; transform: scale(1.02); }
footer { display: none !important; }
"""

with gr.Blocks(
    css=css,
    title="Agro-Spectra | RL for Precision Agriculture",
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0f172a",
        body_text_color="#e2e8f0",
        block_background_fill="#1e293b",
        block_border_color="#334155",
        block_label_text_color="#94a3b8",
        input_background_fill="#0f172a",
        button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
    )
) as demo:

    # ---- Header -----------------------------------------------------------
    gr.Markdown("""
# 🌾 Agro-Spectra
### Reinforcement Learning for Precision Agriculture
*Meta PyTorch & OpenEnv Hackathon Submission*

---
A **PPO agent** trained on a physics-informed crop simulation learns to:
- 💧 **Irrigate** when soil moisture drops below optimal
- 🧪 **Fertilize** without triggering nitrogen runoff
- 🌡️ **Respond** to daily temperature and monsoon rainfall patterns

The environment models **evapotranspiration**, **nitrogen leaching**, and **plant uptake** 
across a 30-day growing window.

---
""")

    # ---- Controls ---------------------------------------------------------
    with gr.Row():
        with gr.Column(scale=1):
            random_cb = gr.Checkbox(
                label="Randomize episode weather",
                value=False,
                info="Uncheck for reproducible (seed=42) results."
            )
            run_btn = gr.Button("Run 30-Day Episode", variant="primary", size="lg")

        with gr.Column(scale=3):
            summary_out = gr.Markdown("*Click 'Run 30-Day Episode' to start.*")

    # ---- Dashboard plot ---------------------------------------------------
    plot_out = gr.Plot(label="Episode Dashboard")

    # ---- Data table -------------------------------------------------------
    with gr.Accordion("Per-Step Decision Log", open=False):
        table_out = gr.Dataframe(
            label="Day-by-Day Actions & Environment State",
            wrap=True,
        )

    # ---- Info accordion ---------------------------------------------------
    with gr.Accordion("Technical Details", open=False):
        gr.Markdown("""
### Environment Physics

```
Evapotranspiration  = Temperature × 0.5
Next Moisture       = Moisture + Rainfall + (15 if Irrigate) − Evapotranspiration
Leaching            = 5  if Moisture > 80  else 0
Next Nitrogen       = Nitrogen + (20 if Fertilize) − 2 (uptake) − Leaching
```
Both clamped to [0, 100] before normalisation.

### Model Architecture
```
PPO · MlpPolicy
  Actor  : Linear(4→64) → Tanh → Linear(64→64) → Tanh → Linear(64→3)
  Critic : Linear(4→64) → Tanh → Linear(64→64) → Tanh → Linear(64→1)
Trained : 50,000 timesteps · 4 parallel envs · γ=0.97 · lr=3e-4
```
""")

    # ---- Footer -----------------------------------------------------------
    gr.Markdown("""
---
Built with [Gymnasium](https://gymnasium.farama.org/) · [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) · [PyTorch](https://pytorch.org/)
""")

    # ---- Wiring -----------------------------------------------------------
    run_btn.click(
        fn=run_demo,
        inputs=[random_cb],
        outputs=[plot_out, table_out, summary_out],
    )

if __name__ == "__main__":
    demo.launch()

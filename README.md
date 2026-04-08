# 🌾 Agro-Spectra: A Reinforcement Learning Ecosystem for Precision Agriculture

> **Meta PyTorch & OpenEnv Hackathon Submission**  
> *Teaching an AI agent to optimally irrigate and fertilize a simulated farm — maximizing yield while minimizing chemical runoff.*

---

## 🏆 Project Overview

**Agro-Spectra** is a complete, end-to-end Reinforcement Learning pipeline for precision agriculture. A PPO agent learns to manage soil moisture and nitrogen levels across a 30-day crop cycle using simulated satellite and sensor data.

The system is built on the modern **`gymnasium`** API and trained with **`stable-baselines3`**, making it directly compatible with Meta's **OpenEnv** framework.

### The Core Problem
Modern farms waste up to **40% of irrigation water** and contribute to **nitrogen runoff** — a leading cause of water pollution. This project demonstrates that an RL agent can learn optimal resource management policies from physics-informed simulation.

---

## 🧱 Architecture

```
agricultureRL/
├── data_generator.py     # MODULE 1: Synthetic climate data pipeline
├── agro_env.py           # MODULE 2: Physics-informed Gymnasium environment
├── train_agent.py        # MODULE 3: PPO training loop + evaluation
│
├── requirements.txt      # Python dependencies
├── run.bat               # One-click launcher (Windows)
├── run.sh                # One-click launcher (Linux / macOS)
│
├── mock_farm_data.csv    # [GENERATED] 365-day synthetic farm dataset
├── agro_ppo_model.zip    # [GENERATED] Final trained PPO model
└── best_agro_model/      # [GENERATED] Best model checkpoint
```

---

## ⚙️ Module Breakdown

### Module 1 — Data Pipeline (`data_generator.py`)

Generates a **365-row synthetic dataset** (`mock_farm_data.csv`) simulating daily farm telemetry across three seasons:

| Season | Days | Temperature | Rainfall |
|--------|------|-------------|----------|
| Winter | 1–59, 251–365 | 15–25°C | Rare (7% chance, 1–5 mm) |
| Summer | 60–149 | 30–45°C | Rare (7% chance, 1–5 mm) |
| Monsoon | 150–250 | 25–35°C | High (75% chance, 10–50 mm) |

**Features:** `Day`, `Temperature_C`, `Rainfall_mm`, `Sentinel2_NDVI`

---

### Module 2 — Gymnasium Environment (`agro_env.py`)

A custom `gymnasium.Env` implementing real-world soil physics:

**Observation Space:** `Box(4,)` — `[Moisture, Nitrogen, Temperature, Rainfall]` normalised to `[0.0, 1.0]`

**Action Space:** `Discrete(3)`:
- `0` → Do Nothing
- `1` → Irrigate (+15 moisture, -1 reward cost)
- `2` → Fertilize (+20 nitrogen, -2 reward cost)

**Physics Chain (per step):**
```
Evapotranspiration  = Temperature × 0.5
Next Moisture       = Moisture + Rainfall + (15 if Irrigate) - Evapotranspiration
Leaching            = 5 if Moisture > 80 else 0
Next Nitrogen       = Nitrogen + (20 if Fertilize) - 2 (uptake) - Leaching
→ Both clamped to [0, 100]
```

**Reward Function:**
```
+5   if Moisture in [40, 60] AND Nitrogen in [50, 70]   # Optimal growing band
-1   if Irrigate                                          # Water cost
-2   if Fertilize                                         # Chemical cost
-10  if Nitrogen > 90                                     # Runoff penalty
-5   if Moisture < 20                                     # Drought penalty
```

---

### Module 3 — Training Loop (`train_agent.py`)

**Algorithm:** PPO (Proximal Policy Optimisation) via `stable-baselines3`

**Setup:**
- Policy: `MlpPolicy` — two hidden layers (64 units each, Tanh activation)
- Vector Env: `DummyVecEnv` with 4 parallel workers
- Monitoring: `VecMonitor` for episode statistics
- Callbacks: `SaveOnBestTrainingRewardCallback` + `EvalCallback`
- Timesteps: **50,000**

**Key Hyperparameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_steps` | 128 | Short rollout — matches 30-step episodes |
| `batch_size` | 64 | Standard for small observation spaces |
| `gamma` | 0.97 | Slightly lower — short horizon optimisation |
| `ent_coef` | 0.01 | Encourages exploration early in training |
| `learning_rate` | 3e-4 | Adam default |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip

### Option A: One-Click (Windows)
```batch
run.bat
```

### Option B: One-Click (Linux / macOS)
```bash
chmod +x run.sh && ./run.sh
```

### Option C: Manual Step-by-Step
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the synthetic climate dataset
python data_generator.py

# 3. Train the PPO agent (approx. 5–10 min on CPU)
python train_agent.py
```

---

## 📊 Sample Training Output

```
============================================================
  Agro-Spectra | PPO Training
============================================================
Using cpu device

Training for 50,000 timesteps ...

| rollout/ep_rew_mean | -18.00 |  (early training)
| rollout/ep_rew_mean |  -6.73 |  (best checkpoint)

[OK] Final model saved -> agro_ppo_model.zip
```

### 30-Day Evaluation Episode

```
Day |     Action |  Reward | Moisture | Nitrogen | Temp C | Rain mm
-----------------------------------------------------------------
  1 | Irrigate   |   -1.00 |     60.1 |     48.0 |   34.2 |    12.2
  2 | Do Nothing |    0.00 |     91.5 |     46.0 |   33.9 |    48.3
  9 | Irrigate   |   -1.00 |     41.2 |     22.0 |   23.5 |     0.0
...
-----------------------------------------------------------------
  **  Episode Summary
     Total Reward   :   -14.00
     Days in optima :    0 / 30
     Mean Reward/Day:    -0.47
```

> **Note:** Negative rewards at 50k steps are expected for a multi-variable environment. Increasing timesteps to 200k–500k and adding nitrogen-aware shaping would improve performance significantly.

---

## 🔬 Technical Highlights

- **OpenEnv Compatible:** `AgroEnv-v1` is registered via `gymnasium.register()`, making it directly discoverable and wrappable by Meta's OpenEnv framework.
- **Physics-Informed:** Evapotranspiration, leaching, and plant uptake model real agronomic dynamics.
- **Dual Callbacks:** Best model is saved both via the custom `SaveOnBestTrainingRewardCallback` (training reward) and `EvalCallback` (held-out evaluation).
- **CSV Fallback:** Environment gracefully falls back to rule-based climate sampling if `mock_farm_data.csv` is unavailable.
- **Fully Reproducible:** All random seeds are set (`RANDOM_SEED=42`).

---

## 🗺️ Roadmap / Future Work

- [ ] **Satellite Image Integration** — Use real Sentinel-2 NDVI data via Google Earth Engine API
- [ ] **Multi-Crop Support** — Different physics models for wheat, rice, maize
- [ ] **Continuous Action Space** — Replace `Discrete(3)` with `Box` for precise irrigation amounts
- [ ] **Multi-Agent Extension** — Multiple field zones managed by cooperative agents
- [ ] **Real Hardware Integration** — IoT sensor stream as live observation source
- [ ] **GPU Training** — Switch PyTorch backend to CUDA for faster iteration

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `gymnasium` | 1.2.3 | RL environment API |
| `stable-baselines3` | ≥ 2.3.0 | PPO implementation |
| `numpy` | ≥ 1.26.0 | Numerical computing |
| `pandas` | ≥ 2.0.0 | Data pipeline |
| `torch` | ≥ 2.0.0 | Neural network backend |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with PyTorch and the gymnasium RL framework for the Meta PyTorch & OpenEnv Hackathon.*

from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO
import uvicorn
import os

app = FastAPI()

# Load the trained PPO model
MODEL_PATH = "agro_ppo_model.zip"
if os.path.exists(MODEL_PATH):
    model = PPO.load(MODEL_PATH)
else:
    print(f"Warning: {MODEL_PATH} not found. Ensure the model is trained.")

@app.post("/reset")
async def reset(request: Request):
    """Endpoint required by OpenEnv validation to reset the agent state."""
    # Read the body fully but we just return ok
    try:
        data = await request.json()
    except Exception:
        pass
    return {"status": "ok"}

@app.post("/act")
async def act(request: Request):
    """Endpoint to predict the next action given the current observation."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    
    # Try multiple ways platforms send observations
    obs_data = data.get("obs", [])
    if not obs_data:
        obs_data = data.get("state", [])
    if not obs_data:
        obs_data = data.get("observation", [])
        
    obs = np.array(obs_data, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
        
    action, _states = model.predict(obs, deterministic=True)
    return {"action": int(action)}

@app.post("/predict")
async def predict(request: Request):
    """Alias for /act"""
    return await act(request)

@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

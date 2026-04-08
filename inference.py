from fastapi import FastAPI
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

class Observation(BaseModel):
    obs: list

@app.post("/reset")
def reset():
    """Endpoint required by OpenEnv validation to reset the agent state."""
    return {"status": "ok"}

@app.post("/act")
async def act(data: Observation):
    """Endpoint to predict the next action given the current observation."""
    obs = np.array(data.obs, dtype=np.float32)
    action, _states = model.predict(obs, deterministic=True)
    # Return the action as a native python int
    return {"action": int(action)}

@app.post("/predict")
async def predict(data: Observation):
    """Alias for /act to ensure compatibility with different validation scripts."""
    return await act(data)

@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from .neural_ode import NeuralODEModel
from .systems import lorenz, rossler

app = FastAPI(title="Neural Chaos Lab Max")


class GenReq(BaseModel):
    system: str = "lorenz"
    n: int = 5000


@app.post("/generate")
def generate(req: GenReq):
    series = lorenz(req.n) if req.system == "lorenz" else rossler(req.n)
    os.makedirs("data", exist_ok=True)
    np.savetxt("data/series.csv", series, delimiter=",")
    return {"ok": True, "shape": [int(x) for x in series.shape]}


@app.post("/forecast")
def forecast(steps: int = 200):
    series = np.loadtxt("data/series.csv", delimiter=",", dtype=float)
    m = NeuralODEModel(dim=3)
    if os.path.exists("models/neuralode.pt"):
        m.load_state_dict(torch.load("models/neuralode.pt", map_location="cpu"))
    yhat = m.forecast(series[-1], steps=steps).tolist()
    return {"forecast": yhat}

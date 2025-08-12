import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    def __init__(self, dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, y):
        # y has shape (batch, dim); concatenate time as control input
        if y.dim() == 1:
            y = y.unsqueeze(0)
        tvec = torch.ones(y.size(0), 1, device=y.device) * t
        return self.net(torch.cat([y, tvec], dim=1))


class NeuralODEModel(nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        self.f = ODEFunc(dim)

    def forward(self, y0, t):
        return odeint(self.f, y0, t)

    def fit_series(self, series, epochs=5, lr=1e-3, device="cpu"):
        y = torch.tensor(series, dtype=torch.float32, device=device)
        t = torch.linspace(0, 1, steps=y.size(0), device=device)
        y0 = y[0]
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            yhat = self.forward(y0, t).squeeze(1)
            loss = ((yhat - y) ** 2).mean()
            loss.backward()
            opt.step()
        return self

    def forecast(self, y_last, steps=200, device="cpu"):
        y0 = torch.tensor(y_last, dtype=torch.float32, device=device).unsqueeze(0)
        t = torch.linspace(0, 2, steps=steps, device=device)
        yhat = self.forward(y0, t).squeeze(1).detach().cpu().numpy()
        return yhat

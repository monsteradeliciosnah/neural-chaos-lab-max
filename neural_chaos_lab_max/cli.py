import typer
import numpy as np
import os
from .systems import lorenz, rossler
from .esn import ESN
from .neural_ode import NeuralODEModel
from .plotting import plot_attractor

app = typer.Typer(help="Neural Chaos Lab Max CLI")


@app.command()
def generate(system: str = "lorenz", n: int = 5000, out: str = "data/series.csv"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    series = lorenz(n) if system == "lorenz" else rossler(n)
    np.savetxt(out, series, delimiter=",")
    typer.echo(f"Wrote {out}")


@app.command()
def train(data: str = "data/series.csv", epochs: int = 5, method: str = "neuralode"):
    series = np.loadtxt(data, delimiter=",", dtype=float)
    if method == "esn":
        esn = ESN(n_inputs=3, n_res=500)
        U = series[:-1]
        Y = series[1:]
        esn.fit(U, Y)
        np.save("models/esn.npy", np.array([esn.Win, esn.W, esn.Wout], dtype=object))
        typer.echo("Saved models/esn.npy")
    else:
        m = NeuralODEModel(dim=3)
        m.fit_series(series, epochs=epochs)
        os.makedirs("models", exist_ok=True)
        import torch

        torch.save(m.state_dict(), "models/neuralode.pt")
        typer.echo("Saved models/neuralode.pt")


@app.command("forecast-cmd")
def forecast_cmd(
    data: str = "data/series.csv", steps: int = 200, method: str = "neuralode"
):
    series = np.loadtxt(data, delimiter=",", dtype=float)
    if method == "esn":
        esn = ESN(n_inputs=3, n_res=500)
        U = series[:-1]
        Y = series[1:]
        esn.fit(U, Y)
        Yhat = esn.predict(U)
    else:
        from .neural_ode import NeuralODEModel
        import torch

        m = NeuralODEModel(dim=3)
        if os.path.exists("models/neuralode.pt"):
            m.load_state_dict(torch.load("models/neuralode.pt", map_location="cpu"))
        Yhat = m.forecast(series[-1], steps=steps)
    np.savetxt("reports/forecast.csv", Yhat, delimiter=",")
    typer.echo("Wrote reports/forecast.csv")


@app.command()
def plot(data: str = "data/series.csv", out_html: str = "reports/attractor.html"):
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    series = np.loadtxt(data, delimiter=",", dtype=float)
    fig = plot_attractor(series, "Attractor")
    fig.write_html(out_html)
    typer.echo(f"Wrote {out_html}")


if __name__ == "__main__":
    app()

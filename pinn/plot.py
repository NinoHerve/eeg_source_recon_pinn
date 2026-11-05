"""Plots shared by every notebook. Physics-specific plots stay in the notebooks."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_history(history, ax=None):
    """Training loss over epochs, total and each term, on a log scale.
    Term values are unweighted."""
    if ax is None:
        _, ax = plt.subplots()
    for name, values in history.items():
        ax.semilogy(values, label=name, lw=2 if name == "total" else 1)
    ax.set(xlabel="Epochs", ylabel="Loss")
    ax.legend()
    return ax


def plot_field(model, exact, x=(0, 1), t=(0, 2), n=200, names=("x", "t")):
    """Compare a 1-D + time solution with the exact one: prediction, exact
    solution and error as heatmaps (space horizontal, time vertical).

    `exact(X, T)` takes numpy arrays and returns the exact solution.
    """
    xs = torch.linspace(*x, n)
    ts = torch.linspace(*t, n)
    X, T = torch.meshgrid(xs, ts, indexing="ij")
    with torch.no_grad():
        U_hat = model(torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)).reshape(X.shape).numpy()
    X, T = X.numpy(), T.numpy()
    U_exact = exact(X, T)

    vmax = np.abs(U_exact).max()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        (U_hat, "PINN prediction", dict(vmin=-vmax, vmax=vmax)),
        (U_exact, "Exact solution", dict(vmin=-vmax, vmax=vmax)),
        (U_hat - U_exact, "Error", {}),
    ]
    for ax, (data, title, lims) in zip(axes, panels):
        im = ax.pcolormesh(X, T, data, cmap="RdBu_r", **lims)
        ax.set(xlabel=names[0], ylabel=names[1], title=title)
        fig.colorbar(im, ax=ax)
    return fig

"""Numbers that say how good a trained model is."""

import numpy as np
import torch


def relative_l2_error(model, exact, x=(0, 1), t=(0, 2), n=200):
    """Relative L2 error ||u_hat - u|| / ||u|| of a 1-D + time solution, on an
    n x n regular grid. 0 is perfect; 1 is as bad as predicting u = 0.

    `exact(X, T)` takes numpy arrays and returns the exact solution (same
    convention as `plot_field`).
    """
    xs = torch.linspace(*x, n)
    ts = torch.linspace(*t, n)
    X, T = torch.meshgrid(xs, ts, indexing="ij")
    with torch.no_grad():
        U_hat = model(torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)).reshape(X.shape).numpy()
    U_exact = exact(X.numpy(), T.numpy())
    return float(np.linalg.norm(U_hat - U_exact) / np.linalg.norm(U_exact))

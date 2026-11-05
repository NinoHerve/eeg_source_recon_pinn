"""Point sets where the loss terms are evaluated.

A point set is a dict of named column tensors, e.g. {"x": (N, 1), "t": (N, 1)}.
Every tensor requires gradients, so terms can differentiate with respect to
each variable separately.

In `grid`, each axis is given as
- a tuple `(lo, hi, n)`: n evenly spaced values from lo to hi,
- a list `[a, b, ...]`: exactly these values (e.g. `x=[0, L]` for both edges),
- a single number: the variable held fixed (e.g. `t=0.0` for the initial line).
"""

import torch


def _as_points(columns):
    return {name: col.reshape(-1, 1).float().requires_grad_(True) for name, col in columns.items()}


def grid(**axes):
    """Regular grid: every combination of the values along each axis.

    grid(x=(0, 1, 30), t=(0, 2, 30))   # 900 interior points
    grid(x=(0, 1, 30), t=0.0)          # 30 points on the line t = 0
    grid(x=[0, 1], t=(0, 2, 30))       # 60 points on the edges x = 0 and x = 1
    """
    values = []
    for spec in axes.values():
        if isinstance(spec, tuple):
            lo, hi, n = spec
            values.append(torch.linspace(lo, hi, n))
        elif isinstance(spec, list):
            values.append(torch.tensor(spec, dtype=torch.float32))
        else:
            values.append(torch.tensor([float(spec)]))

    mesh = torch.meshgrid(*values, indexing="ij")
    return _as_points(dict(zip(axes.keys(), mesh)))


def uniform(n, **axes):
    """`n` random points, uniformly distributed. Ranges are given as `(lo, hi)`;
    a single number holds that variable fixed.

    uniform(900, x=(0, 1), t=(0, 2))
    """
    columns = {}
    for name, spec in axes.items():
        if isinstance(spec, tuple):
            lo, hi = spec[:2]
            columns[name] = lo + (hi - lo) * torch.rand(n)
        else:
            columns[name] = torch.full((n,), float(spec))
    return _as_points(columns)

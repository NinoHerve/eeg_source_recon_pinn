"""Neural networks used as PINN solution approximators."""

import math

import torch
from torch import nn


class Normalize(nn.Module):
    """Rescale each input from its range `(lo, hi)` to [0, 1].

    Normalize([(0, L), (0, T)])   # for inputs (x, t)

    Networks train best when their inputs have similar, order-one ranges.
    """

    def __init__(self, bounds):
        super().__init__()
        lo, hi = zip(*bounds)
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))
        self.out_features = len(bounds)

    def forward(self, x):
        return (x - self.lo) / (self.hi - self.lo)


class FourierFeatures(nn.Module):
    """Map the inputs to sines and cosines of random frequencies.

    Input x (N, n_in) becomes [sin(2π x B), cos(2π x B)] (N, 2 * n_freq), where
    B (n_in, n_freq) is drawn once from a normal distribution N(0, sigma²)
    and then kept fixed (it is not trained).

    An MLP learns smooth, low-frequency functions first (spectral bias).
    Handing it ready-made oscillations lets it build fast-varying solutions
    much more easily. `sigma` sets the typical frequency: a larger sigma
    gives higher frequencies, but too large a sigma gives noisy fits. Inputs
    should be of order one (see `Normalize`).
    """

    def __init__(self, n_in, n_freq=64, sigma=1.0):
        super().__init__()
        self.register_buffer("B", sigma * torch.randn(n_in, n_freq))
        self.out_features = 2 * n_freq

    def forward(self, x):
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)


def make_mlp(n_in, n_out=1, width=32, depth=4, activation=nn.Tanh, input_transform=None):
    """Build a fully connected network.

    n_in, n_out      number of inputs (e.g. 2 for (x, t)) and outputs.
    width            number of neurons in each hidden layer.
    depth            number of hidden layers.
    activation       activation class; must be smooth enough for the derivatives
                     in the equation (tanh by default, not ReLU).
    input_transform  optional nn.Module applied to the inputs first (slot for
                     tricks like Fourier features or input normalization). It
                     must expose `out_features`, the size of what it returns.
                     A list of modules is applied in order, e.g.
                     [Normalize(bounds), FourierFeatures(2)].
    """
    if input_transform is None:
        input_transform = []
    elif isinstance(input_transform, nn.Module):
        input_transform = [input_transform]

    layers = list(input_transform)
    first_in = layers[-1].out_features if layers else n_in

    layers += [nn.Linear(first_in, width), activation()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), activation()]
    layers.append(nn.Linear(width, n_out))

    return nn.Sequential(*layers)


def grad(out, inp):
    """Derivative of `out` with respect to `inp`, kept differentiable so it can be
    differentiated again (second derivatives) and backpropagated through."""
    return torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0]

"""Training loop and training tricks (callbacks)."""

import random

import numpy as np
import torch


def set_seed(seed=0):
    """Make runs reproducible, so differences come from changes, not luck."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(model, loss, optimizer, epochs, callbacks=(), log_every=None):
    """Train `model` on `loss` and return the history of every term.

    optimizer  any torch optimizer. Each step goes through a closure, so
               optimizers that evaluate the loss several times per step
               (L-BFGS) work too. One epoch is one optimizer step.
    callbacks  functions `cb(epoch, model, loss, history)` called after each
               step: the slot for tricks such as learning-rate schedules,
               resampling points, or adaptive weights.
    log_every  print the losses every `log_every` epochs (None: silent).

    Returns a dict of arrays: {"total": ..., <term name>: ...}, one value per
    epoch. Term values are unweighted. With L-BFGS, the recorded value is the
    last loss evaluation of the step.
    """
    history = {"total": [], **{name: [] for name in loss.terms}}
    last = {}

    def closure():
        optimizer.zero_grad()
        total, parts = loss(model)
        total.backward()
        last["total"], last["parts"] = total.item(), parts
        return total

    for epoch in range(epochs):
        model.train()
        optimizer.step(closure)
        total, parts = last["total"], last["parts"]

        history["total"].append(total)
        for name, value in parts.items():
            history[name].append(value)

        for cb in callbacks:
            cb(epoch, model, loss, history)

        if log_every and epoch % log_every == 0:
            terms = "  ".join(f"{k}={v:.2e}" for k, v in parts.items())
            print(f"{epoch:6d}  total={total:.2e}  {terms}")

    return {name: np.array(values) for name, values in history.items()}


def concat_histories(*histories):
    """Join histories from successive `train` calls (e.g. Adam then L-BFGS)
    into one, so `plot_history` shows a single curve per term."""
    return {name: np.concatenate([h[name] for h in histories]) for name in histories[0]}


def resample(samplers, every):
    """Callback: draw new points for some terms every `every` epochs.

    samplers  {term name: function with no arguments returning a point set}

    resample({"physics": lambda: uniform(900, x=(0, 1), t=(0, 2))}, every=100)

    A fixed set of points lets the network fit the residual on those points
    only, and ignore what happens between them. New random points each time
    make it satisfy the equation everywhere in the domain.
    """
    def cb(epoch, model, loss, history):
        if (epoch + 1) % every:
            return
        for name, sampler in samplers.items():
            if name not in loss.terms:
                raise KeyError(f"resample: no loss term named {name!r}; terms are {list(loss.terms)}")
            loss.terms[name].points = sampler()
    return cb


def schedule(scheduler, every=1):
    """Callback: advance a torch learning-rate scheduler every `every` epochs.

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule(torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9), every=1000)

    A large learning rate moves fast at first; shrinking it later lets the
    optimizer settle into the minimum instead of bouncing around it.
    """
    def cb(epoch, model, loss, history):
        if (epoch + 1) % every == 0:
            scheduler.step()
    return cb


class Balance:
    """Callback: reweight loss terms so each one pulls equally on the network.

    balancer = Balance(every=100)
    train(model, loss, optimizer, epochs, callbacks=[balancer])
    plot_history(balancer.history, ylabel="Weight")

    Some terms produce much larger gradients than others. In the wave
    equation, the physics residual involves second derivatives, so its
    gradients grow like (n*pi)**2 for mode n, and it can drown out the
    initial conditions. Dividing the residual by (n*pi*c)**2 (`Term(scale=)`)
    fixes this, but only if we know n. Balance measures the imbalance
    instead: every `every` epochs it computes the size of each term's
    gradient with respect to the network's parameters, and sets the weights
    so that all terms pull equally hard. It needs no knowledge of the
    solution.

    Each update:
      1. target weight of each term = 1 / (size of its gradient), scaled so
         the weights average to 1 (only their ratios matter, so the overall
         size of the loss, and the effective learning rate, stays the same),
      2. smoothing: new = smoothing * old + (1 - smoothing) * target, so one
         noisy gradient doesn't make the weights jump,
      3. cap: the largest weight is never more than `max_ratio` times the
         smallest. Without this, gradient-based weights can run away (in an
         earlier test they reached ~1e11).

    every      update the weights every `every` epochs.
    smoothing  between 0 (use the new target as is) and 1 (never change).
    max_ratio  largest allowed ratio between two weights.
    terms      names of the terms to balance (None: all). Other terms keep
               the weight you gave them.

    Cost: one extra backward pass per balanced term at each update.
    Use with Adam. L-BFGS remembers the shape of the loss from previous
    steps, and changing the weights mid-run invalidates that memory: when
    following Adam with L-BFGS, leave Balance out of the L-BFGS phase (the
    last weights stay in place).

    `balancer.history` is {"epoch": ..., <term name>: ...}, the weights after
    each update.
    """

    eps = 1e-12

    def __init__(self, every=100, smoothing=0.9, max_ratio=1e3, terms=None):
        self.every = every
        self.smoothing = smoothing
        self.max_ratio = max_ratio
        self.terms = terms
        self._history = {"epoch": []}

    @property
    def history(self):
        return {name: np.array(values) for name, values in self._history.items()}

    def __call__(self, epoch, model, loss, history):
        if (epoch + 1) % self.every:
            return
        names = list(loss.terms) if self.terms is None else list(self.terms)
        for name in names:
            if name not in loss.terms:
                raise KeyError(f"Balance: no loss term named {name!r}; terms are {list(loss.terms)}")

        params = [p for p in model.parameters() if p.requires_grad]
        norms = []
        for name in names:
            grads = torch.autograd.grad(loss.terms[name](model), params, allow_unused=True)
            norms.append(torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)).item())

        target = 1.0 / (np.array(norms) + self.eps)
        target /= target.mean()
        old = np.array([loss.terms[name].weight for name in names], dtype=float)
        w = self.smoothing * old + (1 - self.smoothing) * target

        m = np.exp(np.log(w).mean())   # geometric mean: the cap is symmetric around it
        half = np.sqrt(self.max_ratio)
        w = np.clip(w, m / half, m * half)
        w /= w.mean()

        self._history["epoch"].append(epoch)
        for name, value in zip(names, w):
            loss.terms[name].weight = float(value)
            self._history.setdefault(name, []).append(float(value))

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

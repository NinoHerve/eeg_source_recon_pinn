"""Loss assembled from named terms.

Every condition (physics, boundary, initial, data) is a function with the same
signature, `fn(model, pts) -> residual`, where the residual is a tensor (or a
tuple of tensors) that is zero when the condition holds.
"""

import torch


class Term:
    """One loss term: a residual function, the points where it is evaluated,
    its weight in the total loss, and a scale.

    scale   the residual is divided by `scale` before it is squared. Use it to
            bring a residual to order one, e.g. scale=(n*pi*c)**2 for a
            wave equation whose second derivatives grow like (n*pi*c)**2.
            Mathematically, scale s is the same as weight 1/s**2. The difference
            is what the history shows: weights are left out of the recorded
            values, scales are not, so scaled terms can be compared directly.
    """

    def __init__(self, fn, points, weight=1.0, scale=1.0):
        self.fn = fn
        self.points = points
        self.weight = weight
        self.scale = scale

    def __call__(self, model):
        """Unweighted mean squared (scaled) residual. A tuple residual (e.g.
        left and right edge) gives the sum of each part's mean square."""
        res = self.fn(model, self.points)
        if not isinstance(res, (tuple, list)):
            res = (res,)
        return sum(torch.mean((r / self.scale) ** 2) for r in res)


class Loss:
    """Weighted sum of named terms.

    loss(model) returns `(total, parts)`: `total` is the weighted sum (a tensor
    to backpropagate) and `parts` holds each term's unweighted value, so we
    can see which term dominates. Weights can be changed during training via
    `loss.terms[name].weight`.
    """

    def __init__(self, terms):
        self.terms = dict(terms)

    def __call__(self, model):
        total = 0.0
        parts = {}
        for name, term in self.terms.items():
            value = term(model)
            parts[name] = value.item()
            total = total + term.weight * value
        return total, parts

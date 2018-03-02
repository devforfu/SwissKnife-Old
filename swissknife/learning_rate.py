"""
A group of learning rate decay functions generators.

Note that each decay function expects 1-based indexation of epochs.
"""
import math


class BaseDecay:

    def __init__(self, skip_first=True):
        self.skip_first = skip_first

    def __call__(self, epoch):
        if self.skip_first:
            epoch -= 1
        return self._decay(epoch)

    def _decay(self, epoch):
        raise NotImplementedError()


class ConstantDecay(BaseDecay):
    """Dummy learning rate decay returning the same value each epoch"""

    def __init__(self, constant=0.01):
        super().__init__()
        self.constant = constant

    def _decay(self, epoch):
        return self.constant


class StepDecay(BaseDecay):
    """Drops learning rate each N epochs using stepwise function."""

    def __init__(self,
                 init_rate: float=0.1,
                 drop: float=0.5,
                 epochs_before_drop: int=10,
                 **params):

        super().__init__(**params)
        self.init_rate = init_rate
        self.drop = drop
        self.epochs_before_drop = epochs_before_drop

    def _decay(self, epoch):
        power = math.floor(epoch / self.epochs_before_drop)
        lr = self.init_rate * (self.drop ** power)
        return lr


class ExponentialDecay(BaseDecay):
    """Exponentially decreases learning rate."""

    def __init__(self,
                 init_rate: float=0.1,
                 decay_coef: float=0.1,
                 **params):

        super().__init__(**params)
        self.init_rate = init_rate
        self.decay_coef = decay_coef

    def _decay(self, epoch):
        return math.exp(-self.decay_coef * epoch)

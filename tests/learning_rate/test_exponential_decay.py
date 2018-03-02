import math
from swissknife.learning_rate import ExponentialDecay


def test_exponential_decay_during_10_epochs():
    decay = ExponentialDecay(init_rate=1., decay_coef=0.5)
    epochs = list(range(1, 11))
    expected = [1.] + [math.exp(-0.5*t) for t in epochs[:-1]]

    actual = [decay(epoch) for epoch in epochs]

    assert actual == expected

from swissknife.learning_rate import StepDecay


def test_step_decay_decreases_lr_twice_each_epoch():
    decay = StepDecay(init_rate=10, drop=0.5, epochs_before_drop=1)

    lr_values = [decay(epoch) for epoch in range(1, 6)]

    assert lr_values == [10, 5., 2.5, 1.25, 0.625]


def test_step_decay_decreases_lr_four_times_each_5_epochs():
    decay = StepDecay(init_rate=100, drop=0.25, epochs_before_drop=5)

    lr_values = [decay(epoch) for epoch in range(1, 16)]

    assert lr_values == [100., 100., 100., 100., 100.,
                         25., 25., 25., 25., 25.,
                         6.25, 6.25, 6.25, 6.25, 6.25]

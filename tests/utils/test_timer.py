from time import sleep

from swissknife.utils import Timer


def test_timer_time_measurement():
    with Timer() as timer:
        sleep(1)

    assert round(timer.elapsed, 3) == 1.0
    assert timer.verbose() == '00:00:01'

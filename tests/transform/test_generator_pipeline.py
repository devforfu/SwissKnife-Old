import pytest

from swissknife.transform import GeneratorPipeline


@pytest.mark.parametrize('size,expected', [
    (1, [1]),
    (2, [1, 3]),
    (5, [1, 3, 5, 7, 11]),
    (7, [1, 3, 5, 7, 11, 13, 17]),
    (10, [1, 3, 5, 7, 11, 13, 17, 19, 23, 29])
])
def test_generator_pipeline_transforms_data_from_source(size, expected):
    pipeline = GeneratorPipeline(
        numbers(),
        filter_evens(),
        primes_only()
    )

    generated = [next(pipeline) for _ in range(size)]

    assert generated == expected


def test_generator_pipeline_stops_on_finite_data_source():
    pipeline = GeneratorPipeline(
        numbers(limit=15),
        filter_evens(),
        primes_only()
    )

    sequence = list(pipeline)

    assert sequence == [1, 3, 5, 7, 11, 13]


def test_generator_pipeline_stops_on_infinite_data_source_after_max_iters():
    pipeline = GeneratorPipeline(
        numbers(),
        filter_evens(),
        primes_only(),
        max_iters=5
    )

    sequence = list(pipeline)

    assert sequence == [1, 3, 5]


def numbers(start=1, limit=None):
    """Generator infinitely yielding numbers"""
    while True:
        yield start
        start += 1
        if limit is not None and start >= limit:
            break


def filter_evens():
    """Filters out even numbers from stream."""
    while True:
        number = yield
        is_odd = number % 2 != 0
        yield number if is_odd else None


def primes_only():
    """Keeps only prime numbers in stream."""
    import math
    while True:
        number = yield
        threshold = int(math.floor(math.sqrt(number))) + 1
        is_prime = all([number % x != 0 for x in range(2, threshold)])
        yield number if is_prime else None

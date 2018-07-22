from math import ceil

import pytest
import numpy as np

from swissknife.iterators.arrays import BatchArrayIterator


def test_batch_iterator_has_required_properties():
    """Tests the iterator object has all required public properties."""

    iterator = BatchArrayIterator(np.zeros(32))

    assert not iterator.infinite
    assert not iterator.same_size_batches
    assert iterator.n_batches == 1
    assert iterator.batch_size == 32


@pytest.mark.parametrize('array_size,batch_size', [
    (10, 1),
    (128, 32),
    (100, 32),
    (1000, 10)
])
def test_finite_batch_iterator_yields_expected_number_of_batches(
        array_size,
        batch_size):
    """Tests that the iterator yields an expected number of batches."""

    arr = np.zeros(array_size)
    expected = ceil(array_size / batch_size)

    iterator = BatchArrayIterator(arr, batch_size=batch_size)
    batches = list(iterator)

    assert len(batches) == expected
    assert iterator.n_batches == expected


@pytest.mark.parametrize('array_size,batch_size,last_size', [
    (10, 3, 1),
    (100, 15, 10),
    (63, 32, 31)
])
def test_finite_batch_iterator_generates_non_even_batch_sizes(
        array_size,
        batch_size,
        last_size):
    """
    Tests that the last batch yielded by the iterator has smaller size then
    others if the length of iterated array is not divided by the batch size
    parameter's value.
    """
    arr = np.zeros(array_size)

    iterator = BatchArrayIterator(arr, batch_size=batch_size)
    *_, last = list(iterator)

    assert len(last) == last_size


@pytest.mark.parametrize('array_size,batch_size', [
    (10, 5),
    (100, 25),
    (256, 32)
])
def test_infinite_batch_iterator_indefinitely_yields_batches(
        array_size,
        batch_size):
    """
    Tests that the iterator with infinite=True yields unbounded number of
    iterations.
    """
    arr = np.zeros(array_size)

    iterator = BatchArrayIterator(
        arr, batch_size=batch_size, infinite=True)
    total = iterator.n_batches * 100
    batches = [next(iterator) for _ in range(total)]

    assert len(batches) == total


def test_infinite_batch_iterator_yields_same_size_batches():
    """
    Tests that the iterator with infinite=True and same_size_batches=True
    generates unlimited number of batches with the same size.
    """
    iterator = BatchArrayIterator(
        np.zeros(10),
        batch_size=8,
        infinite=True,
        same_size_batches=True)

    b1 = next(iterator)
    b2 = next(iterator)

    assert iterator.n_batches == 1
    assert len(b1) == len(b2)
    assert np.array_equal(b1, b2)

from math import ceil

import pytest
import numpy as np

from swissknife.iterators.arrays import BatchArrayIterator


def test_batch_iterator_has_required_properties():
    iterator = BatchArrayIterator(np.zeros(32))

    assert not iterator.inifite
    assert not iterator.same_size_batches
    assert iterator.batch_size == 32


@pytest.mark.parametrize('array_size,batch_size', [
    (10, 1),
    (128, 32),
    (100, 32),
    (1000, 10)
])
def test_batch_iterator_creates_valid_number_of_batches(
        array_size,
        batch_size):

    arr = np.zeros(array_size)
    expected = ceil(array_size / batch_size)

    iterator = BatchArrayIterator(arr)
    batches = list(iterator)

    assert len(batches) == expected
    assert iterator.n_batches == expected

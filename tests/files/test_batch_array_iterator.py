import pytest

from swissknife.files import BatchArrayIterator


@pytest.mark.parametrize('n', [0, 1, 101, 1000])
def test_batch_array_iterator_splits_list_into_finite_number_of_batches(n):
    items = list(range(n))
    iterator = BatchArrayIterator(
        array=items,
        infinite=False,
        same_size_batches=False)

    generated = []
    for _ in range(iterator.n_batches):
        generated.extend(next(iterator))

    assert len(generated) == n
    assert generated == items


@pytest.mark.parametrize('n', [0, 1, 101, 1000])
def test_batch_array_iterator_infinitely_iterates_list(n):
    items = list(range(n))
    iterator = BatchArrayIterator(
        array=items,
        batch_size=1,
        infinite=True,
        same_size_batches=False)

    ten_times = n * 10
    generated = []
    for _ in range(ten_times):
        generated.extend(next(iterator))

    assert len(generated) == ten_times
    assert set(generated) == set(items)


@pytest.mark.parametrize('n', [0, 1, 101, 1000])
@pytest.mark.parametrize('batch_size', [1, 10, 32, 50, 101])
def test_batch_array_iterator_generators_same_size_batches(n, batch_size):
    items = list(range(n))
    iterator = BatchArrayIterator(
        array=items,
        batch_size=batch_size,
        infinite=True,
        same_size_batches=True)

    batches = []
    for _ in range(iterator.n_batches):
        batches.append(next(iterator))

    assert all(len(batch) == batch_size for batch in batches)


def test_batch_array_iterator_raises_exception_on_invalid_config():
    with pytest.raises(ValueError):
        BatchArrayIterator([], infinite=False, same_size_batches=True)

import pytest
import numpy as np

from swissknife.utils import BatchGenerator


@pytest.mark.parametrize('array,batch_size,expected', [
    ([1, 2, 3, 4, 5, 6], 1, [[1], [2], [3], [4], [5], [6]]),
    ([1, 2, 3, 4, 5, 6], 2, [[1, 2], [3, 4], [5, 6]]),
    ([1, 2, 3, 4, 5, 6], 6, [[1, 2, 3, 4, 5, 6]])
])
def test_generate_batches_from_1d_array_with_dividable_batch_size(
        array,
        batch_size,
        expected):
    """Tests generating batches from 1D-array of data when the size of array
    is dividable on batch size.
    """
    gen = BatchGenerator(array, batch_size=batch_size)

    actual = gen.drain()

    assert actual == expected


@pytest.mark.parametrize('arrays,batch_size,expected', [
    (
        [[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]],
        3,
        ([1, 2, 3], [10, 20, 30])
    ),
    (
        [[1, 2, 3, 4, 5, 6], list('abcdef')],
        2,
        ([1, 2], ['a', 'b'])
    )
])
def test_generate_batch_from_several_1d_arrays_with_dividable_batch_size(
        arrays,
        batch_size,
        expected):
    """Tests generating batches from several 1D-array. The final result should
    be an array of 1D arrays.
    """
    gen = BatchGenerator(*arrays, batch_size=batch_size)

    first, second = next(gen.flow())

    assert first == expected[0]
    assert second == expected[1]


@pytest.mark.parametrize('array,batch_size,expected', [
    ([1, 2, 3, 4, 5, 6, 7], 2, [[1, 2], [3, 4], [5, 6], [7]]),
    ([1, 2, 3, 4, 5, 6, 7], 3, [[1, 2, 3], [4, 5, 6], [7]]),
    ([1, 2, 3, 4, 5, 6, 7], 4, [[1, 2, 3, 4], [5, 6, 7]])
])
def test_generate_batches_from_1d_array_with_incomplete_batch(
        array,
        batch_size,
        expected):
    """Tests generating batches in case when array size cannot be divided
    without remainder by batch size.
    """
    gen = BatchGenerator(array, batch_size=batch_size)

    actual = gen.drain()

    assert actual == expected


@pytest.mark.parametrize('array,batch_size,expected', [
    (
        [[1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0]],
        2,
        [
            [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
            [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
        ]
    ),
    (
        [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            [[4, 4, 4], [4, 4, 4], [4, 4, 4]],
        ],
        2,
        [
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            ],
            [
                [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                [[4, 4, 4], [4, 4, 4], [4, 4, 4]],
            ]
        ]
    )
])
def test_generate_batches_from_nd_array(
        array,
        batch_size,
        expected):
    """Tests generating batches from n-dimensional arrays"""
    gen = BatchGenerator(array, batch_size=batch_size)

    actual = gen.drain()

    assert actual == expected


def test_batches_generator_returns_numpy_arrays():
    array = list(range(10))
    gen = BatchGenerator(array, batch_size=2, np_arrays=True)

    batches = list(gen.flow())

    assert all(isinstance(b, np.ndarray) for b in batches)

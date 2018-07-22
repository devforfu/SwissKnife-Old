import math
from itertools import cycle

import numpy as np

from . import IteratorMixin


class BatchArrayIterator(IteratorMixin):
    """Iterates an array or several arrays in smaller batches.

    Attributes:
        batch_size: Size of batch.
        infinite: If True, then the iterator doesn't raise StopIteration
            exception when the array is completely traversed but restarts the
            process again.
        same_size_batches: If True and `infinite` attribute is True, then all
            the batches yielded by the iterator have the same size even if
            the total length of the iterated array is not evenly divided by the
            `batch_size`. If the last batch is smaller then `batch_size`, it is
            discarded.

    """
    def __init__(self,
                 array, *arrays,
                 batch_size: int=32,
                 infinite: bool=False,
                 same_size_batches: bool=False):

        if not infinite and same_size_batches:
            raise ValueError('Incompatible configuration: cannot guarantee '
                             'same size of batches when yielding finite '
                             'number of files.')

        arrays = _convert_to_arrays(array, *arrays)

        self.arrays = arrays
        self.batch_size = batch_size
        self.infinite = infinite
        self.same_size_batches = same_size_batches

        self._n = _num_of_batches(arrays, batch_size, same_size_batches)
        self._batch_index = 0
        self._epoch_index = 0

    def next(self):
        if self._batch_index >= self._n:
            if not self.infinite:
                raise StopIteration()
            self._batch_index = 0
            self._epoch_index += 1

        batches = tuple([self._take_next_batch(arr) for arr in self.arrays])
        self._batch_index += 1
        return batches[0] if len(batches) == 1 else batches

    def _take_next_batch(self, array):
        start = self._batch_index * self.batch_size
        end = (self._batch_index + 1) * self.batch_size
        return array[start:end]


def _convert_to_arrays(seq, *seqs):
    sequences = [seq] + list(seqs)
    arrays = [np.asarray(seq) for seq in sequences]
    n = len(arrays[0])
    for arr in arrays[1:]:
        if len(arr) != n:
            raise ValueError('arrays should have the same length')
    return arrays


def _num_of_batches(arrays, batch_size, same_size):
    n = len(arrays[0])
    if same_size:
        return n // batch_size
    return int(math.ceil(n / batch_size))


def _create_iterable(arrays, looped):
    return cycle(arrays) if looped else iter(arrays)

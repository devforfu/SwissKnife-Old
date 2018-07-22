class IteratorMixin:

    def __iter__(self):
        self._batch_index = 0
        self._epoch_index = 0
        return self

    def __next__(self):
        return self.next()

    @property
    def n_batches(self):
        """Returns total number of batches required to iterate all elements of
        the array.
        """
        return self._n

    @property
    def batch_index(self):
        return self._batch_index

    @property
    def epoch_index(self):
        return self._epoch_index

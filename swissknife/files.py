"""
A group of utilities operating with training data files and simplifying files
access.
"""
import math
from pathlib import Path
from itertools import chain, cycle, islice


class FilesStream:
    """Generator that yields file names in cyclic fashion.

    The main purpose of this class is to read files from filesystem in batches,
    instead of reading them all at once. Also, it supports a special processing
    mode which yields files indefinitely in loop. This mode is helpful when
    used for model training purposes during several epochs.
    """
    def __init__(self,
                 folder: str,
                 batch_size: int=32,
                 pattern: str='jpg|jpeg|png|bmp|tiff'):

        self.folder = folder
        self.batch_size = batch_size
        folder = Path(folder)
        extensions = pattern.split('|') if '|' in pattern else [pattern]
        self._files = [
            path.as_posix()
            for path in chain(*[
                folder.glob('*.' + ext)
                for ext in extensions])]

    def __call__(self, infinite=True, same_size_batches=False):
        """Creates generator yielding file paths from folder.

        Args:
            infinite: If True, then generator will be yielding files
                indefinitely. Otherwise, only until the folder is completely
                exhausted.
            same_size_batches: If True, then each batch will have same size.
                In this case, files stream will be "looped", i.e. first files
                will be used to fill the last batch if its size is less then
                required.

        """
        return BatchArrayIterator(self._files,
                                  batch_size=self.batch_size,
                                  infinite=infinite,
                                  same_size_batches=same_size_batches)


class BatchArrayIterator:
    """Iterates array in batch by batch fashion."""

    def __init__(self,
                 array: list,
                 batch_size: int=32,
                 infinite: bool=False,
                 same_size_batches: bool=False):

        self.array = array
        self.batch_size = batch_size
        self.infinite = infinite
        self.same_size_batches = same_size_batches

        if not infinite and same_size_batches:
            raise ValueError('Incompatible configuration: cannot guarantee '
                             'same size of batches when yielding finite '
                             'number of files.')

        n_files = len(self.array)
        if same_size_batches:
            n_batches = n_files // batch_size
        else:
            n_batches = int(math.ceil(n_files / batch_size))
        self._n_batches = n_batches

        looped = infinite and same_size_batches
        self._array = cycle(self.array) if looped else iter(self.array)
        self._count = 0

    @property
    def n_batches(self) -> int:
        return self._n_batches

    def __iter__(self):
        return self

    def __next__(self):
        if not self.infinite and self._count >= self._n_batches:
            raise StopIteration()
        item = self.next()
        self._count += 1
        return item

    def next(self):
        bs = self.batch_size
        if self.infinite and self._count == self._n_batches:
            self._array = iter(self.array)
            self._count = 0
        batch = [x for x in islice(self._array, 0, bs)]
        return batch

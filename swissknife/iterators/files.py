from . import IteratorMixin
from .arrays import BatchArrayIterator
from ..utils import glob


class FilesIterator(IteratorMixin):
    """Iterator that yields files paths.

    The main purpose of this class is to read files from filesystem in batches,
    instead of reading them all at once. Also, it supports a special processing
    mode which yields files indefinitely in loop. This mode is helpful when
    used for model training purposes during several epochs.
    """
    def __init__(self,
                 folder: str,
                 pattern: str,
                 batch_size: int=32,
                 infinite: bool=False,
                 same_size_batches: bool=False):

        self.folder = str(folder)
        self.pattern = pattern
        self.infinite = infinite
        self.same_size_batches = same_size_batches
        self.batch_size = batch_size

        extensions = pattern.split('|') if '|' in pattern else [pattern]
        files = list(glob(self.folder, extensions))

        self._extensions = extensions
        self._files = files
        self._n = len(self._files)
        self._iter = BatchArrayIterator(
            self._files, batch_size=batch_size,
            infinite=infinite, same_size_batches=same_size_batches)

    @property
    def batch_index(self):
        return self._iter.batch_index

    @property
    def epoch_index(self):
        return self._iter.epoch_index

    @property
    def extensions(self):
        return self._extensions

    def next(self):
        return next(self._iter)

import time
import random
from io import StringIO
from pathlib import Path
from timeit import default_timer
from itertools import combinations

import numpy as np


def random_string(size: int, domain: str='abcdef1234567890') -> str:
    """Creates a random string using provided set of symbols."""
    return ''.join([random.choice(domain) for _ in range(size)])


def disjoint(*seq) -> bool:
    """Checks if each pair of provided collections doesn't have any elements
    in common."""

    def disjoint_pair(a, b):
        return set(a).isdisjoint(set(b))

    if len(seq) == 1:
        return False

    if len(seq) == 2:
        return disjoint_pair(*seq)

    for fst, snd in combinations(seq, 2):
        if not disjoint_pair(fst, snd):
            return False

    return True


def get_values(keys: list, d: dict, replace_missing: bool=True):
    """Gets a list of keys from dictionary in specific order."""
    if replace_missing:
        return [d.get(key) for key in keys]
    return [d[key] for key in keys if key in d]


def without_nones(seq):
    """Checks if collection doesn't have any None value."""
    return all([item is not None for item in seq])


def has_extension(filename, exts=None):
    """Checks if file has extension from list. If extensions list is empty,
    then function returns True as default value.
    """
    if exts is None or not exts:
        return True
    file_ext = Path(filename).suffix
    for ext in exts.split('|'):
        if file_ext == ext:
            return True
    return False


def n_files(folder: str, exts=None):
    """Returns number of direct children in folder which themselves are not
    folders."""

    counter = 0
    for filename in Path(folder).glob('*.*'):
        if filename.is_dir():
            continue
        if has_extension(filename, exts):
            counter += 1

    return counter


class StringBuffer:
    """Context manager helping to redirect output into string buffer and
    get captured values.
    """

    def __init__(self):
        self.buffer = StringIO()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.buffer.seek(0)

    def __getattr__(self, item):
        delegate = self.__dict__['buffer']
        if item not in self.__dict__:
            if item in dir(delegate):
                return getattr(delegate, item)
        raise AttributeError(item)

    @property
    def captured(self):
        return self.buffer.getvalue()

    @property
    def lines(self):
        return self.captured.split()


class Timer:
    """Simple util to measure execution time.

    Examples
    --------
    >>> import time
    >>> with Timer() as timer:
    ...     time.sleep(1)
    >>> print(timer)
    00:00:01
    """
    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = default_timer() - self.start

    def __str__(self):
        return self.verbose()

    def verbose(self):
        if self.elapsed is None:
            return '<not-measured>'
        return time.strftime('%H:%M:%S', time.gmtime(self.elapsed))


class MockImage:

    def __init__(self, grey=True, width=100, height=100, rate=0.5):
        self.grey = grey
        self.width = width
        self.height = height
        self.rate = rate

    def random_pixel(self, p):
        if np.random.binomial(1, p):
            return 1 if self.grey else [1., 1., 1.]
        if self.grey:
            return 0
        else:
            return [random.random() for _ in range(3)]

    def salt(self):
        width, height = self.width, self.height
        pixels = np.array([
            self.random_pixel(self.rate) for _ in range(width * height)])
        img = pixels.reshape((width, height))
        return img

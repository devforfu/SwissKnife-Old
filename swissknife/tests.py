import random
from itertools import combinations


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

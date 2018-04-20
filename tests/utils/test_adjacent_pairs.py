import pytest

from swissknife.utils import adjacent_pairs


@pytest.mark.parametrize('seq,expected', [
    ([1], []),
    ([1, 2, 3, 4, 5], [(1, 2), (2, 3), (3, 4), (4, 5)])
])
def test_adjacent_pairs_applying_to_list_of_integers(seq, expected):
    actual = list(adjacent_pairs(seq))

    assert actual == expected

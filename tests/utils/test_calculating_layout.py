import pytest

from swissknife.utils import calculate_layout


@pytest.mark.parametrize('axes,rows,cols,expected', [
    (10, 2,    None, (2, 5)),
    (10, None, 2,    (5, 2)),
    (3,  3,    None, (3, 1)),
    (7,  None, 2,    (4, 2))
])
def test_calculating_layout_with_valid_args_returns_valid_shape(
    axes, rows, cols, expected
):
    """
    Tests returning correct number of rows and columns.
    """
    actual = calculate_layout(num_axes=axes, n_rows=rows, n_cols=cols)

    assert actual == expected


def test_calculating_layout_with_invalid_args_raises_error():
    """
    Tests raising error if user provides both rows and columns.
    """
    with pytest.raises(ValueError):
        calculate_layout(1, 1, 1)
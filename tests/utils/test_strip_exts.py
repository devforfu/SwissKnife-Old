import pytest

from swissknife.utils import strip_exts


@pytest.mark.parametrize('filename,ext,output', [
    ('image.jpeg', 'jpeg', 'image'),
    ('image.of.dog.png', 'png', 'image.of.dog'),
    ('.folder', 'folder', '.folder'),
    ('labels.csv.zip', 'zip', 'labels.csv')
])
def test_stripping_single_extension_from_file(filename, ext, output):
    """Tests removing a single extension from file while skipping files
    starting with dot.
    """
    actual = strip_exts(filename, exts=ext)

    assert actual == output


@pytest.mark.parametrize('filename,output', [
    ('image.jpeg', 'image'),
    ('image.of.dog.png', 'image'),
    ('.folder', '.folder'),
    ('labels.csv.zip', 'labels')
])
def test_stripping_all_symbols_coming_after_dot(filename, output):
    """Tests removing all characters coming after dot in filename excluding
    files starting with dot.
    """
    actual = strip_exts(filename, strip_all=True)

    assert actual == output

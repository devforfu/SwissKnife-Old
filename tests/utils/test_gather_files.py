import pytest
from pathlib import Path

from swissknife.utils import gather_files


def test_gathering_files_with_extensions_from_tree(dir_pair):
    """Tests collecting files with provided extensions from the source folder
    without deleting source folder.
    """
    src, dst = dir_pair

    gather_files(src=src, dst=dst, exts='log|txt|out')

    expected = read_all(src)
    actual = read_all(dst)
    assert Path(src).exists()
    assert expected == actual


def test_gathering_files_with_deleting_original_tree(dir_pair):
    """Tests deleting original folder after moving files to another one."""
    src, dst = dir_pair

    gather_files(src=src, dst=dst, exts='log', delete_source=True)

    assert not Path(src).exists()


def test_gathering_files_returns_list_of_copied_paths(dir_pair):
    """Tests returning a list of paths to gathered files."""
    src, dst = dir_pair

    paths = gather_files(src=src, dst=dst, exts='log|txt|out')

    expected = read_all(src)
    assert len(paths) == len(expected)
    assert all(Path(p).exists() for p in paths)


@pytest.fixture
def dir_pair(tmpdir):
    content = 'content'
    src = tmpdir.mkdir('src')
    dst = tmpdir.mkdir('dst')
    files = (
        src.join('results.log'),
        src.join('failure.out'),
        src.mkdir('sub').join('log.txt'))
    for file in files:
        file.write(content)
    return str(src), str(dst)


def read_all(path):
    return [open(file.as_posix()).read() for file in Path(path).glob('**/*.*')]

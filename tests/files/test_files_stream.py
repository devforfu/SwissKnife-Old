import pytest

from swissknife.tests import random_string
from swissknife.files import FilesStream


@pytest.mark.parametrize('n_files', [32, 100, 1000])
def test_files_stream_yields_all_files_from_folder(n_files, make_files):
    root, files = make_files(n_files)
    stream = FilesStream(root, pattern='mock')

    generated = []
    for batch in stream(infinite=False):
        generated.extend(batch)

    assert sorted(generated) == sorted(files)


@pytest.fixture
def make_files(tmpdir):

    def files_maker(size=100):
        root = tmpdir.mkdir('folder')
        files = []
        for _ in range(size):
            filename = '%s.mock' % random_string(size=20)
            f = root.join(filename)
            f.write('content')
            files.append(str(f))
        return str(root), files

    return files_maker

import math
import pytest
from pathlib import Path

from swissknife.utils import split_dataset_files
from swissknife.tests import random_string


@pytest.mark.parametrize('valid_size', [0.2, 0.5, 0.9])
def test_splitting_files_from_folders_into_train_and_validation_sets(
        tmpdir,
        make_dataset,
        valid_size):
    """Tests splitting original set of files into two training and validation
    subsets with predefined size of validation set.
    """
    n_files = 100
    output = tmpdir.mkdir('output')
    folder, classes = make_dataset(n_files)

    files = split_dataset_files(dataset_dir=folder,
                                output_dir=str(output),
                                classes=classes,
                                holdout=False,
                                extensions='mock',
                                data_split=valid_size)

    assert 'valid' in files
    assert 'train' in files
    assert len(files['valid']) == math.ceil(n_files*valid_size)
    assert len(files['train']) == math.ceil(n_files*(1 - valid_size))


def test_splitting_files_from_folders_into_train_validation_holdout_sets():
    pass


@pytest.fixture
def make_dataset(tmpdir):

    def files_maker(size=100, proportion=0.5):
        root = tmpdir.mkdir('dataset')
        x, y = [], [0]*int(size*proportion) + [1]*int(size*(1 - proportion))
        for filename in _make_files(size):
            p = root.join(filename)
            p.write('content')
            x.append(Path(str(p)).stem)
        return str(root), dict(zip(x, y))

    return files_maker


def _make_files(n, ext='mock'):
    for _ in range(n):
        yield '%s.%s' % (random_string(size=20), ext)

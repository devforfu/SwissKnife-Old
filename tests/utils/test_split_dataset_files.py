import pytest
from pathlib import Path

from swissknife.utils import split_dataset_files
from swissknife.tests import random_string, disjoint, get_values, without_nones


@pytest.mark.parametrize('valid_size,n_valid,n_train', [
    [0.2, 20, 80],
    [0.5, 50, 50],
    [0.9, 90, 10]
])
def test_splitting_files_from_folders_into_train_and_validation_sets(
        tmpdir,
        make_dataset,
        valid_size,
        n_valid,
        n_train):
    """Tests splitting original set of files into two training and validation
    subsets with predefined size of validation set.
    """
    output = tmpdir.mkdir('output')
    folder, classes = make_dataset()

    files = split_dataset_files(dataset_dir=folder,
                                output_dir=str(output),
                                classes=classes,
                                holdout=False,
                                extensions='mock',
                                data_split=valid_size)

    valid, train = get_values(['valid', 'train'], files)
    assert valid is not None and train is not None
    assert len(valid) == n_valid
    assert len(train) == n_train
    assert disjoint(valid, train)


@pytest.mark.parametrize('data_split,counts', [
    [[0.6, 0.20, 0.20], (60, 20, 20)],
    [[0.5, 0.25, 0.25], (50, 25, 25)]
])
def test_splitting_files_from_folders_into_train_validation_holdout_sets(
        tmpdir,
        make_dataset,
        data_split,
        counts):
    """Tests splitting original set of files into three non-intersected
    categories: training, validation and holdout sets.
    """
    output = tmpdir.mkdir('output')
    folder, classes = make_dataset()

    files = split_dataset_files(dataset_dir=folder,
                                output_dir=str(output),
                                classes=classes,
                                holdout=True,
                                extensions='mock',
                                data_split=data_split)

    splits = get_values(['train', 'valid', 'holdout'], files)
    assert without_nones(splits)
    assert all([len(split) == count for split, count in zip(splits, counts)])
    assert disjoint(*splits)


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

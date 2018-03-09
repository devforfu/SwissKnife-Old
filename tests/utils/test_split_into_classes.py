from pathlib import Path
from itertools import chain

import pytest

from swissknife.tests import random_string, n_files
from swissknife.utils import split_into_class_folders


def test_copying_files_from_single_folder_into_class_based_subfolders(
        tmpdir,
        make_dataset):
    """Tests copying files from single folder into new directory tree where
    each subfolder contains files of a single class.
    """
    output_dir = str(tmpdir.mkdir('output'))
    n_files_per_class, n_classes = 10, 3
    dataset_dir, classes = make_dataset(n_files_per_class)

    dirs = split_into_class_folders(dataset_dir=dataset_dir,
                                    output_dir=output_dir,
                                    classes=classes,
                                    copy=True)
    folders = list(dirs.values())

    assert len(dirs) == n_classes
    assert all(n_files(folder) == n_files_per_class for folder in folders)
    assert n_files(dataset_dir) == n_classes * n_files_per_class


def test_moving_files_from_single_folder_into_class_based_subfolders(
        tmpdir,
        make_dataset):
    """Tests moving original files from single folder into new directory tree
    where each subfolder contains files of a single class. In this case,
    original files are not kept intact.
    """
    output_dir = str(tmpdir.mkdir('output'))
    n_files_per_class, n_classes = 10, 3
    dataset_dir, classes = make_dataset(n_files_per_class)

    dirs = split_into_class_folders(dataset_dir=dataset_dir,
                                    output_dir=output_dir,
                                    classes=classes,
                                    copy=False)
    folders = list(dirs.values())

    assert len(dirs) == n_classes
    assert all(n_files(folder) == n_files_per_class for folder in folders)
    assert n_files(dataset_dir) == 0


@pytest.fixture
def make_dataset(tmpdir):

    def files_maker(n_files_per_class=20, classes=None):
        root = tmpdir.mkdir('dataset')
        classes = classes or ['dog', 'cat', 'snake']
        x, y = [], list(chain(*[[c]*n_files_per_class for c in classes]))
        for class_name in classes:
            for filename in _make_files(n_files_per_class, class_name):
                p = root.join(filename)
                p.write(class_name)
                x.append(Path(str(p)).stem)
        return str(root), dict(zip(x, y))

    return files_maker


def _make_files(size, prefix, ext='mock'):
    for _ in range(size):
        yield '%s_%s.%s' % (prefix, random_string(size=size), ext)

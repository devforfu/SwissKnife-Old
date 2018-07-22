from textwrap import dedent

import pytest
import numpy as np

from swissknife.sources import LabelledImagesSource


def test_source_initialization_with_file(source):
    assert source.n_classes == 3
    assert np.array_equal(source.classes, [0, 1, 2])
    assert np.array_equal(source.verbose_classes, ['blue', 'green', 'red'])


def test_source_maps_verbose_classes_to_numerical_labels(source):
    labels = source.to_label(['blue', 'green', 'red'])

    assert np.array_equal(labels, [0, 1, 2])


def test_source_maps_numerical_labels_to_verbose_classes(source):
    classes = source.to_verbose([0, 1, 2])

    assert np.array_equal(classes, ['blue', 'green', 'red'])


@pytest.fixture
def labels_file(tmpdir):
    folder = tmpdir.mkdir('dataset')
    file = folder.join('labels.csv')
    file.write(dedent("""
    id,class
    1,red
    2,red
    3,green
    4,blue
    5,green
    """))
    return file


@pytest.fixture
def source(labels_file):
    return LabelledImagesSource(
        labels_from='file', filename=labels_file,
        id_column='id', label_column='class')

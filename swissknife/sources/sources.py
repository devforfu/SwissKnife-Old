import abc
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from ..utils import read_labels


_labels_sources = {}


class LabelledImagesSource:

    def __new__(cls,
                labels_from: str='file',
                **kwargs):
        if issubclass(cls, LabelledImagesSource):
            cls = get_dataset(labels_from)
        return object.__new__(cls)

    def __init__(self, **kwargs):
        pass

    @property
    def n_classes(self) -> int:
        """"""

    @property
    def classes(self) -> list:
        """"""

    @property
    def verbose_classes(self) -> list:
        """"""

    # @abc.abstractmethod
    # def frequency_histogram(self, bins=None, with_labels=None):
    #     """"""
    #
    # @abc.abstractmethod
    # def build(self):
    #     """"""


class _LabelsFromFileDataset(LabelledImagesSource):

    def __init__(self,
                 filename: str,
                 label_column: str,
                 id_column: str='id',
                 **kwargs):

        super().__init__(**kwargs)

        self.filename = str(filename)
        self.label_column = label_column
        self.id_column = id_column

        self._uid_to_verbose = None
        self._classes = None
        self._binarizer = None
        self._verbose_classes = None
        self._verbose_to_label = None
        self._label_to_verbose = None
        self._one_hot = None

        self.init()

    def init(self):
        uid_to_verbose = read_labels(
            filename=self.filename,
            class_column=self.label_column,
            skip_header=self.id_column)

        string_classes = list(uid_to_verbose.values())
        binarizer = LabelBinarizer()
        one_hot = binarizer.fit_transform(string_classes)
        numerical_classes = one_hot.argmax(axis=1)

        self._uid_to_verbose = uid_to_verbose
        self._classes = np.unique(numerical_classes)
        self._binarizer = binarizer
        self._verbose_classes = np.unique(string_classes)
        self._verbose_to_label = dict(zip(string_classes, numerical_classes))
        self._label_to_verbose = {
            v: k for k, v in self._verbose_to_label.items()}
        self._one_hot = one_hot

    @property
    def n_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def verbose_classes(self):
        return self._verbose_classes

    def to_label(self, names):
        return np.array([self._verbose_to_label[name] for name in names])

    def to_verbose(self, labels):
        return np.array([self._label_to_verbose[label] for label in labels])


def register_source(name, cls):
    global _labels_sources
    _labels_sources[name] = cls


def get_dataset(name):
    if name not in _labels_sources:
        raise ValueError('dispatcher with name \'%s\' is not found' % name)
    return _labels_sources[name]


register_source('file', _LabelsFromFileDataset)

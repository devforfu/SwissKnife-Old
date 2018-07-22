import abc
from . import IteratorMixin
from ..utils import read_labels


_labels_sources = {}


class LabelledImagesDataset(IteratorMixin, metaclass=abc.ABCMeta):

    def __new__(cls, labels_from='file', **kwargs):
        if isinstance(cls, LabelledImagesDataset):
            cls = get_dataset(labels_from)
        return object.__new__(cls)

    @abc.abstractmethod
    @property
    def n_classes(self) -> int:
        """"""

    @abc.abstractmethod
    def frequency_histogram(self, bins=None, with_labels=None):
        """"""

    @abc.abstractmethod
    def build(self):
        """"""


class _LabelsFromFileDataset(LabelledImagesDataset):

    def __init__(self,
                 filename: str,
                 label_column: str,
                 id_column: str='id'):

        self.filename = filename
        self.label_column = label_column
        self.id_column = id_column

    def build(self):
        classes = read_labels(self.filename, self.label_column, self.id_column)





def register_source(name, cls):
    global _labels_sources
    _labels_sources[name] = cls


def get_dataset(name):
    if name not in _labels_sources:
        raise ValueError('dispatcher with name \'%s\' is not found' % name)
    return _labels_sources[name]


register_source('file', _LabelsFromFileDataset)

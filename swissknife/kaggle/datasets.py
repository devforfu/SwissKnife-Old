"""
Datasets loading and processing.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelBinarizer

from ..images import FilesStream, FallbackImageLoader


class KaggleClassifiedImagesSource:
    """Class that helps generate (x, y) samples from images data provided in
    Kaggle-specific format.

    Usually each training dataset's image has a unique filename that could be
    one-to-one mapped onto class that is represented on that image. And, the
    mapping from filename to class name is stored within separate CSV file.

    For example, consider the following folder with images:

        /dataset
            /train
              - 08ab870.jpeg
              - 73c913f.jpeg
              - fe10512.jpeg
              ...
              - cca89cf.jpeg
            /test
              ...

    Then, file with labels should have the following structure:

        +---------+-------+
        | UUID    | Class |
        +---------+-------+
        | 08ab870 | dog   |
        | 73c913f | cat   |
        | fe1051  | dog   |
            ...     ...
        | cca89cf | fox   |
        +---------+-------+

    Attributes:
        labels_path: Path to file with mapping from file names to classes.
        binarizer: Instance of sklearn.LabelBinarizer class used to convert
            string classes representation into one-hot encoded vectors.
        name_to_label: Mapping from string representation of class into its
            numerical value.
        identifier_to_label: Mapping from unique filename identifier into
            its class's numerical value.

    """
    def __init__(self,
                 labels_path: str,
                 label_column: str,
                 load_image=None):

        if load_image is None:
            load_image = FallbackImageLoader()

        self.labels_path = labels_path
        self.label_column = label_column
        self.load_image = load_image
        self.binarizer = None
        self.name_to_label = None
        self.identifier_to_label = None
        classes = self.read_labels(labels_path, label_column=label_column)
        self.build(classes)

    def __call__(self, *args, **kwargs):
        return self.flow(*args, **kwargs)

    @property
    def n_classes(self) -> int:
        """Returns total number of classes represented by dataset images."""
        return len(self.binarizer.classes_)

    @staticmethod
    def read_labels(filename: str, label_column: str, id_column: str='id'):
        """Reads file with class labels.

        Args:
            filename: Path to file with labels.
            id_column: Column with unique image identifier.
            label_column: Column with verbose classes names.

        Returns:
            labels: The mapping from ID to verbose label.

        """
        if not Path(filename).exists():
            raise ValueError('labels file is not found: %s' % filename)
        with open(filename) as fp:
            reader = csv.DictReader(fp)
            try:
                labels = {row[id_column]: row[label_column] for row in reader}
            except KeyError:
                raise ValueError(
                    "please check your CSV file: '%s' and/or '%s' "
                    "column was not found" % (id_column, label_column))
            else:
                return labels

    def build(self, classes: dict):
        """Fits labels one-hot encoder and maps verbose image classes labels
        onto one-hot encoded vectors.

        Args:
            classes: A dictionary with classes. Should be a mapping from file
                ID to string with class name.

        """
        identifiers = [Path(path).stem for path in classes.keys()]
        string_labels = list(classes.values())
        binarizer = LabelBinarizer()
        one_hot = binarizer.fit_transform(string_labels)

        name_to_label = {
            binarizer.classes_[vec.argmax()]: vec for vec in one_hot}

        identifier_to_label = {
            uid: name_to_label[classes[uid]] for uid in identifiers}

        self.binarizer = binarizer
        self.name_to_label = name_to_label
        self.identifier_to_label = identifier_to_label

    def get_class_name(self, image_path) -> str:
        """Returns class represented by image using file name as unique ID."""

        uid = Path(image_path).stem
        if uid not in self.identifier_to_label:
            raise ValueError('The file with ID \'%s\' is not present in '
                             'mapping. Probably it was taken from different '
                             'dataset or file with labels which was used to '
                             'build mapping is incomplete.')
        return self.identifier_to_label[uid]

    def one_hot_to_verbose(self, vec) -> str:
        """Converts one-hot encoded vector into verbose class name."""

        try:
            vec = column_or_1d(vec, warn=False)
        except ValueError:
            raise ValueError('Input value should be a 1D array or column, '
                             'but has shape %s instead' % vec.shape)

        if len(vec) > self.n_classes:
            raise ValueError('Input array is longer then number of classes '
                             'in dataset: %d > %d' %
                             (len(vec), self.n_classes))

        return self.binarizer.classes_[vec.argmax()]

    def integer_to_verbose(self, label: int) -> str:
        """Converts class represented with integer label into verbose class
        name.
        """
        return self.binarizer.classes_[label]

    def flow(self, folder: str, target_size: tuple, batch_size: int,
             infinite: bool=False):
        """Creates a generator that iterates through directory with files and
        reads images from folder in batches, converting them into (x, y) pairs
        for model's training.

        Args:
            folder: Directory with labelled files.
            target_size: Shape of generated image samples.
            batch_size: Size of batch.
            infinite: If True, then created generator will infinitely iterate
                through available files. Otherwise, it will stop as soon as all
                samples visited.

        Returns:
            TrainingSamplesIterator: Generator-like object yielding training
                pairs in batches.

        """
        return TrainingSamplesIterator(
            self, folder, target_size, batch_size, infinite)


class TrainingSamplesIterator:
    """Supplementary class iterating through training samples."""

    def __init__(self, delegate, folder, target_size, batch_size,
                 infinite=False):

        self.delegate = delegate
        self.folder = folder
        self.target_size = target_size
        self.batch_size = batch_size
        self.infinite = infinite
        stream = FilesStream(self.folder, batch_size=self.batch_size)
        self._source = stream(infinite=infinite, same_size_batches=infinite)

    @property
    def steps_per_epoch(self):
        return self._source.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        delegate = self.delegate
        paths = next(self._source)
        arrays = np.asarray([
            delegate.load_image(path, target_size=self.target_size)
            for path in paths])
        targets = np.asarray([delegate.get_class_name(path) for path in paths])
        return arrays, targets


class KaggleTestImagesIterator:
    """Class that helps generate test (un-labelled) samples from images data
    provided in Kaggle-specific format.

    The class is similar to KaggleClassifiedImagesSource, but instead loads
    test or unsupervised files. It keeps file names to simplify generation of
    Kaggle submission file. Also, the class does not create iterator on demand,
    but is iterator itself yielding images (and their identifiers if required).
    """
    def __init__(self,
                 test_folder: str,
                 target_size: tuple,
                 batch_size: int=32,
                 with_identifiers: bool=False,
                 load_image=None):

        if load_image is None:
            load_image = FallbackImageLoader()

        self.test_folder = test_folder
        self.target_size = target_size
        self.batch_size = batch_size
        self.with_identifiers = with_identifiers
        self.load_image = load_image
        self.identifiers = []
        stream = FilesStream(
            self.test_folder, batch_size=self.batch_size)
        self._source = stream(infinite=False, same_size_batches=False)

    @property
    def n_batches(self):
        return self._source.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        batch = next(self._source)
        identifiers = [Path(filename).stem for filename in batch]
        images = np.asarray([
            self.load_image(path, self.target_size) for path in batch])
        self.identifiers.extend(identifiers)
        result = (images, identifiers) if self.with_identifiers else images
        return result

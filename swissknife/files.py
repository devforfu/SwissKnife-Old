"""
A group of utilities operating with training data files and simplifying files
access.
"""
import os
import re
import csv
import math
import shutil
import logging
from pathlib import Path
from os.path import join, exists
from itertools import chain, cycle, islice


class FilesStream:
    """Generator that yields file names in cyclic fashion.

    The main purpose of this class is to read files from filesystem in batches,
    instead of reading them all at once. Also, it supports a special processing
    mode which yields files indefinitely in loop. This mode is helpful when
    used for model training purposes during several epochs.
    """
    def __init__(self,
                 folder: str,
                 batch_size: int=32,
                 pattern: str='jpg|jpeg|png|bmp|tiff'):

        self.folder = folder
        self.batch_size = batch_size
        folder = Path(folder)
        extensions = pattern.split('|') if '|' in pattern else [pattern]
        self._files = [
            path.as_posix()
            for path in chain(*[
                folder.glob('*.' + ext)
                for ext in extensions])]

    def __call__(self, infinite=True, same_size_batches=False):
        """Creates generator yielding file paths from folder.

        Args:
            infinite: If True, then generator will be yielding files
                indefinitely. Otherwise, only until the folder is completely
                exhausted.
            same_size_batches: If True, then each batch will have same size.
                In this case, files stream will be "looped", i.e. first files
                will be used to fill the last batch if its size is less then
                required.

        """
        return BatchArrayIterator(self._files,
                                  batch_size=self.batch_size,
                                  infinite=infinite,
                                  same_size_batches=same_size_batches)


class BatchArrayIterator:
    """Iterates array in batch by batch fashion."""

    def __init__(self,
                 array: list,
                 batch_size: int=32,
                 infinite: bool=False,
                 same_size_batches: bool=False):

        self.array = array
        self.batch_size = batch_size
        self.infinite = infinite
        self.same_size_batches = same_size_batches

        if not infinite and same_size_batches:
            raise ValueError('Incompatible configuration: cannot guarantee '
                             'same size of batches when yielding finite '
                             'number of files.')

        n_files = len(self.array)
        if same_size_batches:
            n_batches = n_files // batch_size
        else:
            n_batches = int(math.ceil(n_files / batch_size))
        self._n_batches = n_batches

        looped = infinite and same_size_batches
        self._array = cycle(self.array) if looped else iter(self.array)
        self._count = 0

    @property
    def n_batches(self) -> int:
        return self._n_batches

    def __iter__(self):
        return self

    def __next__(self):
        if not self.infinite and self._count >= self._n_batches:
            raise StopIteration()
        item = self.next()
        self._count += 1
        return item

    def next(self):
        bs = self.batch_size
        if self.infinite and self._count == self._n_batches:
            self._array = iter(self.array)
            self._count = 0
        batch = [x for x in islice(self._array, 0, bs)]
        return batch


class SavingFolder:
    """Tool preparing filesystem to save model and restoring trained model and
    training process history (if available).

    The main purpose of the class is to provide set of model saving paths,
    like directory with checkpoints, weights files, training history, etc.
    helpful to save and load previously saved models.

    Attributes:
        models_root: Directory with models's subfolders.
        model_name: Model's subfolder name.
        model_path: Absolute path to saved model's file.
        model_ext: Extension for model saving file.
        history_path: Absolute path to saved model's training history.
        history_ext: Extension for history file.
        env_var: Environment variable with path to directory with models.
        log: Logger instance.

    """
    def __init__(self, model_name: str, models_root: str=None,
                 env_var='MODELS_DIR', model_ext='.h5', history_ext='.csv',
                 log=None):

        if models_root is None:
            env_value = os.environ.get(env_var, '')
            models_root = os.path.expandvars(os.path.expanduser(env_value))

        self.models_root = models_root
        self.model_name = model_name
        self.model_dir = join(self.models_root, self.model_name)
        self.model_path = join(self.model_dir, self.model_name + model_ext)
        self.history_path = join(self.model_dir, self.model_name + history_ext)
        self.log = log or logging.getLogger()

    def create_model_dir(self, ask_on_rewrite: bool=True) -> bool:
        """Creates a directory for trained model to save weights, checkpoints
        and training history.

        Args:
            ask_on_rewrite: If True, then console prompt will be shown with
                confirmation request in case when model directory with same
                name already exist. Otherwise, the directory will be re-written
                with new training results.

        Returns:
            bool: True if folder was (re)created, False - otherwise.

        """
        log = self.log
        model_dir = self.model_dir
        if not exists(model_dir):
            os.mkdir(model_dir)
            log.info('Model folder is created: %s', model_dir)

        elif not ask_on_rewrite:
            log.info('Deleting old model\'s folder...')
            shutil.rmtree(model_dir)
            os.mkdir(model_dir)
            log.info('Model folder is re-created: %s', model_dir)

        else:
            log.warning('Model folder already exists. It will be deleted')
            while True:
                log.info('Proceed? [y/n]')
                response = input().lower().strip()
                if response in ('y', 'n'):
                    if response == 'n':
                        return False
                    else:
                        shutil.rmtree(model_dir)
                        os.mkdir(model_dir)
                        log.info('New folder created: %s', model_dir)
                        break
                log.info('Please choose Y or N')

        return True

    def load_model(self, loader='keras'):
        """Loads model from model's training output folder.

        Args:
            loader: Library which model loader should be used to restore model.
                The only value supported is 'keras'.

        Returns:
            model: The model restored from file.

        Raises:
            ImportError: If model loading library is not installed.

        """
        if loader != 'keras':
            raise ValueError('Loader \'%s\' is not supported' % loader)

        try:
            from keras.models import load_model
        except ImportError:
            raise ImportError(
                'Cannot load Keras model without library installed')

        return load_model(self.model_path)

    def best_checkpoint(self, regex='^[\d\w]+(\d.\d+){1}.hdf5$'):
        """Returns best model checkpoint if any present.

        By default, the method expects that each checkpoint file name has a
        specific format which should contain validation loss value.

        Args:
            regex: Regular expression which is used to find checkpoint files in
                model's directory and to match validation loss value in
                filename's string.

        Returns:
            path: The path to checkpoint with the lowest validation loss or
                None if file is not found.

        """
        regex = re.compile(regex)
        best_score, best_model = float('inf'), None
        for filename in os.listdir(self.model_dir):
            match = regex.match(filename)
            if match is None:
                continue
            try:
                val_loss = float(match.group(1))
            except (ValueError, TypeError):
                continue
            else:
                if val_loss < best_score:
                    best_score = val_loss
                    best_model = filename

        if best_model is None:
            self.log.warning(
                'Cannot load checkpoint: there are no files matching '
                'regex in model\'s directory')
            return None

        return join(self.model_dir, best_model)

    def load_history(self, csv_params=None, as_dataframe=False):
        """Reads training history from CSV file.

        Args:
            csv_params: Dictionary with parameters passed into CSV reader
                defining separation symbols, quote symbols, etc.

            as_dataframe: If True, then pandas.DataFrame object will be
                returned. Otherwise, the result is provided as list of
                dictionaries.

        Returns:
            list or dataframe: Training history read from file.

        Raises:
            ImportError: If `as_dataframe` parameter is True and pandas
                library is not installed.

        """
        csv_params = csv_params or {}

        if as_dataframe:
            try:
                import pandas
            except ImportError:
                raise ImportError(
                    'Cannot return dataframe without pandas library installed')
            else:
                return pandas.read_csv(self.history_path, **csv_params)

        history = []
        with open(self.history_path) as fp:
            reader = csv.reader(fp, **csv_params)
            header = next(reader)
            for row in reader:
                record = dict(zip(header, row))
                history.append(record)

        return history

    @property
    def model_files(self, abspath=False):
        """Returns a list of files in model folder."""

        filenames = os.listdir(self.model_dir)
        if not abspath:
            return filenames
        return [os.path.join(self.model_dir, name) for name in filenames]

"""
Miscellaneous tools to manage dataset files and prepare data for training.
"""
import math
import shutil
import fnmatch
from pathlib import Path
from itertools import islice
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def calculate_layout(num_axes, n_rows=None, n_cols=None):
    """Calculates number of rows/columns required to fit `num_axes` plots
    onto figure if specific number of columns/rows is specified.
    """
    if n_rows is not None and n_cols is not None:
        raise ValueError(
            'cannot derive number of rows/columns if both values provided')
    if n_rows is None and n_cols is None:
        n_cols = 2
    if n_rows is None:
        n_rows = max(1, math.ceil(num_axes / n_cols))
    else:
        n_cols = max(1, math.ceil(num_axes / n_rows))
    return n_rows, n_cols


def gather_files(src: str,
                 dst: str,
                 exts: str='txt|log|pdf|zip|png|json|csv',
                 rewrite: bool=False,
                 delete_source: bool=False):
    """
    Gathers files from directory tree with specific extensions into new folder.

    Args:
        src: Directory with original files.
        dst: Directory where to copy files.
        exts: Pipe-separated string with extensions.
        rewrite: If True and output directory contains files, then it will
            be rewritten.
        delete_source: If True, then ALL files from the original tree will
            be deleted.

    Returns:
        copied_files: List of copied files.

    """
    src, dst = Path(src), Path(dst)

    if '|' not in exts:
        ext_list = [exts]
    else:
        ext_list = exts.split('|')

    if dst.exists():
        for _ in dst.iterdir():
            if not rewrite:
                raise OSError('folder exists: %s' % dst)
            break
        shutil.rmtree(dst.as_posix())

    copied_files = []
    for old_path in src.glob('**/*.*'):
        for ext in ext_list:
            if fnmatch.fnmatch(old_path.as_posix(), '*.' + ext):
                relative_src = old_path.relative_to(src)
                new_path = dst.joinpath(relative_src)
                parent = new_path.parent
                if not parent.exists():
                    parent.mkdir(parents=True)
                shutil.copy(old_path.as_posix(), new_path.as_posix())
                copied_files.append(new_path.as_posix())
                break

    if delete_source:
        shutil.rmtree(src.as_posix())

    return copied_files


def split_dataset_files(dataset_dir: str,
                        output_dir: str,
                        classes: dict,
                        valid_size=0.2,
                        holdout_size=None,
                        rewrite: bool=False,
                        extensions: str='jpg|jpeg|png',
                        random_state: int=None):
    """
    Separates training files into training and validation folders.

    This function is intended to behave in a similar way like
    `test_train_split` function from `sklearn` package, but instead of
    returning array of indexes it returns file paths.

    Args:
        dataset_dir: Path to directory with original dataset files.
        output_dir: Path where to save prepared catalogues with files.
        classes: Dictionary that maps file name to class represented by
            that file.
        valid_size: Validation subset size (as a fraction of original dataset)
        holdout_size: Holdout subset size (as a fraction of original dataset)
        rewrite: If True, then previously organized files will be rewritten.
            Otherwise, the file will be ignored.
        extensions: File extensions considered when gathering files.
        random_state: Random state for sklearn.StratifiedShuffleSplit object
            which is used to split files into train/validation/test subsets.

    Returns:
        files: Dictionary with following keys:
            * train - array of train files paths
            * valid - array of validation files paths
            * holdout (if present) - array of holdout files paths

    Raises:
        ValueError: valid_size or holdout_size argument has invalid value.

    """
    if not (0.0 < valid_size < 1.0):
        raise ValueError(
            'valid_size parameter should take values from range (0.0, 1.0)')

    if holdout_size is not None:
        non_training = valid_size + holdout_size
        if not (0.0 < non_training < 1.0):
            raise ValueError(
                'the sum of valid_size and holdout_size '
                'should be in range (0.0, 1.0)')

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    filepaths = np.array([
        Path(path)
        for path in gather_files(
            src=dataset_dir,
            dst=output_dir,
            exts=extensions,
            rewrite=rewrite,
            delete_source=False)])

    split = StratifiedShuffleSplit(n_splits=1, random_state=random_state)
    uids = [filename.stem for filename in filepaths]
    targets = [classes[uid] for uid in uids]

    if holdout_size:
        split.test_size = holdout_size
        visible, hidden = next(split.split(filepaths, targets))

        n_samples = int(valid_size*len(filepaths))
        split.test_size = n_samples
        np_targets = np.array(targets)
        train, valid = next(split.split(
            filepaths[visible], np_targets[visible]))

        folders = [
            (filepaths[visible][train], 'train'),
            (filepaths[visible][valid], 'valid'),
            (filepaths[hidden], 'holdout')]
        files = _split_into_folders(folders, output_dir, rewrite)

    else:
        split.test_size = valid_size
        train, valid = next(split.split(filepaths, targets))
        folders = [
            (filepaths[train], 'train'),
            (filepaths[valid], 'valid')]
        files = _split_into_folders(folders, output_dir, rewrite)

    return files


def _split_into_folders(folders, output_dir, rewrite):
    """Moves files into separate train and validation folders."""

    files = defaultdict(list)

    for paths, folder_name in folders:
        folder = output_dir.joinpath(folder_name)
        folder.mkdir()

        for old_path in paths:
            new_path = folder.joinpath(old_path.name)
            if not new_path.exists() or rewrite:
                if new_path.exists():
                    new_path.unlink()
                shutil.move(src=old_path.as_posix(), dst=new_path.as_posix())
            files[folder_name].append(new_path.as_posix())

    return dict(files)


def split_into_class_folders(dataset_dir: str,
                             output_dir: str,
                             classes: dict,
                             rewrite: bool=False):
    """
    Splits dataset files from folder into set of subfolders where each
    subfolder contains a single class of images.

    Consider that original dataset has the following organization of files:

        /path/to/dataset
            - cat1512.jpeg
            - dog5131.jpeg
            - snake8067.jpeg
            - cat900.jpeg
            ...

    Then, after function is invoked, a new tree structure will be created:

        /path/to/output/dir
            /cat
                - cat1512.jpeg
                - cat900.jpeg
                ...
            /dog
                - dog5131.jpeg
                ...
            /snake
                - snake8067.jpeg
                ...

    Args:
        dataset_dir: Folder with original files.
        output_dir: New folder with files sorted into class subfolders.
        classes: Dictionary that maps file name to class represented by
            that file.
        rewrite: If True, then previously organized files will be rewritten.
            Otherwise, an exception is raised if any of files exists.

    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir()

    unique_classes = set(classes.values())
    class_folders = {}
    for class_name in unique_classes:
        folder = output_dir.joinpath(class_name)
        folder.mkdir()
        class_folders[class_name] = folder.as_posix()

    classified_files = defaultdict(list)
    for old_path in dataset_dir.glob('*.*'):
        filename = old_path.stem
        if filename not in classes:
            raise ValueError('there is no class for filename %s' % old_path)
        class_name = classes[filename]
        folder = class_folders[class_name]
        new_path = Path(folder).joinpath(old_path.name)
        if not new_path.exists() or rewrite:
            if new_path.exists():
                new_path.unlink()
            shutil.move(src=old_path.as_posix(), dst=new_path.as_posix())
        classified_files[class_name].append(new_path.as_posix())

    return class_folders, dict(classified_files)


class BatchGenerator:
    """Generates batches from a single array or a list of sequences."""

    def __init__(self, *arrays, batch_size=1, np_arrays=False):
        n = np.asarray(arrays[0]).shape[0]

        for arr in arrays[1:]:
            if len(arr) != n:
                raise ValueError('arrays are not of equal lengths')

        self.zipped = len(arrays) > 1
        self.arrays = arrays
        self.array_size = n
        self.batch_size = batch_size
        self.np_arrays = np_arrays

    def drain(self):
        return list(self.flow())

    def flow(self):
        if not self.np_arrays:
            yield from self._batches()
        else:
            for group in self._batches():
                yield np.asarray(group)

    def _batches(self):
        if self.batch_size == 1:
            yield from (list(x) for x in zip(*self.arrays))
        else:
            n_batches = math.ceil(self.array_size / self.batch_size)
            array = zip(*self.arrays) if self.zipped else iter(self.arrays[0])
            for _ in range(n_batches):
                batches = list(islice(array, self.batch_size))
                if self.zipped:
                    yield [list(x) for x in zip(*batches)]
                else:
                    yield batches

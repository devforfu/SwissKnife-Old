import math
import shutil
import fnmatch
from pathlib import Path
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
                        data_split=0.2,
                        holdout: bool=False,
                        rewrite: bool=False,
                        extensions: str='jpg|jpeg|png',
                        random_state: int=None):
    """
    Separates training files into training and validation folders.

    This function is intended to behave in a similar way like
    `test_train_split` function from `sklearn` package, but with files.

    Args:
        dataset_dir:
        output_dir:
        classes:
        data_split:
        holdout:
        rewrite:
        extensions:
        random_state:

    Returns:

    """
    if holdout and len(data_split) != 3:
        raise ValueError(
            'data_split tuple should have 3 items if holdout '
            'parameter is set to True')

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

    if holdout:
        train_size, valid_size, holdout_size = data_split
        split.test_size = train_size + valid_size

    else:
        split.test_size = data_split
        train, valid = next(split.split(filepaths, targets))
        files = defaultdict(list)
        for index, folder_name in ((train, 'train'), (valid, 'valid')):
            folder = output_dir.joinpath(folder_name)
            folder.mkdir()
            for filepath in filepaths[index]:
                old_path = filepath.as_posix()
                new_path = folder.joinpath(filepath.name).as_posix()
                shutil.move(src=old_path, dst=new_path)
                files[folder_name].append(new_path)
        return dict(files)

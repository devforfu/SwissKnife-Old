import math
import shutil
import fnmatch
from pathlib import Path
from itertools import chain


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

    ext_list = exts.split('|')
    src, dst = Path(src), Path(dst)

    if dst.exists():
        for _ in dst.iterdir():
            if not rewrite:
                raise OSError('folder exists: %s' % dst)
            break
        shutil.rmtree(dst)

    for old_path in src.glob('**/*.*'):
        for ext in ext_list:
            if fnmatch.fnmatch(old_path.as_posix(), '*.' + ext):
                relative_src = old_path.relative_to(src)
                new_path = dst.joinpath(relative_src)
                parent = new_path.parent
                if not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)
                shutil.move(old_path, new_path)
                break

    if delete_source:
        shutil.rmtree(src)

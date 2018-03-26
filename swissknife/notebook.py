"""
A group of snippets helpful while working with Jupyter Notebooks.
"""
import os
import sys


def format_list(seq):
    """Converts list-like objects into comma separated string representation.
    """
    if hasattr(seq, 'tolist'):
        seq = seq.tolist()
    elif hasattr(seq, '__len__'):
        seq = list(seq)
    return ', '.join(list(map(str, seq)))


def print_list(prompt, seq, newline=True):
    """Prints sequence into standard output."""

    print(prompt, format_list(seq))
    if newline:
        print()


def pandas_show_all():
    """Returns a context manager that temporary removes print limits on maximum
    number of displayed rows and columns of dataframe.
    """
    try:
        import pandas
    except ImportError:
        print('This utility cannot be used without \'pandas\' installed')
        return None
    else:
        ctx = pandas.option_context(
            'display.max_rows', None,
            'display.max_columns', None)
        return ctx


def hprint(msg, color='black', tag='p', **message_kwargs):
    """Wraps message with HTML tag."""

    try:
        from IPython.display import display, HTML
    except ImportError:
        print('This utility cannot be used without \'IPython\' installed')
        return None
    tag_text = msg.format(**message_kwargs)
    template = '<{0} style="color: {1}">{2}</{0}>'
    tag = template.format(tag, color, tag_text)
    display(HTML(tag))


class SysPath:
    """A singleton class with group of static that help modify interpreter
    search path. Useful when need to include additional modules and scripts
    which are not in working directory or among installed packages.
    """

    _instance = None

    def __new__(cls):
        if SysPath._instance is None:
            SysPath._instance = object.__new__(cls)
        return SysPath._instance

    def __init__(self):
        self.original_sys_path = None

    def extend(self, path, *paths):
        """Adds additional folders into Python interpreter search paths list.

        Note that this function should be called only ones per script execution
        or notebook kernel running or only after restore() method is called.
        """
        if self.original_sys_path is not None:
            return sys.path

        paths = [path] + list(paths)
        self.original_sys_path = sys.path.copy()
        expanded = [os.path.expanduser(p) for p in paths]
        unique = [p for p in expanded if p not in sys.path]
        new_sys_path = unique + sys.path
        sys.path = new_sys_path
        return new_sys_path

    def restore(self):
        """Restores original search path list if extend() method was called
        previously. Otherwise, the call does nothing.
        """
        if self.original_sys_path is not None:
            sys.path = self.original_sys_path
        self.original_sys_path = None
        return sys.path

    @staticmethod
    def print_paths(paths):
        """Prints a list of provided search paths in pretty format."""
        n_digits = len(str(len(paths)))
        template = '[{:%d}] {}' % n_digits
        print('Search paths:')
        for index, path in enumerate(sorted(paths), 1):
            string = template.format(index, path or 'working dir')
            print(string)


def extend_search_path(path, *paths):
    """Convenience wrapper for SysPath singleton extending interpreter's search
    path list
    """
    syspath = SysPath()
    updated = syspath.extend(path, *paths)
    syspath.print_paths(updated)

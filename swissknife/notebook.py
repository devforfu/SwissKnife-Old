"""
A group of snippets helpful while working with Jupyter Notebooks.
"""


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

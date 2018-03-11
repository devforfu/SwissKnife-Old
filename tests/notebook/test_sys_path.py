import sys

from swissknife.notebook import SysPath


def test_adding_new_search_path_onto_sys_path(monkeypatch):
    """Tests extending interpreter search paths with additional strings."""

    monkeypatch.setattr('sys.path', [])
    extension = ['/new/path', '/another/one']
    sys_path = SysPath()

    sys_path.extend(extension)

    assert sys.path == extension


def test_restoring_search_path_to_original_state(monkeypatch):
    """Tests restoring original search path if there was a call of paths
    extension method previously.
    """

    original = ['/path/to/module', '/path/to/another/module']
    monkeypatch.setattr('sys.path', original)
    sys_path = SysPath()
    sys_path.extend(['/new/path'])

    sys_path.restore()

    assert sys.path == original

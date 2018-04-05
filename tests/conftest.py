import os
import errno
import shutil
import fnmatch
from pathlib import Path
from itertools import chain

import pytest


PROJECT_ROOT = Path(__file__).parents[1]
TESTS_OUTPUT = PROJECT_ROOT / 'tests_output'
TESTS_FOLDER = PROJECT_ROOT / 'tests'
TESTS_FIXTURES = PROJECT_ROOT / 'test_fixtures'


@pytest.fixture(scope='session', autouse=True)
def setup_result_folders(request):
    """Gathers all files and reports created during tests run."""

    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    def gather_results():
        exts = 'txt', 'log', 'pdf', 'zip', 'png', 'json', 'csv'
        tests_output = TESTS_OUTPUT.as_posix()
        tests_folder = TESTS_FOLDER.as_posix()

        if os.path.exists(tests_output):
            shutil.rmtree(tests_output)

        for root, dirname, files in os.walk(tests_folder):
            for result in chain(
                    *[fnmatch.filter(files, '*.' + ext) for ext in exts]):
                src_path = os.path.abspath(os.path.join(root, result))
                relative_path = src_path.replace(tests_folder, "").strip('/')
                dst_path = os.path.join(tests_output, relative_path)
                new_folder = os.path.dirname(dst_path)
                if not os.path.exists(new_folder):
                    mkdir_p(new_folder)
                shutil.move(src_path, dst_path)

        for root, dirname, files in os.walk(tests_folder):
            if not dirname and not files:
                shutil.rmtree(root)

    request.addfinalizer(gather_results)


@pytest.fixture
def dog_image():
    return (TESTS_FIXTURES / 'images' / 'dog.png').as_posix()

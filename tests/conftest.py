from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1]
TESTS_OUTPUT = PROJECT_ROOT / 'tests_output'
TESTS_FOLDER = PROJECT_ROOT / 'tests'


@pytest.fixture(scope='session', autouse=True)
def setup_result_folders(request):
    """Gathers all files and reports created during tests run."""


@pytest.fixture
def tests_output():
    return TESTS_OUTPUT

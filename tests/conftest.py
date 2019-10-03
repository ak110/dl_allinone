import pathlib

import pytest


@pytest.fixture
def data_dir():
    yield pathlib.Path(__file__).parent / "data"

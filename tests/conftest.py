import matplotlib as mpl
import pytest


@pytest.fixture(scope="module", autouse=True)
def _setup():
    mpl.use("Agg")

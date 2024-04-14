
import matplotlib
import pytest


@pytest.fixture(scope="module", autouse=True)
def setup():
    matplotlib.use("Agg")
    yield

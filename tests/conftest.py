"""Reference data must be present before test modules load their fixtures."""
from .download_fixtures import ensure_fixtures


def pytest_sessionstart(session):
    ensure_fixtures()

"""
Shared fixtures and configuration for all tests.
"""

import sys
import numpy as np
import pytest
from pathlib import Path


# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "case_studies" / "src"))


@pytest.fixture
def random_seed():
    """Set and return random seed for reproducibility."""
    seed = 2026
    np.random.seed(seed)
    return seed


@pytest.fixture
def simple_quadratic():
    """Simple quadratic function for testing."""
    def f(X):
        X = np.atleast_2d(X)
        return X[:, 0]**2 + 2*X[:, 1] + 3*X[:, 2]
    return f


@pytest.fixture
def linear_function():
    """Simple linear function for testing."""
    def f(X):
        X = np.atleast_2d(X)
        return 2*X[:, 0] + 3*X[:, 1] - X[:, 2]
    return f


@pytest.fixture
def quadratic_sum_function():
    """Quadratic sum function for testing."""
    def f(X):
        X = np.atleast_2d(X)
        return np.sum(X**2, axis=1)
    return f


@pytest.fixture
def constant_function():
    """Constant function for testing."""
    def f(X):
        X = np.atleast_2d(X)
        return np.ones(X.shape[0]) * 5.0
    return f


@pytest.fixture
def test_point():
    """Standard test point and baseline."""
    return np.array([1.0, 1.0, 1.0]), np.zeros(3)


@pytest.fixture
def test_point_nonzero_baseline():
    """Test point with non-zero baseline."""
    return np.array([2.0, 3.0, 4.0]), np.ones(3)


# Pytest configuration via hook
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

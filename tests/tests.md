# Testing Guide for Axiomatic Interpretability

This document describes the testing approach for the Axiomatic Interpretability package.

## Test Files

### `tests/conftest.py`
Shared fixtures and configuration used across all tests:
- `random_seed`: Fixed seed for reproducibility
- `simple_quadratic`, `linear_function`, `quadratic_sum_function`: Test functions
- `constant_function`: Constant function for edge case testing
- `test_point`, `test_point_nonzero_baseline`: Standard test points

### `tests/test_auc.py`
Comprehensive tests for AUC computation:
- `TestAUCBasics`: Basic AUC computation and mathematical properties
- `TestMajorityBaselineCriterion`: Tests for majority-baseline-plus-10% criterion
- `TestAUCWithProbabilities`: AUC with probability scores
- `TestAUCWithRandomForest`: AUC with scikit-learn Random Forest models
- `TestAUCIntegration`: Integration tests with `save_results` function

### `tests/test_integrated_gradients.py`
Comprehensive tests for Integrated Gradients implementation:
- `TestIntegratedGradientsBasic`: Basic functionality and gradient computation
- `TestIntegratedGradientsEdgeCases`: Edge cases (zero/non-zero baseline, single feature, etc.)
- `TestIntegratedGradientsAdvanced`: Mathematical properties (completeness, gradient magnitudes)
- `TestIntegratedGradientsSklearn`: sklearn model integration (currently skipped)

## Running Tests

```bash
# Run all tests
python -m pytest tests -v

# Run specific test file
python -m pytest tests/test_auc.py -v
```

### Test Markers

We use pytest markers to categorize tests:

```bash
# Run only fast tests (skip slow)
python -m pytest -m "not slow" tests -v
```

Available markers:
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.integration`: Integration tests with external dependencies

### Test Configuration

The `pytest.ini` file contains:
- Test discovery patterns
- Test paths (`testpaths = tests`)
- Minimum pytest version requirement
- Strict marker enforcement
- Short traceback format

## Dependencies

- pytest ≥ 7.0
- numpy
- pandas
- scikit-learn

All dependencies are installed in the `ni_case_studies` conda environment.
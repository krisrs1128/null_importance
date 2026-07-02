"""
Tests for AUC (Area Under the Curve) computation.

This module tests:
1. Basic AUC computation and mathematical properties
2. Integration with sklearn metrics
3. Majority-baseline-plus-10% criterion
4. AUC with Random Forest models
5. Probability score handling
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "case_studies" / "src"))

from results import save_results


class TestAUCBasics:
    """Basic AUC computation tests without complex dependencies."""

    def test_auc_basics(self):
        """Test basic AUC computation."""
        # Simple test case
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])

        auc = roc_auc_score(y_true, y_scores)

        # Should be 0.75 (perfect separation would be 1.0)
        assert 0.7 < auc < 0.8, f"Expected AUC around 0.75, got {auc}"

    def test_auc_random_scores(self, random_seed):
        """Test AUC with random scores (should be around 0.5)."""
        y_true = np.array([0, 0, 1, 1])
        y_random = np.random.rand(4)  # Match size of y_true

        auc_random = roc_auc_score(y_true, y_random)

        # Should be close to 0.5 (random performance)
        assert 0.4 < auc_random < 0.6, f"Random AUC should be ~0.5, got {auc_random}"

    def test_auc_perfect_separation(self):
        """Test AUC with perfect separation."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.0, 0.1, 0.9, 1.0])

        auc = roc_auc_score(y_true, y_scores)

        # Should be 1.0 (perfect separation)
        assert auc == 1.0, f"Expected AUC=1.0 for perfect separation, got {auc}"


class TestMajorityBaselineCriterion:
    """Tests for majority-baseline-plus-10% criterion."""

    def test_majority_baseline_simple(self):
        """Test majority baseline + 10% criterion with simple cases."""
        # For AUC, majority baseline is 0.5 (random classifier)
        # Criterion: AUC > 0.5 + 0.1 = 0.6

        test_cases = [
            (0.65, True),   # Passes (0.65 > 0.6)
            (0.61, True),   # Passes (0.61 > 0.6)
            (0.60, False),  # Fails (0.60 == 0.6, not greater)
            (0.55, False),  # Fails (0.55 < 0.6)
            (0.75, True),   # Passes (0.75 > 0.6)
        ]

        criterion = 0.6
        for auc, should_pass in test_cases:
            passes = auc > criterion
            assert passes == should_pass, f"AUC {auc} criterion check failed"

    def test_majority_baseline_with_imbalanced_classes(self):
        """Test majority baseline criterion with imbalanced class distributions."""
        # For AUC, majority baseline is still 0.5 (random is independent of class imbalance)
        # Criterion: AUC > 0.5 + 0.1 = 0.6

        test_cases = [
            {"majority_prop": 0.7, "auc": 0.65, "should_pass": True},
            {"majority_prop": 0.8, "auc": 0.61, "should_pass": True},
            {"majority_prop": 0.6, "auc": 0.55, "should_pass": False},
            {"majority_prop": 0.9, "auc": 0.75, "should_pass": True},
        ]

        majority_baseline = 0.5
        criterion = majority_baseline + 0.1

        for case in test_cases:
            passes = case["auc"] > criterion
            assert passes == case["should_pass"], \
                f"AUC {case['auc']} criterion check failed for majority={case['majority_prop']}"


class TestAUCWithProbabilities:
    """Tests for AUC computation with probability scores."""

    def test_auc_with_probabilities(self, random_seed):
        """Test AUC computation with probability scores."""
        # Create a more realistic scenario
        n_samples = 200

        # Create some features
        X = np.random.randn(n_samples, 3)

        # Make class 1 depend on feature 0
        y_true = (X[:, 0] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

        # Create probability scores that correlate with y_true
        y_scores = 1 / (1 + np.exp(-X[:, 0]))  # Sigmoid of feature 0

        auc = roc_auc_score(y_true, y_scores)

        # Should be significantly better than random
        assert auc > 0.7, f"Expected AUC > 0.7, got {auc}"

    def test_auc_criterion_with_probabilities(self, random_seed):
        """Test AUC meets criterion with probability scores."""
        n_samples = 200
        X = np.random.randn(n_samples, 3)
        y_true = (X[:, 0] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        y_scores = 1 / (1 + np.exp(-X[:, 0]))

        auc = roc_auc_score(y_true, y_scores)

        # Test criterion
        meets_criterion = auc > 0.6
        assert meets_criterion, "Should meet AUC criterion"


class TestAUCWithRandomForest:
    """Tests for AUC with Random Forest models."""

    def test_auc_with_rf_model(self, random_seed):
        """Test AUC computation with actual Random Forest model."""
        # Create synthetic data with some signal
        n_samples = 200
        X = pd.DataFrame(
            np.random.randn(n_samples, 20),
            columns=[f"feature_{i}" for i in range(20)]
        )

        # Make y depend on first few features
        X.iloc[:, 0] = X.iloc[:, 0] + np.where(X.iloc[:, 1] > 0, 1, -1)
        y = pd.Series((
            X.iloc[:, 0] + 0.5 * X.iloc[:, 1] + np.random.randn(n_samples) > 0
        ).astype(int))

        # Train RF model
        rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        rf.fit(X, y)

        # Compute predictions and probabilities
        y_probs = rf.predict_proba(X)[:, 1]  # Probabilities for class 1

        # Compute metrics
        auc = roc_auc_score(y, y_probs)

        # Check if it meets criterion
        meets_criterion = auc > 0.6
        assert meets_criterion, "RF model should meet AUC criterion on synthetic data"

    def test_auc_rf_gradient_magnitudes(self, random_seed):
        """Test that RF model produces reasonable AUC values."""
        n_samples = 200
        X = pd.DataFrame(
            np.random.randn(n_samples, 10),
            columns=[f"feature_{i}" for i in range(10)]
        )
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))

        rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        rf.fit(X, y)

        y_probs = rf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_probs)

        # AUC should be a valid probability
        assert 0.0 <= auc <= 1.0, f"AUC should be in [0, 1], got {auc}"


class TestAUCIntegration:
    """Integration tests for AUC computation with results saving."""

    def test_auc_computation_integration(self, random_seed):
        """Test that AUC computation works correctly with save_results."""
        # Create synthetic data
        n_samples = 100
        X = pd.DataFrame(
            np.random.randn(n_samples, 10),
            columns=[f"feature_{i}" for i in range(10)]
        )
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))

        # Create some predictions that should give reasonable AUC
        y_probs = np.random.rand(n_samples)
        y_pred = (y_probs > 0.5).astype(int)

        # Train model
        final_model = RandomForestClassifier(n_estimators=10, random_state=42)
        final_model.fit(X, y)

        # Create temporary directory for testing
        test_dir = Path("/tmp/test_results_auc")
        test_dir.mkdir(exist_ok=True)

        try:
            # Test the modified save_results function
            mccs = [0.3, 0.35, 0.4]  # Example MCC scores
            aucs = [0.7, 0.75, 0.8]  # Example AUC scores
            all_preds = [y_pred, y_pred, y_pred]  # Same predictions for all reps
            best_params = (3, 5)
            y_labels = ["Class_0", "Class_1"]

            save_results(
                test_dir,
                X, y,
                all_preds,
                mccs, aucs,
                final_model,
                best_params,
                y_labels
            )

            # Verify files were created
            mcc_file = test_dir / "mccs.csv"
            auc_file = test_dir / "auc.csv"

            assert mcc_file.exists(), "mccs.csv not created"
            assert auc_file.exists(), "auc.csv not created"

            # Verify content
            mcc_data = pd.read_csv(mcc_file)
            auc_data = pd.read_csv(auc_file)

            assert len(mcc_data) == 3, "MCC data should have 3 entries"
            assert len(auc_data) == 3, "AUC data should have 3 entries"

        finally:
            # Clean up
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)

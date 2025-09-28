"""Evaluation metrics for pass prediction models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class PassPredictionEvaluator:
    """Evaluate pass prediction models."""

    def __init__(self):
        """Initialize evaluator."""
        self.results_ = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate evaluation metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            sample_weight: Sample weights

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics["mae"] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
        metrics["mape"] = self._mean_absolute_percentage_error(y_true, y_pred, sample_weight)
        metrics["r2"] = r2_score(y_true, y_pred, sample_weight=sample_weight)

        # Count-specific metrics
        metrics["poisson_deviance"] = self._poisson_deviance(y_true, y_pred, sample_weight)
        metrics["chi_squared"] = self._chi_squared_statistic(y_true, y_pred)

        # Percentile metrics
        metrics["median_ae"] = np.median(np.abs(y_true - y_pred))
        metrics["percentile_90_ae"] = np.percentile(np.abs(y_true - y_pred), 90)

        # Under/over prediction
        residuals = y_true - y_pred
        metrics["mean_residual"] = np.mean(residuals)
        metrics["pct_over_predicted"] = np.mean(residuals < 0) * 100
        metrics["pct_under_predicted"] = np.mean(residuals > 0) * 100

        return metrics

    def evaluate_by_group(self, y_true: np.ndarray, y_pred: np.ndarray,
                         groups: pd.Series, group_name: str = "group") -> pd.DataFrame:
        """Evaluate metrics by group.

        Args:
            y_true: True values
            y_pred: Predicted values
            groups: Group labels
            group_name: Name for the group column

        Returns:
            DataFrame with metrics by group
        """
        results = []

        for group in groups.unique():
            mask = groups == group
            group_metrics = self.evaluate(y_true[mask], y_pred[mask])
            group_metrics[group_name] = group
            group_metrics["n_samples"] = mask.sum()
            results.append(group_metrics)

        return pd.DataFrame(results)

    def evaluate_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_lower: np.ndarray, y_upper: np.ndarray,
                                     confidence_level: float = 0.9) -> Dict[str, float]:
        """Evaluate prediction intervals.

        Args:
            y_true: True values
            y_pred: Point predictions
            y_lower: Lower bound predictions
            y_upper: Upper bound predictions
            confidence_level: Expected coverage level

        Returns:
            Dictionary of interval metrics
        """
        metrics = {}

        # Coverage
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        metrics["coverage"] = coverage
        metrics["coverage_gap"] = coverage - confidence_level

        # Interval width
        interval_width = y_upper - y_lower
        metrics["mean_interval_width"] = np.mean(interval_width)
        metrics["median_interval_width"] = np.median(interval_width)

        # Interval score (lower is better)
        alpha = 1 - confidence_level
        lower_penalty = (y_lower - y_true) * (y_true < y_lower) * (2 / alpha)
        upper_penalty = (y_true - y_upper) * (y_true > y_upper) * (2 / alpha)
        metrics["interval_score"] = np.mean(interval_width + lower_penalty + upper_penalty)

        # Calibration by percentile
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        for p in percentiles:
            metrics[f"calibration_p{p}"] = np.mean(y_true <= np.percentile(y_pred, p))

        return metrics

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        title: str = "Pass Predictions vs Actual") -> plt.Figure:
        """Plot predictions vs actual values.

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Scatter plot
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([0, max(y_true)], [0, max(y_true)], 'r--', label='Perfect prediction')
        ax.set_xlabel("Actual Passes")
        ax.set_ylabel("Predicted Passes")
        ax.set_title("Predictions vs Actual")
        ax.legend()

        # Residual plot
        ax = axes[0, 1]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Predicted Passes")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")

        # Distribution of residuals
        ax = axes[1, 0]
        ax.hist(residuals, bins=30, edgecolor='black')
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Residual Distribution (mean={np.mean(residuals):.2f})")

        # Error by prediction magnitude
        ax = axes[1, 1]
        pred_bins = pd.qcut(y_pred, q=10, duplicates='drop')
        error_by_bin = pd.DataFrame({
            'pred_bin': pred_bins,
            'abs_error': np.abs(residuals)
        }).groupby('pred_bin')['abs_error'].mean()

        ax.bar(range(len(error_by_bin)), error_by_bin.values)
        ax.set_xlabel("Prediction Decile")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title("Error by Prediction Magnitude")

        plt.suptitle(title)
        plt.tight_layout()

        return fig

    def cross_validate_temporal(self, model, X: pd.DataFrame, y: pd.Series,
                               dates: pd.Series, n_splits: int = 5,
                               exposure: Optional[pd.Series] = None) -> pd.DataFrame:
        """Temporal cross-validation.

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target values
            dates: Date series for temporal ordering
            n_splits: Number of CV splits
            exposure: Exposure variable

        Returns:
            DataFrame with CV results
        """
        # Sort by date
        sorted_idx = dates.argsort()
        X_sorted = X.iloc[sorted_idx]
        y_sorted = y.iloc[sorted_idx]
        exposure_sorted = exposure.iloc[sorted_idx] if exposure is not None else None

        # Calculate split points
        n_samples = len(X)
        test_size = n_samples // (n_splits + 1)

        cv_results = []

        for i in range(n_splits):
            # Define train and test indices
            test_start = (i + 1) * test_size
            test_end = min(test_start + test_size, n_samples)

            train_idx = range(0, test_start)
            test_idx = range(test_start, test_end)

            # Split data
            X_train = X_sorted.iloc[train_idx]
            X_test = X_sorted.iloc[test_idx]
            y_train = y_sorted.iloc[train_idx]
            y_test = y_sorted.iloc[test_idx]

            if exposure is not None:
                exposure_train = exposure_sorted.iloc[train_idx]
                exposure_test = exposure_sorted.iloc[test_idx]

                # Fit and predict
                if hasattr(model, 'fit') and "exposure" in model.fit.__code__.co_varnames:
                    model.fit(X_train, y_train, exposure=exposure_train)
                    y_pred = model.predict(X_test, exposure=exposure_test)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Evaluate
            fold_metrics = self.evaluate(y_test, y_pred)
            fold_metrics["fold"] = i
            fold_metrics["n_train"] = len(train_idx)
            fold_metrics["n_test"] = len(test_idx)

            cv_results.append(fold_metrics)

        return pd.DataFrame(cv_results)

    def _mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       sample_weight: Optional[np.ndarray] = None) -> float:
        """Calculate MAPE."""
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return np.nan

        mape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])

        if sample_weight is not None:
            return np.average(mape, weights=sample_weight[mask]) * 100
        else:
            return np.mean(mape) * 100

    def _poisson_deviance(self, y_true: np.ndarray, y_pred: np.ndarray,
                         sample_weight: Optional[np.ndarray] = None) -> float:
        """Calculate Poisson deviance."""
        # Avoid log(0)
        y_pred_safe = np.maximum(y_pred, 1e-10)

        deviance = 2 * (y_true * np.log(y_true / y_pred_safe + 1e-10) - (y_true - y_pred))

        if sample_weight is not None:
            return np.average(deviance, weights=sample_weight)
        else:
            return np.mean(deviance)

    def _chi_squared_statistic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate chi-squared statistic."""
        # Avoid division by zero
        y_pred_safe = np.maximum(y_pred, 1e-10)

        chi_squared = np.sum((y_true - y_pred) ** 2 / y_pred_safe)

        return chi_squared
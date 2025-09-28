"""Baseline models for pass prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import statsmodels.api as sm
from statsmodels.discrete.count_model import Poisson
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


class HistoricalAverageBaseline(BaseEstimator, RegressorMixin):
    """Simple baseline using historical averages."""

    def __init__(self, group_columns: Optional[list] = None):
        """Initialize the baseline model.

        Args:
            group_columns: Columns to group by for averaging
        """
        self.group_columns = group_columns or ["position_group"]
        self.averages_ = {}
        self.global_average_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalAverageBaseline":
        """Fit the baseline model.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Self
        """
        # Combine X and y for grouping
        data = X.copy()
        data["target"] = y

        # Calculate global average
        self.global_average_ = y.mean()

        # Calculate group averages
        for col in self.group_columns:
            if col in X.columns:
                self.averages_[col] = data.groupby(col)["target"].mean().to_dict()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        predictions = np.full(len(X), self.global_average_)

        # Use group averages where available
        for idx, (i, row) in enumerate(X.iterrows()):
            for col in self.group_columns:
                if col in X.columns and col in self.averages_:
                    group_value = row[col]
                    if group_value in self.averages_[col]:
                        predictions[idx] = self.averages_[col][group_value]
                        break

        return predictions


class PoissonRegression(BaseEstimator, RegressorMixin):
    """Poisson regression model for count data."""

    def __init__(self, use_exposure: bool = True, alpha: float = 0.0):
        """Initialize Poisson regression.

        Args:
            use_exposure: Whether to use exposure (minutes played)
            alpha: Regularization parameter
        """
        self.use_exposure = use_exposure
        self.alpha = alpha
        self.model_ = None
        self.results_ = None
        self.feature_columns_ = None  # Track columns used in fitting

    def fit(self, X: pd.DataFrame, y: pd.Series, exposure: Optional[pd.Series] = None) -> "PoissonRegression":
        """Fit the Poisson regression model.

        Args:
            X: Feature matrix
            y: Target values (counts)
            exposure: Exposure variable (e.g., minutes played)

        Returns:
            Self
        """
        # Handle missing values in features
        X_clean = X.fillna(X.mean())

        # Store original column means for prediction
        self.feature_means_ = X.mean()

        # Add constant (force add even if one might exist)
        X_with_const = sm.add_constant(X_clean, prepend=True, has_constant='add')

        # Check for and remove any columns with zero variance to avoid singularity
        col_variance = X_with_const.var()
        non_zero_var_cols = col_variance > 1e-10
        if not all(non_zero_var_cols):
            dropped_cols = X_with_const.columns[~non_zero_var_cols].tolist()
            print(f"Warning: Dropping zero-variance columns: {dropped_cols}")
            X_with_const = X_with_const.loc[:, non_zero_var_cols]

        # Store the columns that were actually used for fitting
        self.feature_columns_ = X_with_const.columns.tolist()

        # Prepare exposure
        if self.use_exposure and exposure is not None:
            # Use log of exposure as offset
            offset = np.log(np.maximum(exposure, 1))  # Avoid log(0)
        else:
            offset = None

        try:
            # Fit Poisson model
            self.model_ = Poisson(
                endog=y,
                exog=X_with_const,
                offset=offset
            )

            # Add regularization if specified
            if self.alpha > 0:
                self.results_ = self.model_.fit_regularized(alpha=self.alpha)
            else:
                self.results_ = self.model_.fit(disp=False)

        except np.linalg.LinAlgError as e:
            # If standard fit fails, use regularization to handle singularity
            print(f"Warning: Standard Poisson regression failed ({e}), using regularization")
            self.model_ = Poisson(
                endog=y,
                exog=X_with_const,
                offset=offset
            )
            self.results_ = self.model_.fit_regularized(alpha=0.01, disp=False)

        return self

    def predict(self, X: pd.DataFrame, exposure: Optional[pd.Series] = None) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix
            exposure: Exposure variable

        Returns:
            Predictions
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted before prediction")

        # Handle missing values first - use stored means from training
        if hasattr(self, 'feature_means_'):
            X_clean = X.fillna(self.feature_means_)
        else:
            X_clean = X.fillna(X.mean())

        # Add constant - must match training exactly
        X_with_const = sm.add_constant(X_clean, prepend=True, has_constant='add')

        # Ensure we only use the columns that were used during fitting
        if self.feature_columns_ is not None:
            # Only keep columns that were used during training
            cols_to_use = [col for col in self.feature_columns_ if col in X_with_const.columns]
            if len(cols_to_use) < len(self.feature_columns_):
                # Some columns from training are missing, add them as zeros
                for col in self.feature_columns_:
                    if col not in X_with_const.columns:
                        X_with_const[col] = 0
            # Reorder columns to match training
            X_with_const = X_with_const[self.feature_columns_]

        # Prepare exposure
        if self.use_exposure and exposure is not None:
            # Ensure exposure is aligned with X
            if isinstance(exposure, pd.Series):
                # Reset index to ensure alignment
                exposure_values = exposure.values
            else:
                exposure_values = exposure

            offset = np.log(np.maximum(exposure_values, 1))
            # Manual prediction with offset: exp(X*beta + offset)
            linear_pred = X_with_const @ self.results_.params
            predictions = np.exp(linear_pred + offset)
        else:
            # Without offset
            predictions = self.results_.predict(X_with_const)

        # Handle any NaN or inf values in predictions
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1000.0, neginf=0.0)

        return predictions

    def predict_interval(self, X: pd.DataFrame, exposure: Optional[pd.Series] = None,
                        alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals.

        Args:
            X: Feature matrix
            exposure: Exposure variable
            alpha: Significance level for intervals

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        predictions = self.predict(X, exposure)

        # For Poisson, use the mean as the parameter for the distribution
        # Approximate confidence intervals using normal approximation
        z_score = np.abs(np.percentile(np.random.standard_normal(10000), (alpha/2) * 100))
        std_dev = np.sqrt(predictions)  # For Poisson, variance = mean

        lower = predictions - z_score * std_dev
        upper = predictions + z_score * std_dev

        return predictions, np.maximum(lower, 0), upper

    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients.

        Returns:
            DataFrame with coefficients and statistics
        """
        if self.results_ is None:
            return pd.DataFrame()

        summary = self.results_.summary2().tables[1]
        return summary


class NegativeBinomialRegression(BaseEstimator, RegressorMixin):
    """Negative Binomial regression for overdispersed count data."""

    def __init__(self, use_exposure: bool = True):
        """Initialize Negative Binomial regression.

        Args:
            use_exposure: Whether to use exposure (minutes played)
        """
        self.use_exposure = use_exposure
        self.model_ = None
        self.results_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, exposure: Optional[pd.Series] = None) -> "NegativeBinomialRegression":
        """Fit the Negative Binomial regression model.

        Args:
            X: Feature matrix
            y: Target values (counts)
            exposure: Exposure variable (e.g., minutes played)

        Returns:
            Self
        """
        # Add constant
        X_with_const = sm.add_constant(X)

        # Prepare exposure
        if self.use_exposure and exposure is not None:
            offset = np.log(np.maximum(exposure, 1))
        else:
            offset = None

        # Fit Negative Binomial model
        try:
            from statsmodels.discrete.count_model import NegativeBinomial
            self.model_ = NegativeBinomial(
                endog=y,
                exog=X_with_const,
                offset=offset
            )
            self.results_ = self.model_.fit(disp=False)
        except:
            # Fallback to GLM with negative binomial family
            import statsmodels.genmod.families as families
            self.model_ = sm.GLM(
                endog=y,
                exog=X_with_const,
                family=families.NegativeBinomial(),
                offset=offset
            )
            self.results_ = self.model_.fit()

        return self

    def predict(self, X: pd.DataFrame, exposure: Optional[pd.Series] = None) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix
            exposure: Exposure variable

        Returns:
            Predictions
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted before prediction")

        # Add constant
        X_with_const = sm.add_constant(X)

        # Prepare exposure
        if self.use_exposure and exposure is not None:
            offset = np.log(np.maximum(exposure, 1))
        else:
            offset = None

        # Make predictions
        if offset is not None:
            # Manual calculation with offset
            linear_pred = X_with_const @ self.results_.params + offset
            predictions = np.exp(linear_pred)
        else:
            predictions = self.results_.predict(X_with_const)

        return predictions


class EnsembleBaseline(BaseEstimator, RegressorMixin):
    """Ensemble of baseline models."""

    def __init__(self, models: Optional[Dict[str, BaseEstimator]] = None):
        """Initialize ensemble.

        Args:
            models: Dictionary of models to ensemble
        """
        if models is None:
            models = {
                "historical": HistoricalAverageBaseline(),
                "poisson": PoissonRegression()
            }
        self.models = models
        self.weights_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, exposure: Optional[pd.Series] = None) -> "EnsembleBaseline":
        """Fit all models in the ensemble.

        Args:
            X: Feature matrix
            y: Target values
            exposure: Exposure variable

        Returns:
            Self
        """
        for name, model in self.models.items():
            if hasattr(model, 'fit') and callable(model.fit):
                if "exposure" in model.fit.__code__.co_varnames:
                    model.fit(X, y, exposure=exposure)
                else:
                    model.fit(X, y)

        # Simple equal weighting for now
        self.weights_ = {name: 1.0 / len(self.models) for name in self.models}

        return self

    def predict(self, X: pd.DataFrame, exposure: Optional[pd.Series] = None) -> np.ndarray:
        """Make ensemble predictions.

        Args:
            X: Feature matrix
            exposure: Exposure variable

        Returns:
            Weighted average predictions
        """
        predictions = np.zeros(len(X))

        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                if "exposure" in model.predict.__code__.co_varnames:
                    pred = model.predict(X, exposure=exposure)
                else:
                    pred = model.predict(X)

                predictions += self.weights_[name] * pred

        return predictions
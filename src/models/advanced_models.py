"""Advanced models for pass prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
from ..models.baseline import PoissonRegression, HistoricalAverageBaseline

warnings.filterwarnings('ignore')


class PositionSpecificModel(BaseEstimator, RegressorMixin):
    """Train separate models for each position group."""

    def __init__(self, base_model_class=PoissonRegression,
                 position_column: str = "position_encoded",
                 model_params: Optional[Dict] = None):
        """Initialize position-specific model.

        Args:
            base_model_class: Base model class to use for each position
            position_column: Column containing position information
            model_params: Parameters to pass to base model
        """
        self.base_model_class = base_model_class
        self.position_column = position_column
        self.model_params = model_params or {}
        self.models_ = {}
        self.global_model_ = None
        self.position_groups_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, exposure: Optional[pd.Series] = None) -> "PositionSpecificModel":
        """Fit position-specific models.

        Args:
            X: Feature matrix with position column
            y: Target values
            exposure: Exposure variable (minutes played)

        Returns:
            Self
        """
        # Train global model as fallback
        self.global_model_ = self.base_model_class(**self.model_params)
        try:
            if hasattr(self.global_model_, 'fit') and 'exposure' in self.global_model_.fit.__code__.co_varnames:
                self.global_model_.fit(X, y, exposure=exposure)
            else:
                self.global_model_.fit(X, y)
        except (np.linalg.LinAlgError, ValueError) as e:
            # If global model fails, use a simple historical average
            print(f"Warning: Global model failed to fit ({e}), using historical average")
            from ..models.baseline import HistoricalAverageBaseline
            self.global_model_ = HistoricalAverageBaseline()
            self.global_model_.fit(X, y)

        # Get unique positions
        if self.position_column in X.columns:
            self.position_groups_ = X[self.position_column].unique()

            # Train model for each position
            for position in self.position_groups_:
                position_mask = X[self.position_column] == position
                X_pos = X[position_mask]
                y_pos = y[position_mask]

                if len(X_pos) >= 50:  # Only train if sufficient data
                    try:
                        model = self.base_model_class(**self.model_params)

                        if exposure is not None:
                            exposure_pos = exposure[position_mask]
                            if hasattr(model, 'fit') and 'exposure' in model.fit.__code__.co_varnames:
                                model.fit(X_pos, y_pos, exposure=exposure_pos)
                            else:
                                model.fit(X_pos, y_pos)
                        else:
                            model.fit(X_pos, y_pos)

                        self.models_[position] = model

                    except (np.linalg.LinAlgError, ValueError) as e:
                        # Skip this position model if it fails
                        print(f"Warning: Position {position} model failed to fit ({e}), will use global model")
                        continue

        return self

    def predict(self, X: pd.DataFrame, exposure: Optional[pd.Series] = None) -> np.ndarray:
        """Make position-specific predictions.

        Args:
            X: Feature matrix with position column
            exposure: Exposure variable

        Returns:
            Predictions
        """
        predictions = np.zeros(len(X))

        if self.position_column in X.columns:
            # Use enumerate to get positional indices
            for i, (idx, row) in enumerate(X.iterrows()):
                position = row[self.position_column]

                # Use position-specific model if available
                if position in self.models_:
                    model = self.models_[position]
                    X_row = X.loc[[idx]]  # Use .loc with actual index

                    if exposure is not None:
                        exposure_row = exposure.loc[[idx]] if isinstance(exposure, pd.Series) else exposure[[i]]
                        if hasattr(model, 'predict') and 'exposure' in model.predict.__code__.co_varnames:
                            pred = model.predict(X_row, exposure=exposure_row)
                        else:
                            pred = model.predict(X_row)
                    else:
                        pred = model.predict(X_row)

                    predictions[i] = pred[0]  # Use positional index for storing
                else:
                    # Use global model as fallback
                    X_row = X.loc[[idx]]  # Use .loc with actual index
                    if exposure is not None:
                        exposure_row = exposure.loc[[idx]] if isinstance(exposure, pd.Series) else exposure[[i]]
                        if hasattr(self.global_model_, 'predict') and 'exposure' in self.global_model_.predict.__code__.co_varnames:
                            pred = self.global_model_.predict(X_row, exposure=exposure_row)
                        else:
                            pred = self.global_model_.predict(X_row)
                    else:
                        pred = self.global_model_.predict(X_row)

                    predictions[i] = pred[0]  # Use positional index for storing
        else:
            # No position column, use global model
            if exposure is not None:
                if hasattr(self.global_model_, 'predict') and 'exposure' in self.global_model_.predict.__code__.co_varnames:
                    predictions = self.global_model_.predict(X, exposure=exposure)
                else:
                    predictions = self.global_model_.predict(X)
            else:
                predictions = self.global_model_.predict(X)

        return predictions


class XGBoostRegressor(BaseEstimator, RegressorMixin):
    """XGBoost model for pass prediction."""

    def __init__(self, use_exposure: bool = True, base_params: Optional[Dict] = None,
                 max_depth: int = 5, learning_rate: float = 0.1, n_estimators: int = 100,
                 subsample: float = 0.8, colsample_bytree: float = 0.8,
                 min_child_weight: float = 1, gamma: float = 0,
                 reg_alpha: float = 0, reg_lambda: float = 1):
        """Initialize XGBoost regressor.

        Args:
            use_exposure: Whether to use exposure (minutes played)
            base_params: Base parameters dict (overrides individual params)
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            n_estimators: Number of trees
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            min_child_weight: Minimum child weight
            gamma: Gamma regularization
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
        """
        self.use_exposure = use_exposure
        self.base_params = base_params
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.model_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, exposure: Optional[pd.Series] = None) -> "XGBoostRegressor":
        """Fit XGBoost model.

        Args:
            X: Feature matrix
            y: Target values (counts)
            exposure: Exposure variable (e.g., minutes played)

        Returns:
            Self
        """
        # Handle missing values
        X_clean = X.fillna(X.mean())
        self.feature_names_ = X_clean.columns.tolist()

        # Build parameters from instance attributes
        if self.base_params:
            # Use base_params if provided (for tuned parameters)
            params = self.base_params.copy()
        else:
            # Build from individual parameters
            params = {
                'objective': 'count:poisson',  # Poisson regression for counts
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'min_child_weight': self.min_child_weight,
                'gamma': self.gamma,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'random_state': 42,
                'verbosity': 0
            }

        # Create DMatrix with exposure if provided
        if self.use_exposure and exposure is not None:
            # XGBoost uses base_margin for offset/exposure
            base_margin = np.log(np.maximum(exposure, 1))
            dtrain = xgb.DMatrix(X_clean, label=y, base_margin=base_margin)
        else:
            dtrain = xgb.DMatrix(X_clean, label=y)

        # Train model
        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100)
        )

        return self

    def predict(self, X: pd.DataFrame, exposure: Optional[pd.Series] = None) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix
            exposure: Exposure variable

        Returns:
            Predictions
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction")

        # Handle missing values
        X_clean = X.fillna(X.mean())

        # Create DMatrix
        if self.use_exposure and exposure is not None:
            base_margin = np.log(np.maximum(exposure, 1))
            dtest = xgb.DMatrix(X_clean, base_margin=base_margin)
        else:
            dtest = xgb.DMatrix(X_clean)

        # Make predictions
        predictions = self.model_.predict(dtest)

        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.

        Returns:
            DataFrame with feature importance
        """
        if self.model_ is None:
            return pd.DataFrame()

        importance_dict = self.model_.get_score(importance_type='gain')
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)

        return importance_df


class TwoStageModel(BaseEstimator, RegressorMixin):
    """Two-stage model: Team volume × Player share."""

    def __init__(self, team_model_class=XGBoostRegressor,
                 player_model_class=PoissonRegression,
                 team_features: Optional[List[str]] = None,
                 player_features: Optional[List[str]] = None):
        """Initialize two-stage model.

        Args:
            team_model_class: Model class for team volume prediction
            player_model_class: Model class for player share prediction
            team_features: Features for team model
            player_features: Features for player model
        """
        self.team_model_class = team_model_class
        self.player_model_class = player_model_class
        self.team_features = team_features
        self.player_features = player_features
        self.team_model_ = None
        self.player_model_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, exposure: Optional[pd.Series] = None) -> "TwoStageModel":
        """Fit two-stage model.

        Args:
            X: Feature matrix
            y: Target values (player passes)
            exposure: Exposure variable (minutes played)

        Returns:
            Self
        """
        # Stage 1: Aggregate to team level
        if 'match_id' in X.columns and 'team' in X.columns:
            team_data = X.groupby(['match_id', 'team']).agg({
                col: 'mean' for col in X.columns
                if col not in ['match_id', 'team', 'player']
            }).reset_index()

            # Calculate team total passes
            team_totals = pd.DataFrame(
                X.groupby(['match_id', 'team'])['passes_attempted'].sum()
            ).reset_index()
            team_totals.columns = ['match_id', 'team', 'team_total_passes']

            # Merge team totals
            team_data = team_data.merge(team_totals, on=['match_id', 'team'])

            # Select team features
            if self.team_features:
                X_team = team_data[self.team_features]
            else:
                X_team = team_data.drop(columns=['match_id', 'team', 'team_total_passes'], errors='ignore')

            y_team = team_data['team_total_passes']

            # Train team model
            self.team_model_ = self.team_model_class()
            if hasattr(self.team_model_, 'fit'):
                self.team_model_.fit(X_team, y_team)

        # Stage 2: Player share model
        # Calculate player share of team passes
        if 'passes_attempted' in X.columns:
            player_share = y / X.groupby(['match_id', 'team'])['passes_attempted'].transform('sum')
            player_share = player_share.fillna(0).clip(0, 1)

            # Select player features
            if self.player_features:
                X_player = X[self.player_features]
            else:
                X_player = X

            # Train player share model
            self.player_model_ = self.player_model_class()
            if hasattr(self.player_model_, 'fit'):
                if exposure is not None and 'exposure' in self.player_model_.fit.__code__.co_varnames:
                    self.player_model_.fit(X_player, player_share, exposure=exposure)
                else:
                    self.player_model_.fit(X_player, player_share)

        return self

    def predict(self, X: pd.DataFrame, exposure: Optional[pd.Series] = None) -> np.ndarray:
        """Make two-stage predictions.

        Args:
            X: Feature matrix
            exposure: Exposure variable

        Returns:
            Predictions (team_volume × player_share)
        """
        predictions = np.zeros(len(X))

        # Stage 1: Predict team volume
        if self.team_model_ is not None and 'match_id' in X.columns and 'team' in X.columns:
            team_data = X.groupby(['match_id', 'team']).agg({
                col: 'mean' for col in X.columns
                if col not in ['match_id', 'team', 'player']
            }).reset_index()

            if self.team_features:
                X_team = team_data[self.team_features]
            else:
                X_team = team_data.drop(columns=['match_id', 'team'], errors='ignore')

            team_predictions = self.team_model_.predict(X_team)

            # Map team predictions back to players
            team_pred_dict = {}
            for i, (match_id, team) in enumerate(team_data[['match_id', 'team']].values):
                team_pred_dict[(match_id, team)] = team_predictions[i]

            team_volumes = X.apply(
                lambda row: team_pred_dict.get((row['match_id'], row['team']), 50),  # Default to 50 passes
                axis=1
            ).values

        else:
            # Default team volume
            team_volumes = np.full(len(X), 50)

        # Stage 2: Predict player share
        if self.player_model_ is not None:
            if self.player_features:
                X_player = X[self.player_features]
            else:
                X_player = X

            if exposure is not None and hasattr(self.player_model_, 'predict') and \
               'exposure' in self.player_model_.predict.__code__.co_varnames:
                player_shares = self.player_model_.predict(X_player, exposure=exposure)
            else:
                player_shares = self.player_model_.predict(X_player)

            # Combine: team_volume × player_share
            predictions = team_volumes * player_shares

        else:
            # Default to team volume divided by 11 players
            predictions = team_volumes / 11

        return predictions


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """Stacking ensemble with meta-learner for optimal model combination."""

    def __init__(self, base_models: Dict[str, BaseEstimator] = None,
                 meta_learner=None, use_cv: bool = True):
        """Initialize stacking ensemble.

        Args:
            base_models: Dictionary of base models
            meta_learner: Meta-learner model (default: Ridge)
            use_cv: Use cross-validation for generating meta features
        """
        self.base_models = base_models or {}
        self.meta_learner = meta_learner
        self.use_cv = use_cv
        self.meta_features_train_ = None
        self.fitted_models_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, exposure: Optional[pd.Series] = None):
        """Fit stacking ensemble.

        Args:
            X: Feature matrix
            y: Target values
            exposure: Exposure variable

        Returns:
            Self
        """
        from sklearn.model_selection import KFold
        from sklearn.linear_model import Ridge

        # Use Ridge regression as default meta-learner
        if self.meta_learner is None:
            self.meta_learner = Ridge(alpha=1.0)

        n_samples = len(X)
        n_models = len(self.base_models)

        if self.use_cv:
            # Generate out-of-fold predictions for meta-learner training
            meta_features = np.zeros((n_samples, n_models))
            kf = KFold(n_splits=3, shuffle=True, random_state=42)

            for model_idx, (name, model) in enumerate(self.base_models.items()):
                oof_predictions = np.zeros(n_samples)

                for train_idx, val_idx in kf.split(X):
                    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

                    # Clone model for CV
                    model_clone = model.__class__(**model.get_params() if hasattr(model, 'get_params') else {})

                    # Fit with exposure if needed
                    if exposure is not None:
                        exp_train_cv = exposure.iloc[train_idx]
                        exp_val_cv = exposure.iloc[val_idx]
                        if hasattr(model_clone, 'fit') and 'exposure' in model_clone.fit.__code__.co_varnames:
                            model_clone.fit(X_train_cv, y_train_cv, exposure=exp_train_cv)
                        else:
                            model_clone.fit(X_train_cv, y_train_cv)

                        # Predict with exposure
                        if hasattr(model_clone, 'predict') and 'exposure' in model_clone.predict.__code__.co_varnames:
                            oof_predictions[val_idx] = model_clone.predict(X_val_cv, exposure=exp_val_cv)
                        else:
                            oof_predictions[val_idx] = model_clone.predict(X_val_cv)
                    else:
                        model_clone.fit(X_train_cv, y_train_cv)
                        oof_predictions[val_idx] = model_clone.predict(X_val_cv)

                meta_features[:, model_idx] = oof_predictions

        else:
            # Simple train-validation split for meta features
            from sklearn.model_selection import train_test_split

            X_base, X_meta, y_base, y_meta = train_test_split(X, y, test_size=0.2, random_state=42)
            if exposure is not None:
                exp_base, exp_meta = train_test_split(exposure, test_size=0.2, random_state=42)
            else:
                exp_base = exp_meta = None

            meta_features = np.zeros((len(X_meta), n_models))

            for model_idx, (name, model) in enumerate(self.base_models.items()):
                # Fit on base set
                if exposure is not None and hasattr(model, 'fit') and 'exposure' in model.fit.__code__.co_varnames:
                    model.fit(X_base, y_base, exposure=exp_base)
                else:
                    model.fit(X_base, y_base)

                # Predict on meta set
                if exposure is not None and hasattr(model, 'predict') and 'exposure' in model.predict.__code__.co_varnames:
                    meta_features[:, model_idx] = model.predict(X_meta, exposure=exp_meta)
                else:
                    meta_features[:, model_idx] = model.predict(X_meta)

            y = y_meta  # Use meta set targets for meta-learner

        # Train meta-learner on out-of-fold predictions
        self.meta_learner.fit(meta_features, y)

        # Retrain all base models on full training data
        for name, model in self.base_models.items():
            if exposure is not None and hasattr(model, 'fit') and 'exposure' in model.fit.__code__.co_varnames:
                model.fit(X, y.iloc[:len(X)] if hasattr(y, 'iloc') else y, exposure=exposure)
            else:
                model.fit(X, y.iloc[:len(X)] if hasattr(y, 'iloc') else y)
            self.fitted_models_[name] = model

        return self

    def predict(self, X: pd.DataFrame, exposure: Optional[pd.Series] = None) -> np.ndarray:
        """Make stacked predictions.

        Args:
            X: Feature matrix
            exposure: Exposure variable

        Returns:
            Meta-learner predictions
        """
        n_samples = len(X)
        n_models = len(self.fitted_models_)
        meta_features = np.zeros((n_samples, n_models))

        # Generate base model predictions
        for model_idx, (name, model) in enumerate(self.fitted_models_.items()):
            if exposure is not None and hasattr(model, 'predict') and 'exposure' in model.predict.__code__.co_varnames:
                meta_features[:, model_idx] = model.predict(X, exposure=exposure)
            else:
                meta_features[:, model_idx] = model.predict(X)

        # Use meta-learner to combine predictions
        predictions = self.meta_learner.predict(meta_features)

        # Ensure non-negative predictions for count data
        predictions = np.maximum(predictions, 0)

        return predictions


class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """Weighted ensemble based on validation performance."""

    def __init__(self, models: Dict[str, BaseEstimator] = None,
                 weight_method: str = 'inverse_mae'):
        """Initialize weighted ensemble.

        Args:
            models: Dictionary of models
            weight_method: Method for calculating weights ('inverse_mae', 'softmax')
        """
        self.models = models or {}
        self.weight_method = weight_method
        self.weights_ = None
        self.validation_scores_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, exposure: Optional[pd.Series] = None):
        """Fit models and calculate weights.

        Args:
            X: Feature matrix
            y: Target values
            exposure: Exposure variable

        Returns:
            Self
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error

        # Split data for weight calculation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        if exposure is not None:
            exp_train, exp_val = train_test_split(exposure, test_size=0.2, random_state=42)
        else:
            exp_train = exp_val = None

        self.validation_scores_ = {}

        # Train models and evaluate on validation set
        for name, model in self.models.items():
            # Train on training split
            if exposure is not None and hasattr(model, 'fit') and 'exposure' in model.fit.__code__.co_varnames:
                model.fit(X_train, y_train, exposure=exp_train)
            else:
                model.fit(X_train, y_train)

            # Predict on validation split
            if exposure is not None and hasattr(model, 'predict') and 'exposure' in model.predict.__code__.co_varnames:
                y_pred = model.predict(X_val, exposure=exp_val)
            else:
                y_pred = model.predict(X_val)

            # Calculate validation MAE
            self.validation_scores_[name] = mean_absolute_error(y_val, y_pred)

        # Calculate weights based on validation performance
        if self.weight_method == 'inverse_mae':
            # Inverse MAE weighting
            inverse_scores = {name: 1.0 / score for name, score in self.validation_scores_.items()}
            total = sum(inverse_scores.values())
            self.weights_ = {name: score / total for name, score in inverse_scores.items()}

        elif self.weight_method == 'softmax':
            # Softmax on negative MAE
            import math
            neg_scores = {name: -score for name, score in self.validation_scores_.items()}
            exp_scores = {name: math.exp(score) for name, score in neg_scores.items()}
            total = sum(exp_scores.values())
            self.weights_ = {name: score / total for name, score in exp_scores.items()}

        else:
            # Equal weights fallback
            self.weights_ = {name: 1.0 / len(self.models) for name in self.models}

        # Retrain models on full dataset
        for name, model in self.models.items():
            if exposure is not None and hasattr(model, 'fit') and 'exposure' in model.fit.__code__.co_varnames:
                model.fit(X, y, exposure=exposure)
            else:
                model.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame, exposure: Optional[pd.Series] = None) -> np.ndarray:
        """Make weighted predictions.

        Args:
            X: Feature matrix
            exposure: Exposure variable

        Returns:
            Weighted predictions
        """
        predictions = np.zeros(len(X))

        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                if exposure is not None and 'exposure' in model.predict.__code__.co_varnames:
                    pred = model.predict(X, exposure=exposure)
                else:
                    pred = model.predict(X)

                predictions += self.weights_[name] * pred

        return predictions
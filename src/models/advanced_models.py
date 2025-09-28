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
        if hasattr(self.global_model_, 'fit') and 'exposure' in self.global_model_.fit.__code__.co_varnames:
            self.global_model_.fit(X, y, exposure=exposure)
        else:
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
            for idx, row in X.iterrows():
                position = row[self.position_column]

                # Use position-specific model if available
                if position in self.models_:
                    model = self.models_[position]
                    X_row = X.iloc[[idx]]

                    if exposure is not None:
                        exposure_row = exposure.iloc[[idx]] if isinstance(exposure, pd.Series) else exposure[[idx]]
                        if hasattr(model, 'predict') and 'exposure' in model.predict.__code__.co_varnames:
                            pred = model.predict(X_row, exposure=exposure_row)
                        else:
                            pred = model.predict(X_row)
                    else:
                        pred = model.predict(X_row)

                    predictions[idx] = pred[0]
                else:
                    # Use global model as fallback
                    X_row = X.iloc[[idx]]
                    if exposure is not None:
                        exposure_row = exposure.iloc[[idx]] if isinstance(exposure, pd.Series) else exposure[[idx]]
                        if hasattr(self.global_model_, 'predict') and 'exposure' in self.global_model_.predict.__code__.co_varnames:
                            pred = self.global_model_.predict(X_row, exposure=exposure_row)
                        else:
                            pred = self.global_model_.predict(X_row)
                    else:
                        pred = self.global_model_.predict(X_row)

                    predictions[idx] = pred[0]
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

    def __init__(self, use_exposure: bool = True, **xgb_params):
        """Initialize XGBoost regressor.

        Args:
            use_exposure: Whether to use exposure (minutes played)
            **xgb_params: Parameters for XGBoost
        """
        self.use_exposure = use_exposure
        self.xgb_params = xgb_params
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

        # Default XGBoost parameters for count regression
        default_params = {
            'objective': 'count:poisson',  # Poisson regression for counts
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }

        # Update with user parameters
        params = {**default_params, **self.xgb_params}

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
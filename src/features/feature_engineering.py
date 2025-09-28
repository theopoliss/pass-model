"""Feature engineering for pass prediction."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Advanced feature engineering for pass prediction."""

    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.feature_stats = {}
        self.ewma_alpha = 0.3  # Exponential weight parameter

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables.

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with interaction features added
        """
        df = df.copy()

        # Position × Home/Away interaction
        if "position_encoded" in df.columns and "is_home" in df.columns:
            df["position_home_interaction"] = df["position_encoded"] * df["is_home"]

        # Team strength × Minutes played
        if "team_strength_diff" in df.columns and "minutes_played" in df.columns:
            df["strength_minutes_interaction"] = df["team_strength_diff"] * df["minutes_played"] / 90

        # Form × Opponent strength
        if "passes_attempted_rolling_5" in df.columns and "opponent_points_per_game" in df.columns:
            df["form_vs_opponent"] = df["passes_attempted_rolling_5"] * (1 - df["opponent_points_per_game"] / 3)

        return df

    def create_polynomial_features(self, df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for specified columns.

        Args:
            df: DataFrame with base features
            columns: Columns to create polynomial features for
            degree: Polynomial degree

        Returns:
            DataFrame with polynomial features added
        """
        df = df.copy()

        for col in columns:
            if col in df.columns:
                for d in range(2, degree + 1):
                    df[f"{col}_pow{d}"] = df[col] ** d

        return df

    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio-based features.

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with ratio features added
        """
        df = df.copy()

        # Pass completion rate from rolling data
        if "passes_completed_rolling_5" in df.columns and "passes_attempted_rolling_5" in df.columns:
            df["pass_completion_rate_rolling"] = (
                df["passes_completed_rolling_5"] / df["passes_attempted_rolling_5"].replace(0, 1)
            ).clip(0, 1)

        # Minutes percentage
        if "minutes_played" in df.columns:
            df["minutes_percentage"] = df["minutes_played"] / 90

        # Goal difference ratio
        if "team_avg_goals" in df.columns and "opponent_avg_goals" in df.columns:
            df["goal_ratio"] = df["team_avg_goals"] / (df["opponent_avg_goals"] + 0.1)  # Avoid division by zero

        return df

    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical indicator features.

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with categorical features added
        """
        df = df.copy()

        # Game outcome categories
        if "goal_difference" in df.columns:
            df["game_result"] = pd.cut(
                df["goal_difference"],
                bins=[-np.inf, -1, 0, np.inf],
                labels=["loss", "draw", "win"]
            )
            # One-hot encode
            result_dummies = pd.get_dummies(df["game_result"], prefix="result")
            df = pd.concat([df, result_dummies], axis=1)

        # Minutes played categories
        if "minutes_played" in df.columns:
            df["minutes_category"] = pd.cut(
                df["minutes_played"],
                bins=[0, 30, 60, 90],
                labels=["low", "medium", "high"]
            )
            minutes_dummies = pd.get_dummies(df["minutes_category"], prefix="minutes_cat")
            df = pd.concat([df, minutes_dummies], axis=1)

        return df

    def create_zone_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on pass zones.

        Args:
            df: DataFrame with zone features

        Returns:
            DataFrame with zone-based features added
        """
        df = df.copy()

        # Calculate zone percentages
        if all(col in df.columns for col in ['passes_from_defensive', 'passes_from_middle', 'passes_from_attacking', 'passes_attempted']):
            total_passes = df['passes_attempted'].replace(0, 1)  # Avoid division by zero
            df['pct_passes_from_defensive'] = df['passes_from_defensive'] / total_passes
            df['pct_passes_from_middle'] = df['passes_from_middle'] / total_passes
            df['pct_passes_from_attacking'] = df['passes_from_attacking'] / total_passes

        # Progressive pass percentage
        if 'progressive_passes' in df.columns and 'passes_attempted' in df.columns:
            df['progressive_pass_pct'] = df['progressive_passes'] / df['passes_attempted'].replace(0, 1)

        # Direction percentages
        if all(col in df.columns for col in ['forward_passes', 'backward_passes', 'sideways_passes', 'passes_attempted']):
            total_passes = df['passes_attempted'].replace(0, 1)
            df['forward_pass_pct'] = df['forward_passes'] / total_passes
            df['backward_pass_pct'] = df['backward_passes'] / total_passes

        # Pressure percentage
        if 'passes_under_pressure' in df.columns and 'passes_attempted' in df.columns:
            df['pressure_pass_pct'] = df['passes_under_pressure'] / df['passes_attempted'].replace(0, 1)

        return df

    def create_game_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on game state.

        Args:
            df: DataFrame with game state features

        Returns:
            DataFrame with game state features added
        """
        df = df.copy()

        # Calculate percentage of passes in different game states
        if all(col in df.columns for col in ['passes_while_winning', 'passes_while_losing', 'passes_while_drawing', 'passes_attempted']):
            total_passes = df['passes_attempted'].replace(0, 1)
            df['pct_passes_winning'] = df['passes_while_winning'] / total_passes
            df['pct_passes_losing'] = df['passes_while_losing'] / total_passes
            df['pct_passes_drawing'] = df['passes_while_drawing'] / total_passes

        return df

    def create_exponential_weighted_features(self, df: pd.DataFrame,
                                           columns: List[str],
                                           alpha: float = None) -> pd.DataFrame:
        """Create exponentially weighted moving average features.

        Args:
            df: DataFrame sorted by player and match date
            columns: Columns to create EWMA for
            alpha: Smoothing parameter (0 < alpha <= 1), higher = more weight on recent

        Returns:
            DataFrame with EWMA features added
        """
        if alpha is None:
            alpha = self.ewma_alpha

        df = df.copy()

        # Sort by player and date to ensure proper time ordering
        if 'player' in df.columns and 'match_date' in df.columns:
            df = df.sort_values(['player', 'match_date'])

            for col in columns:
                if col in df.columns:
                    ewma_col = f"{col}_ewma"

                    # Calculate EWMA per player
                    df[ewma_col] = (df.groupby('player')[col]
                                   .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean()
                                             .shift(1)))  # Shift to avoid leakage

                    # Fill NaN values (first game for each player) with overall mean
                    df[ewma_col].fillna(df[col].mean(), inplace=True)

                    # Also calculate EWMA standard deviation for consistency
                    ewma_std_col = f"{col}_ewma_std"
                    df[ewma_std_col] = (df.groupby('player')[col]
                                       .transform(lambda x: x.ewm(alpha=alpha, adjust=False).std()
                                                 .shift(1)))
                    df[ewma_std_col].fillna(df[col].std(), inplace=True)

        return df

    def create_lag_features(self, df: pd.DataFrame, group_col: str = "player", lag_periods: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """Create lag features for time series aspects.

        Args:
            df: DataFrame sorted by date
            group_col: Column to group by (usually player)
            lag_periods: List of lag periods to create

        Returns:
            DataFrame with lag features added
        """
        df = df.sort_values(["player", "match_date"]).copy()

        lag_columns = ["passes_attempted", "minutes_played"]

        for col in lag_columns:
            if col in df.columns:
                for lag in lag_periods:
                    df[f"{col}_lag_{lag}"] = df.groupby(group_col)[col].shift(lag)

        return df

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
        """Handle missing values in features.

        Args:
            df: DataFrame with features
            strategy: Strategy for imputation ('median', 'mean', 'zero')

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if df[col].isna().any():
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "zero":
                    df[col] = df[col].fillna(0)

        return df

    def scale_features(self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Scale features using StandardScaler.

        Args:
            X_train: Training features
            X_test: Test features (optional)

        Returns:
            Scaled features
        """
        # Fit scaler on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled

        return X_train_scaled, None

    def select_features(self, df: pd.DataFrame, target: str, threshold: float = 0.01) -> List[str]:
        """Select features based on correlation with target.

        Args:
            df: DataFrame with features and target
            target: Target column name
            threshold: Minimum absolute correlation to keep feature

        Returns:
            List of selected feature names
        """
        correlations = df.corr()[target].abs().sort_values(ascending=False)
        selected_features = correlations[correlations > threshold].index.tolist()

        # Remove target from features
        if target in selected_features:
            selected_features.remove(target)

        return selected_features

    def engineer_features(self, df: pd.DataFrame, feature_set: str = "basic") -> pd.DataFrame:
        """Main feature engineering pipeline.

        Args:
            df: DataFrame with base features
            feature_set: Feature set to create ('basic', 'intermediate', 'advanced')

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        if feature_set in ["basic", "intermediate", "advanced"]:
            # Basic features - always included
            df = self.create_ratio_features(df)
            # Add zone and game state features if available
            df = self.create_zone_based_features(df)
            df = self.create_game_state_features(df)

        if feature_set in ["intermediate", "advanced"]:
            # Intermediate features
            df = self.create_interaction_features(df)
            df = self.create_categorical_features(df)

        if feature_set == "advanced":
            # Advanced features
            df = self.create_polynomial_features(
                df,
                columns=["minutes_played", "team_strength_diff"],
                degree=2
            )
            df = self.create_lag_features(df)

            # Add exponentially weighted moving averages for key metrics
            ewma_columns = ['passes_attempted', 'passes_completed', 'progressive_passes',
                          'passes_under_pressure', 'forward_passes']
            # Only apply EWMA to columns that exist
            ewma_columns = [col for col in ewma_columns if col in df.columns]
            if ewma_columns:
                df = self.create_exponential_weighted_features(df, ewma_columns)

        # Handle missing values
        df = self.handle_missing_values(df)

        return df
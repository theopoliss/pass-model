"""Data processing pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class PassDataProcessor:
    """Process raw pass data for modeling."""

    def __init__(self, min_minutes: int = 15, exclude_goalkeepers: bool = True):
        """Initialize the processor.

        Args:
            min_minutes: Minimum minutes played to include a player
            exclude_goalkeepers: Whether to exclude goalkeepers from analysis
        """
        self.min_minutes = min_minutes
        self.exclude_goalkeepers = exclude_goalkeepers
        self.position_encoder = LabelEncoder()
        self.team_encoder = LabelEncoder()

    def clean_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean player-level data.

        Args:
            df: Raw player data

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        # Filter by minutes played
        df_clean = df_clean[df_clean["minutes_played"] >= self.min_minutes]

        # Exclude goalkeepers if specified
        if self.exclude_goalkeepers:
            df_clean = df_clean[df_clean["position"] != "Goalkeeper"]

        # Handle missing values
        df_clean["avg_pass_length"] = df_clean["avg_pass_length"].fillna(
            df_clean.groupby("position")["avg_pass_length"].transform("median")
        )

        # Remove rows with missing critical values
        critical_columns = ["passes_attempted", "team", "player"]
        df_clean = df_clean.dropna(subset=critical_columns)

        return df_clean

    def encode_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode positions as numerical values.

        Args:
            df: DataFrame with position column

        Returns:
            DataFrame with encoded positions
        """
        df = df.copy()

        # Create position groups
        position_groups = {
            "Goalkeeper": "GK",
            "Center Back": "DEF",
            "Left Back": "DEF",
            "Right Back": "DEF",
            "Left Wing Back": "DEF",
            "Right Wing Back": "DEF",
            "Center Defensive Midfield": "MID",
            "Center Midfield": "MID",
            "Center Attacking Midfield": "MID",
            "Left Midfield": "MID",
            "Right Midfield": "MID",
            "Left Wing": "FWD",
            "Right Wing": "FWD",
            "Center Forward": "FWD",
            "Left Center Forward": "FWD",
            "Right Center Forward": "FWD",
            "Secondary Striker": "FWD"
        }

        # Map positions to groups
        df["position_group"] = df["position"].map(position_groups).fillna("OTHER")

        # Encode position groups
        df["position_encoded"] = self.position_encoder.fit_transform(df["position_group"])

        return df

    def calculate_team_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team strength metrics.

        Args:
            df: DataFrame with match results

        Returns:
            DataFrame with team strength features
        """
        df = df.copy()

        # Calculate team win rate (simple version)
        team_results = []

        for team in df["team"].unique():
            team_games = df[df["team"] == team]
            wins = (team_games["goal_difference"] > 0).sum()
            draws = (team_games["goal_difference"] == 0).sum()
            losses = (team_games["goal_difference"] < 0).sum()
            total = len(team_games)

            if total > 0:
                win_rate = wins / total
                points_per_game = (3 * wins + draws) / total
            else:
                win_rate = 0.5
                points_per_game = 1.0

            team_results.append({
                "team": team,
                "team_win_rate": win_rate,
                "team_points_per_game": points_per_game,
                "team_avg_goals": team_games["team_goals"].mean(),
                "team_avg_goals_against": team_games["opponent_goals"].mean()
            })

        team_strength_df = pd.DataFrame(team_results)

        # Merge back to main dataframe
        df = df.merge(team_strength_df, on="team", how="left")

        # Get opponent team name
        df["opponent_team"] = df.apply(
            lambda x: x["away_team"] if x["is_home"] else x["home_team"], axis=1
        )

        # Add opponent strength
        opponent_strength = team_strength_df.rename(
            columns={col: f"opponent_{col.replace('team_', '')}"
                    for col in team_strength_df.columns if col != "team"}
        ).rename(columns={"team": "opponent_team"})

        df = df.merge(opponent_strength, on="opponent_team", how="left")

        # Calculate strength difference
        df["team_strength_diff"] = df["team_points_per_game"] - df["opponent_points_per_game"]

        return df

    def add_rolling_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Add rolling average features for players.

        Args:
            df: DataFrame sorted by date
            window: Number of games for rolling window

        Returns:
            DataFrame with rolling features
        """
        df = df.sort_values(["player", "match_date"]).copy()

        # Calculate rolling averages per player
        rolling_cols = ["passes_attempted", "passes_completed", "minutes_played"]

        for col in rolling_cols:
            df[f"{col}_rolling_{window}"] = (
                df.groupby("player")[col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            )

        # Fill NaN values (first games for each player)
        for col in rolling_cols:
            df[f"{col}_rolling_{window}"] = df[f"{col}_rolling_{window}"].fillna(
                df.groupby("position_group")[col].transform("median")
            )

        return df

    def create_model_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create final features for modeling.

        Args:
            df: Processed DataFrame

        Returns:
            Tuple of (feature matrix, feature names)
        """
        feature_columns = [
            # Position
            "position_encoded",

            # Match context
            "is_home",
            "minutes_played",

            # Team strength
            "team_strength_diff",
            "team_points_per_game",
            "opponent_points_per_game",
            "team_avg_goals",
            "opponent_avg_goals",

            # Player form (rolling averages)
            "passes_attempted_rolling_5",
            "passes_completed_rolling_5",
            "minutes_played_rolling_5",

            # Pass characteristics
            "avg_pass_length"
        ]

        # Keep only available features
        available_features = [col for col in feature_columns if col in df.columns]

        # Convert boolean to int
        for col in available_features:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)

        return df[available_features], available_features

    def process_data(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Full processing pipeline.

        Args:
            raw_df: Raw data from StatsBomb

        Returns:
            Tuple of (processed_df, feature_matrix, feature_names)
        """
        logger.info("Starting data processing pipeline")

        # Clean data
        df = self.clean_player_data(raw_df)
        logger.info(f"After cleaning: {len(df)} records")

        # Encode positions
        df = self.encode_positions(df)

        # Calculate team strength
        df = self.calculate_team_strength(df)

        # Add rolling features
        df = self.add_rolling_features(df)

        # Create model features
        feature_matrix, feature_names = self.create_model_features(df)

        logger.info(f"Created {len(feature_names)} features for {len(df)} observations")

        return df, feature_matrix, feature_names
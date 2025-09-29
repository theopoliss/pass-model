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
        self.player_stats = {}  # Store player career statistics

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

    def calculate_player_statistics(self, df: pd.DataFrame, train_mask: Optional[pd.Series] = None) -> pd.DataFrame:
        """Calculate player-specific statistics.

        Args:
            df: DataFrame with player data
            train_mask: Boolean mask for training data (to avoid leakage)

        Returns:
            DataFrame with player statistics added
        """
        df = df.copy()

        # Use only training data for calculating stats if mask provided
        stats_df = df[train_mask] if train_mask is not None else df

        # Calculate player career statistics
        player_stats = stats_df.groupby('player').agg({
            'passes_attempted': ['mean', 'std', 'count'],
            'minutes_played': 'mean',
            'passes_completed': 'mean'
        })

        # Flatten column names
        player_stats.columns = ['_'.join(col).strip() for col in player_stats.columns]
        player_stats.rename(columns={
            'passes_attempted_mean': 'player_career_avg_passes',
            'passes_attempted_std': 'player_pass_consistency',
            'passes_attempted_count': 'player_games_played',
            'minutes_played_mean': 'player_avg_minutes',
            'passes_completed_mean': 'player_career_avg_completed'
        }, inplace=True)

        # Normalize per 90 minutes
        player_stats['player_career_avg_passes_per90'] = (
            player_stats['player_career_avg_passes'] / player_stats['player_avg_minutes'] * 90
        )
        player_stats['player_career_completion_rate'] = (
            player_stats['player_career_avg_completed'] / player_stats['player_career_avg_passes']
        ).clip(0, 1)

        # Calculate position group averages for fallback
        position_stats = stats_df.groupby('position_group').agg({
            'passes_attempted': 'mean',
            'passes_completed': 'mean',
            'minutes_played': 'mean'
        }).rename(columns={
            'passes_attempted': 'position_avg_passes',
            'passes_completed': 'position_avg_completed',
            'minutes_played': 'position_avg_minutes'
        })

        # Merge player stats
        df = df.merge(player_stats[['player_career_avg_passes', 'player_pass_consistency',
                                     'player_games_played', 'player_career_avg_passes_per90',
                                     'player_career_completion_rate']],
                      left_on='player', right_index=True, how='left')

        # Merge position stats for fallback
        df = df.merge(position_stats, left_on='position_group', right_index=True, how='left')

        # Handle new/unseen players with position-based fallback
        df['player_career_avg_passes'] = df['player_career_avg_passes'].fillna(df['position_avg_passes'])
        df['player_pass_consistency'] = df['player_pass_consistency'].fillna(
            df.groupby('position_group')['passes_attempted'].transform('std')
        )
        df['player_games_played'] = df['player_games_played'].fillna(0)
        df['player_career_avg_passes_per90'] = df['player_career_avg_passes_per90'].fillna(
            df['position_avg_passes'] / df['position_avg_minutes'] * 90
        )
        df['player_career_completion_rate'] = df['player_career_completion_rate'].fillna(
            df['position_avg_completed'] / df['position_avg_passes']
        )

        # Create experience level feature
        df['player_experience_level'] = pd.cut(
            df['player_games_played'],
            bins=[-0.1, 5, 15, 30, np.inf],
            labels=['new', 'developing', 'experienced', 'veteran']
        )

        # One-hot encode experience level
        experience_dummies = pd.get_dummies(df['player_experience_level'], prefix='exp')
        df = pd.concat([df, experience_dummies], axis=1)

        # Drop temporary columns
        df = df.drop(columns=['position_avg_passes', 'position_avg_completed', 'position_avg_minutes'])

        return df

    def add_player_form_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Add player-specific form features.

        Args:
            df: DataFrame sorted by player and date
            window: Number of games for form calculation

        Returns:
            DataFrame with player form features
        """
        df = df.sort_values(['player', 'match_date']).copy()

        # Calculate player-specific recent form
        df['player_recent_passes_avg'] = (
            df.groupby('player')['passes_attempted']
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )

        df['player_recent_completion_rate'] = (
            df.groupby('player')['passes_completed']
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1)) /
            df.groupby('player')['passes_attempted']
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        ).clip(0, 1)

        # Calculate form trend (recent vs career average)
        df['player_form_trend'] = (
            df['player_recent_passes_avg'] / df['player_career_avg_passes'].replace(0, 1)
        ).fillna(1.0)

        # Fill NaN values with career averages
        df['player_recent_passes_avg'] = df['player_recent_passes_avg'].fillna(df['player_career_avg_passes'])
        df['player_recent_completion_rate'] = df['player_recent_completion_rate'].fillna(df['player_career_completion_rate'])

        return df

    def create_model_features(self, df: pd.DataFrame, use_formation_features: bool = False) -> Tuple[pd.DataFrame, List[str]]:
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

            # Player-specific features
            "player_career_avg_passes",
            "player_career_avg_passes_per90",
            "player_pass_consistency",
            "player_games_played",
            "player_career_completion_rate",

            # Player form
            "player_recent_passes_avg",
            "player_recent_completion_rate",
            "player_form_trend",

            # Player experience level (one-hot encoded)
            "exp_new",
            "exp_developing",
            "exp_experienced",
            "exp_veteran",

            # Player form (rolling averages)
            "passes_attempted_rolling_5",
            "passes_completed_rolling_5",
            "minutes_played_rolling_5",

            # Pass characteristics
            "avg_pass_length"
        ]

        # Add formation features if requested
        if use_formation_features:
            formation_features = [
                # Numerical advantages
                "midfield_advantage",
                "defensive_advantage",
                "attacking_advantage",

                # Position-tactical interaction
                "position_tactical_impact",
                "tactical_workload",

                # Team composition
                "team_midfielders",
                "team_defenders",
                "team_forwards",
                "opp_midfielders",
                "opp_defenders",
                "opp_forwards",
            ]

            # Add formation-specific one-hot columns (dynamically find them)
            formation_encoded_cols = [col for col in df.columns if
                                     col.startswith('team_formation_') or
                                     col.startswith('opp_formation_')]

            feature_columns.extend(formation_features)
            feature_columns.extend(formation_encoded_cols)

        # Keep only available features
        available_features = [col for col in feature_columns if col in df.columns]

        # Convert boolean to int
        for col in available_features:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)

        return df[available_features], available_features

    def process_data(self, raw_df: pd.DataFrame, train_mask: Optional[pd.Series] = None,
                    use_formation_features: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Full processing pipeline.

        Args:
            raw_df: Raw data from StatsBomb
            train_mask: Boolean mask for training data (to avoid leakage in player stats)
            use_formation_features: Whether to include formation/tactical features

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

        # Calculate player-specific statistics
        df = self.calculate_player_statistics(df, train_mask)
        logger.info("Added player-specific features")

        # Add rolling features
        df = self.add_rolling_features(df)

        # Add player form features
        df = self.add_player_form_features(df)

        # Add formation features if requested
        if use_formation_features and ('team_formation' in df.columns or 'opponent_formation' in df.columns):
            from src.features.feature_engineering import TacticalFeatureEngineer
            tactical_engineer = TacticalFeatureEngineer()
            df = tactical_engineer.engineer_tactical_features(df)
            logger.info("Added tactical formation features")

        # Create model features
        feature_matrix, feature_names = self.create_model_features(df, use_formation_features)

        logger.info(f"Created {len(feature_names)} features for {len(df)} observations")

        return df, feature_matrix, feature_names
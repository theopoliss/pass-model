"""FBRef-specific data processor for player match statistics."""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class FBRefProcessor:
    """Process FBRef player match data for modeling."""

    # Position mapping to simplified groups
    POSITION_MAPPING = {
        # Defenders
        'DF': 'DEF', 'LB': 'DEF', 'RB': 'DEF', 'CB': 'DEF', 'WB': 'DEF',
        'DF,MF': 'DEF',  # Defensive midfielder counted as defender

        # Midfielders
        'MF': 'MID', 'CM': 'MID', 'DM': 'MID', 'AM': 'MID', 'LM': 'MID', 'RM': 'MID',
        'MF,FW': 'MID',  # Attacking midfielder

        # Forwards
        'FW': 'FWD', 'LW': 'FWD', 'RW': 'FWD', 'CF': 'FWD', 'ST': 'FWD',

        # Goalkeeper
        'GK': 'GK'
    }

    def __init__(self):
        """Initialize the processor."""
        self.position_encoder = LabelEncoder()
        self.team_encoder = LabelEncoder()
        self.player_stats = {}  # Cache for player historical stats
        self.team_strength = {}  # Team strength metrics

    def process_data(
        self,
        raw_df: pd.DataFrame,
        train: bool = True
    ) -> pd.DataFrame:
        """Process raw FBRef data into modeling features.

        Args:
            raw_df: Raw data from FBRef collector
            train: Whether this is training data (fits encoders)

        Returns:
            Processed DataFrame ready for modeling
        """
        df = raw_df.copy()

        # Clean and validate data
        df = self._clean_data(df)

        # Process positions
        df = self._process_positions(df, train)

        # Calculate team metrics
        df = self._calculate_team_metrics(df)

        # Create player features
        df = self._create_player_features(df)

        # Create match context features
        df = self._create_match_features(df)

        # Create rolling averages
        df = self._create_rolling_features(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Select and order columns
        df = self._select_features(df)

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Ensure numeric columns are numeric
        numeric_cols = [
            'minutes_played', 'passes_attempted', 'passes_completed',
            'xG', 'xA', 'touches', 'progressive_passes'
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Calculate pass accuracy if not present
        if 'pass_accuracy' not in df.columns and 'passes_completed' in df.columns:
            df['pass_accuracy'] = np.where(
                df['passes_attempted'] > 0,
                df['passes_completed'] / df['passes_attempted'] * 100,
                0
            )

        # Filter out players with very few minutes
        df = df[df['minutes_played'] >= 10].copy()

        # Remove goalkeepers for pass prediction
        if 'position' in df.columns:
            df = df[~df['position'].str.contains('GK', na=False)].copy()

        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        return df

    def _process_positions(self, df: pd.DataFrame, train: bool) -> pd.DataFrame:
        """Process position data.

        Args:
            df: DataFrame with position column
            train: Whether to fit encoder

        Returns:
            DataFrame with encoded positions
        """
        if 'position' not in df.columns:
            df['position'] = 'MID'  # Default if missing

        # Map to simplified positions
        df['position_group'] = df['position'].map(self.POSITION_MAPPING)
        df['position_group'] = df['position_group'].fillna('MID')  # Default

        # Encode positions
        if train:
            df['position_encoded'] = self.position_encoder.fit_transform(df['position_group'])
        else:
            # Handle unknown positions
            df['position_encoded'] = df['position_group'].apply(
                lambda x: self.position_encoder.transform([x])[0]
                if x in self.position_encoder.classes_ else -1
            )

        return df

    def _calculate_team_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team strength metrics.

        Args:
            df: DataFrame with team data

        Returns:
            DataFrame with team metrics
        """
        # Calculate team average passes per game
        team_stats = df.groupby('home_team').agg({
            'passes_attempted': 'mean',
            'passes_completed': 'mean',
            'xG': 'mean'
        }).rename(columns=lambda x: f'team_avg_{x}')

        # Simple team strength based on historical performance
        team_stats['team_strength'] = (
            team_stats['team_avg_passes_attempted'] / team_stats['team_avg_passes_attempted'].mean() * 0.5 +
            team_stats['team_avg_xG'] / team_stats['team_avg_xG'].mean() * 0.5
        )

        self.team_strength = team_stats['team_strength'].to_dict()

        # Add team strength to main df
        df['home_team_strength'] = df['home_team'].map(self.team_strength).fillna(1.0)
        df['away_team_strength'] = df['away_team'].map(self.team_strength).fillna(1.0)

        # Determine if player is home or away
        # This is simplified - in reality would need lineup data
        df['is_home'] = 1  # Placeholder

        df['team_strength'] = np.where(
            df['is_home'] == 1,
            df['home_team_strength'],
            df['away_team_strength']
        )

        df['opponent_strength'] = np.where(
            df['is_home'] == 1,
            df['away_team_strength'],
            df['home_team_strength']
        )

        df['team_strength_diff'] = df['team_strength'] - df['opponent_strength']

        return df

    def _create_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create player-specific features.

        Args:
            df: DataFrame with player data

        Returns:
            DataFrame with player features
        """
        # Calculate player career averages
        player_stats = df.groupby('player_name').agg({
            'passes_attempted': ['mean', 'std', 'count'],
            'pass_accuracy': 'mean',
            'minutes_played': 'mean',
            'xG': 'mean',
            'xA': 'mean',
            'touches': 'mean',
            'progressive_passes': 'mean'
        })

        player_stats.columns = ['_'.join(col).strip() for col in player_stats.columns.values]
        player_stats.columns = [
            'player_career_avg_passes', 'player_pass_std', 'player_games_played',
            'player_career_pass_accuracy', 'player_avg_minutes',
            'player_avg_xG', 'player_avg_xA', 'player_avg_touches',
            'player_avg_progressive_passes'
        ]

        # Calculate passes per 90
        player_stats['player_career_passes_per90'] = (
            player_stats['player_career_avg_passes'] /
            player_stats['player_avg_minutes'] * 90
        )

        # Pass consistency (inverse of coefficient of variation)
        player_stats['player_pass_consistency'] = 1 / (
            player_stats['player_pass_std'] /
            player_stats['player_career_avg_passes'] + 0.1
        )

        # Merge back
        df = df.merge(
            player_stats.reset_index(),
            on='player_name',
            how='left'
        )

        return df

    def _create_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create match context features.

        Args:
            df: DataFrame with match data

        Returns:
            DataFrame with match features
        """
        # Expected match difficulty (based on team strengths)
        df['match_importance'] = abs(df['team_strength_diff'])

        # Normalize minutes to per-90
        df['minutes_ratio'] = df['minutes_played'] / 90.0

        # Create position-specific interactions
        df['position_x_team_strength'] = df['position_encoded'] * df['team_strength']
        df['position_x_minutes'] = df['position_encoded'] * df['minutes_played']

        # FBRef specific: use advanced metrics
        if 'progressive_passes' in df.columns:
            df['progressive_pass_rate'] = np.where(
                df['passes_attempted'] > 0,
                df['progressive_passes'] / df['passes_attempted'],
                0
            )

        if 'touches' in df.columns:
            df['passes_per_touch'] = np.where(
                df['touches'] > 0,
                df['passes_attempted'] / df['touches'],
                0
            )

        return df

    def _create_rolling_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Create rolling average features.

        Args:
            df: DataFrame with player match data
            window: Number of games for rolling window

        Returns:
            DataFrame with rolling features
        """
        # Sort by player and date
        df = df.sort_values(['player_name', 'date'])

        # Calculate rolling averages per player
        rolling_cols = ['passes_attempted', 'pass_accuracy', 'minutes_played', 'xG', 'xA']

        for col in rolling_cols:
            if col in df.columns:
                df[f'{col}_rolling_{window}'] = df.groupby('player_name')[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )

        # Form trend (recent vs career average)
        df['form_trend'] = (
            df[f'passes_attempted_rolling_{window}'] /
            df['player_career_avg_passes'].replace(0, 1)
        ).fillna(1.0)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values.

        Args:
            df: DataFrame with potential missing values

        Returns:
            DataFrame with handled missing values
        """
        # Fill numeric columns with 0 or median
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col.endswith('_rolling_5'):
                # For rolling features, use the career average
                base_col = col.replace('_rolling_5', '')
                if f'player_career_avg_{base_col}' in df.columns:
                    df[col] = df[col].fillna(df[f'player_career_avg_{base_col}'])
                else:
                    df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)

        return df

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order final features.

        Args:
            df: DataFrame with all features

        Returns:
            DataFrame with selected features
        """
        # Core features for modeling
        feature_cols = [
            # Target
            'passes_attempted',

            # Player identifiers
            'player_name',
            'date',
            'match_id',

            # Position
            'position_encoded',
            'position_group',

            # Match context
            'minutes_played',
            'is_home',
            'team_strength_diff',
            'opponent_strength',
            'match_importance',

            # Player features
            'player_career_avg_passes',
            'player_career_passes_per90',
            'player_pass_consistency',
            'player_games_played',
            'player_career_pass_accuracy',

            # Advanced FBRef metrics
            'player_avg_xG',
            'player_avg_xA',
            'player_avg_touches',
            'player_avg_progressive_passes',

            # Rolling features
            'passes_attempted_rolling_5',
            'pass_accuracy_rolling_5',
            'form_trend',

            # FBRef specific
            'progressive_pass_rate',
            'passes_per_touch',

            # Interactions
            'position_x_team_strength',
            'position_x_minutes'
        ]

        # Keep only available features
        available_features = [col for col in feature_cols if col in df.columns]

        return df[available_features]
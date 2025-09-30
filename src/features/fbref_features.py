"""FBRef-specific feature engineering leveraging advanced metrics."""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class FBRefFeatureEngineer:
    """Feature engineering specific to FBRef's advanced metrics."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.feature_importance = {}

    def engineer_features(
        self,
        df: pd.DataFrame,
        include_interactions: bool = True,
        include_ratios: bool = True
    ) -> pd.DataFrame:
        """Create advanced features from FBRef data.

        Args:
            df: Processed DataFrame from FBRefProcessor
            include_interactions: Whether to include interaction features
            include_ratios: Whether to include ratio features

        Returns:
            DataFrame with additional engineered features
        """
        df = df.copy()

        # Create advanced metric features
        df = self._create_advanced_metric_features(df)

        # Create position-specific features
        df = self._create_position_specific_features(df)

        # Create team style features
        df = self._create_team_style_features(df)

        # Create player role features
        df = self._create_player_role_features(df)

        if include_ratios:
            df = self._create_ratio_features(df)

        if include_interactions:
            df = self._create_interaction_features(df)

        # Create composite scores
        df = self._create_composite_scores(df)

        return df

    def _create_advanced_metric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from FBRef's advanced metrics.

        Args:
            df: DataFrame with FBRef data

        Returns:
            DataFrame with advanced metric features
        """
        # Expected involvement (xG + xA contribution)
        if 'player_avg_xG' in df.columns and 'player_avg_xA' in df.columns:
            df['expected_involvement'] = df['player_avg_xG'] + df['player_avg_xA']

            # Relative to position average
            position_avg_involvement = df.groupby('position_group')['expected_involvement'].transform('mean')
            df['involvement_vs_position'] = df['expected_involvement'] / (position_avg_involvement + 0.01)

        # Progressive action rate
        if 'player_avg_progressive_passes' in df.columns and 'player_career_avg_passes' in df.columns:
            df['progressive_rate'] = (
                df['player_avg_progressive_passes'] /
                (df['player_career_avg_passes'] + 1)
            )

            # Is player a progressive passer?
            df['is_progressive_passer'] = (df['progressive_rate'] > df['progressive_rate'].quantile(0.75)).astype(int)

        # Touch efficiency
        if 'player_avg_touches' in df.columns:
            df['passes_per_touch_avg'] = (
                df['player_career_avg_passes'] /
                (df['player_avg_touches'] + 1)
            )

            # High/low touch player
            df['is_high_touch'] = (
                df['player_avg_touches'] > df['player_avg_touches'].median()
            ).astype(int)

        return df

    def _create_position_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific features.

        Args:
            df: DataFrame with position data

        Returns:
            DataFrame with position-specific features
        """
        # Position-specific passing expectations
        position_pass_avg = df.groupby('position_group')['player_career_avg_passes'].transform('mean')
        df['position_expected_passes'] = position_pass_avg

        # Player deviation from position norm
        df['passes_vs_position_norm'] = (
            df['player_career_avg_passes'] /
            (df['position_expected_passes'] + 1)
        )

        # Position-specific form
        if 'passes_attempted_rolling_5' in df.columns:
            df['position_form'] = (
                df['passes_attempted_rolling_5'] /
                (df['position_expected_passes'] + 1)
            )

        # Create position dummy variables for interactions
        position_dummies = pd.get_dummies(df['position_group'], prefix='pos')
        for col in position_dummies.columns:
            df[col] = position_dummies[col]

        return df

    def _create_team_style_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team playing style features.

        Args:
            df: DataFrame with team data

        Returns:
            DataFrame with team style features
        """
        # Team possession style (based on average touches and passes)
        if 'home_team' in df.columns and 'player_avg_touches' in df.columns:
            team_touch_avg = df.groupby('home_team')['player_avg_touches'].transform('mean')
            df['team_possession_style'] = team_touch_avg / (df['player_avg_touches'].mean() + 0.01)

            # High/low possession team
            df['is_possession_team'] = (df['team_possession_style'] > 1.1).astype(int)
        else:
            # Default values if team columns not available
            df['team_possession_style'] = 1.0
            df['is_possession_team'] = 0

        # Team progressive style
        if 'home_team' in df.columns and 'player_avg_progressive_passes' in df.columns:
            team_prog_avg = df.groupby('home_team')['player_avg_progressive_passes'].transform('mean')
            df['team_progressive_style'] = team_prog_avg / (df['player_avg_progressive_passes'].mean() + 0.01)
        else:
            df['team_progressive_style'] = 1.0

        # Match style compatibility
        df['style_matchup'] = df['team_strength_diff'] * df.get('team_possession_style', 1.0)

        return df

    def _create_player_role_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features indicating player role in team.

        Args:
            df: DataFrame with player data

        Returns:
            DataFrame with player role features
        """
        # Key player indicator (plays most minutes)
        if 'player_avg_minutes' in df.columns:
            df['is_key_player'] = (
                df['player_avg_minutes'] > df['player_avg_minutes'].quantile(0.8)
            ).astype(int)

        # Playmaker indicator (high pass volume + accuracy)
        if 'player_career_avg_passes' in df.columns and 'player_career_pass_accuracy' in df.columns:
            playmaker_score = (
                df['player_career_avg_passes'] / df['player_career_avg_passes'].mean() *
                df['player_career_pass_accuracy'] / 100
            )
            df['is_playmaker'] = (playmaker_score > playmaker_score.quantile(0.75)).astype(int)

        # Creator indicator (high xA)
        if 'player_avg_xA' in df.columns:
            df['is_creator'] = (
                df['player_avg_xA'] > df['player_avg_xA'].quantile(0.75)
            ).astype(int)

        return df

    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio-based features.

        Args:
            df: DataFrame

        Returns:
            DataFrame with ratio features
        """
        # Pass completion to attempt ratio trend
        if 'pass_accuracy_rolling_5' in df.columns and 'player_career_pass_accuracy' in df.columns:
            df['accuracy_trend'] = (
                df['pass_accuracy_rolling_5'] /
                (df['player_career_pass_accuracy'] + 0.01)
            )

        # Minutes utilization
        if 'minutes_played' in df.columns and 'player_avg_minutes' in df.columns:
            df['minutes_utilization'] = df['minutes_played'] / (df['player_avg_minutes'] + 1)

        # Progressive to total passes ratio
        if 'progressive_pass_rate' in df.columns:
            df['progressive_ratio_vs_avg'] = (
                df['progressive_pass_rate'] /
                (df['progressive_pass_rate'].mean() + 0.01)
            )

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables.

        Args:
            df: DataFrame

        Returns:
            DataFrame with interaction features
        """
        # Key interactions
        interactions = [
            ('is_home', 'team_strength_diff'),
            ('position_encoded', 'team_possession_style'),
            ('is_key_player', 'team_strength'),
            ('is_playmaker', 'opponent_strength'),
            ('form_trend', 'match_importance'),
            ('minutes_played', 'expected_involvement')
        ]

        for feat1, feat2 in interactions:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]

        # Three-way interactions for critical features
        if all(col in df.columns for col in ['position_encoded', 'minutes_played', 'team_strength_diff']):
            df['position_minutes_strength'] = (
                df['position_encoded'] *
                df['minutes_played'] *
                df['team_strength_diff']
            )

        return df

    def _create_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite scores combining multiple features.

        Args:
            df: DataFrame

        Returns:
            DataFrame with composite scores
        """
        # Overall player quality score
        quality_features = [
            'player_career_avg_passes',
            'player_career_pass_accuracy',
            'player_avg_xG',
            'player_avg_xA',
            'player_pass_consistency'
        ]

        available_quality = [f for f in quality_features if f in df.columns]
        if available_quality:
            # Normalize and combine
            for feat in available_quality:
                df[f'{feat}_norm'] = df[feat] / (df[feat].mean() + 0.01)

            df['player_quality_score'] = df[[f'{feat}_norm' for feat in available_quality]].mean(axis=1)

            # Drop temporary normalized columns
            df = df.drop(columns=[f'{feat}_norm' for feat in available_quality])

        # Match difficulty score
        difficulty_features = [
            'opponent_strength',
            'match_importance',
            'team_strength_diff'
        ]

        available_difficulty = [f for f in difficulty_features if f in df.columns]
        if available_difficulty:
            df['match_difficulty'] = df[available_difficulty].mean(axis=1)

        # Expected pass volume (composite predictor)
        df['expected_pass_volume'] = (
            df.get('player_career_passes_per90', 0) *
            df.get('minutes_ratio', df['minutes_played'] / 90) *
            df.get('form_trend', 1.0) *
            df.get('team_strength', 1.0) /
            df.get('opponent_strength', 1.0)
        )

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of modeling features.

        Args:
            df: DataFrame with all features

        Returns:
            List of feature names for modeling
        """
        # Exclude non-modeling columns
        exclude_cols = [
            'passes_attempted',  # Target
            'player_name', 'date', 'match_id',  # Identifiers
            'home_team', 'away_team',  # Raw team names
            'position_group',  # Categorical (use encoded version)
            'score'  # Match result
        ]

        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and
            not col.endswith('_norm')  # Temporary columns
        ]

        return feature_cols
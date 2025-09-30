"""Production prediction service for upcoming matches using FBRef-trained models."""

import argparse
import logging
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from src.data.collectors.fbref import FBRefCollector
from src.data.processors.fbref_processor import FBRefProcessor
from src.features.fbref_features import FBRefFeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionService:
    """Service for predicting player passes in upcoming matches."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        processor_path: Optional[Path] = None
    ):
        """Initialize the prediction service.

        Args:
            model_path: Path to saved model
            processor_path: Path to saved processor
        """
        self.model_dir = Path("data/models/fbref")
        self.data_dir = Path("data/processed/fbref")

        # Load latest model if not specified
        if model_path:
            self.model = self._load_model(model_path)
        else:
            self.model = self._load_latest_model()

        # Initialize processors
        self.processor = FBRefProcessor()
        self.feature_engineer = FBRefFeatureEngineer()

        # Load historical data for context
        self.historical_data = self._load_historical_data()

    def _load_model(self, model_path: Path):
        """Load a specific model.

        Args:
            model_path: Path to model file

        Returns:
            Loaded model
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {model_path}")
        return model

    def _load_latest_model(self):
        """Load the most recent model.

        Returns:
            Latest trained model
        """
        # Look for ensemble first, then XGBoost
        for model_name in ['ensemble_fbref_latest.pkl', 'xgboost_fbref_latest.pkl']:
            model_path = self.model_dir / model_name
            if model_path.exists():
                return self._load_model(model_path)

        # Fallback to any model
        model_files = list(self.model_dir.glob("*.pkl"))
        if model_files:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            return self._load_model(latest_model)

        raise FileNotFoundError("No models found. Train a model first using train_fbref.py")

    def _load_historical_data(self) -> pd.DataFrame:
        """Load recent historical data for player context.

        Returns:
            DataFrame with recent player data
        """
        # Find most recent processed data
        processed_files = list(self.data_dir.glob("fbref_processed_*.parquet"))

        if processed_files:
            latest_file = max(processed_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading historical data from {latest_file}")
            return pd.read_parquet(latest_file)

        logger.warning("No historical data found. Predictions may be less accurate.")
        return pd.DataFrame()

    def predict_upcoming_matches(
        self,
        days_ahead: int = 7,
        leagues: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Predict passes for upcoming matches.

        Args:
            days_ahead: Number of days to look ahead
            leagues: Specific leagues to predict

        Returns:
            DataFrame with predictions
        """
        logger.info(f"Getting fixtures for next {days_ahead} days...")

        # Get upcoming fixtures
        collector = FBRefCollector()
        fixtures = collector.get_upcoming_fixtures(days_ahead)

        if fixtures.empty:
            logger.warning("No upcoming fixtures found")
            return pd.DataFrame()

        logger.info(f"Found {len(fixtures)} upcoming fixtures")

        # Filter leagues if specified
        if leagues:
            fixtures = fixtures[fixtures['league'].isin(leagues)]

        # Generate predictions for each fixture
        predictions = []
        for _, fixture in fixtures.iterrows():
            fixture_predictions = self._predict_fixture(fixture)
            predictions.extend(fixture_predictions)

        results_df = pd.DataFrame(predictions)

        # Sort by date and predicted passes
        if not results_df.empty:
            results_df = results_df.sort_values(['date', 'predicted_passes'], ascending=[True, False])

        return results_df

    def _predict_fixture(self, fixture: pd.Series) -> List[Dict]:
        """Generate predictions for a single fixture.

        Args:
            fixture: Series with fixture information

        Returns:
            List of player predictions
        """
        predictions = []

        # Get likely lineups for both teams
        home_players = self._get_team_players(fixture['home_team'])
        away_players = self._get_team_players(fixture['away_team'])

        # Predict for home team
        for player_name in home_players:
            pred = self._predict_player(
                player_name=player_name,
                team=fixture['home_team'],
                opponent=fixture['away_team'],
                is_home=True,
                fixture_date=fixture['date']
            )
            if pred:
                pred.update({
                    'fixture': f"{fixture['home_team']} vs {fixture['away_team']}",
                    'date': fixture['date'],
                    'league': fixture.get('league', 'Unknown')
                })
                predictions.append(pred)

        # Predict for away team
        for player_name in away_players:
            pred = self._predict_player(
                player_name=player_name,
                team=fixture['away_team'],
                opponent=fixture['home_team'],
                is_home=False,
                fixture_date=fixture['date']
            )
            if pred:
                pred.update({
                    'fixture': f"{fixture['home_team']} vs {fixture['away_team']}",
                    'date': fixture['date'],
                    'league': fixture.get('league', 'Unknown')
                })
                predictions.append(pred)

        return predictions

    def _get_team_players(self, team_name: str) -> List[str]:
        """Get likely players for a team based on recent matches.

        Args:
            team_name: Team name

        Returns:
            List of player names
        """
        if self.historical_data.empty:
            return []

        # Get players who played recently for this team
        team_data = self.historical_data[
            (self.historical_data['home_team'] == team_name) |
            (self.historical_data['away_team'] == team_name)
        ]

        if team_data.empty:
            return []

        # Get most frequent players (likely starters)
        recent_players = team_data.groupby('player_name')['minutes_played'].agg(['mean', 'count'])
        recent_players = recent_players[recent_players['count'] >= 3]  # At least 3 games
        recent_players = recent_players.sort_values('mean', ascending=False)

        # Return top 11 players (likely starters)
        return recent_players.head(11).index.tolist()

    def _predict_player(
        self,
        player_name: str,
        team: str,
        opponent: str,
        is_home: bool,
        fixture_date: str
    ) -> Optional[Dict]:
        """Predict passes for a single player.

        Args:
            player_name: Player name
            team: Player's team
            opponent: Opponent team
            is_home: Whether playing at home
            fixture_date: Date of fixture

        Returns:
            Dictionary with prediction or None if cannot predict
        """
        # Get player historical data
        player_data = self.historical_data[self.historical_data['player_name'] == player_name]

        if player_data.empty:
            logger.debug(f"No historical data for {player_name}")
            return None

        # Create feature vector for prediction
        features = self._create_prediction_features(
            player_name, player_data, team, opponent, is_home
        )

        if features is None:
            return None

        # Make prediction
        try:
            # Assume 70 minutes played (can be refined with lineup predictions)
            expected_minutes = 70
            exposure = expected_minutes / 90

            # Get feature names from the model (if XGBoost)
            if hasattr(self.model, 'feature_names_in_'):
                feature_cols = self.model.feature_names_in_
                features_df = pd.DataFrame([features])[feature_cols]
            else:
                features_df = pd.DataFrame([features])

            prediction = self.model.predict(features_df, exposure=np.array([exposure]))[0]

            # Create confidence interval (simplified)
            player_std = player_data['passes_attempted'].std()
            lower_bound = max(0, prediction - 1.96 * player_std)
            upper_bound = prediction + 1.96 * player_std

            return {
                'player_name': player_name,
                'team': team,
                'predicted_passes': round(prediction, 1),
                'confidence_interval': f"[{round(lower_bound, 1)}, {round(upper_bound, 1)}]",
                'expected_minutes': expected_minutes,
                'position': player_data['position_group'].mode()[0] if 'position_group' in player_data.columns else 'Unknown',
                'recent_avg': round(player_data['passes_attempted'].tail(5).mean(), 1)
            }

        except Exception as e:
            logger.error(f"Error predicting for {player_name}: {e}")
            return None

    def _create_prediction_features(
        self,
        player_name: str,
        player_data: pd.DataFrame,
        team: str,
        opponent: str,
        is_home: bool
    ) -> Optional[Dict]:
        """Create feature vector for prediction.

        Args:
            player_name: Player name
            player_data: Historical data for player
            team: Player's team
            opponent: Opponent team
            is_home: Whether playing at home

        Returns:
            Feature dictionary or None
        """
        try:
            # Get latest player stats
            latest_stats = player_data.iloc[-1]

            # Calculate rolling averages
            recent_passes = player_data['passes_attempted'].tail(5).mean()
            recent_accuracy = player_data['pass_accuracy'].tail(5).mean() if 'pass_accuracy' in player_data.columns else 85

            # Get team strengths (simplified - in production would have updated team ratings)
            team_strength = self.historical_data[self.historical_data['home_team'] == team]['team_strength'].mean() if 'team_strength' in self.historical_data.columns else 1.0
            opp_strength = self.historical_data[self.historical_data['home_team'] == opponent]['opponent_strength'].mean() if 'opponent_strength' in self.historical_data.columns else 1.0

            features = {
                'position_encoded': latest_stats.get('position_encoded', 1),
                'minutes_played': 70,  # Expected minutes
                'is_home': int(is_home),
                'team_strength_diff': team_strength - opp_strength,
                'opponent_strength': opp_strength,
                'player_career_avg_passes': player_data['passes_attempted'].mean(),
                'player_career_passes_per90': player_data['passes_attempted'].mean() / player_data['minutes_played'].mean() * 90,
                'player_pass_consistency': 1 / (player_data['passes_attempted'].std() / player_data['passes_attempted'].mean() + 0.1),
                'player_games_played': len(player_data),
                'passes_attempted_rolling_5': recent_passes,
                'form_trend': recent_passes / player_data['passes_attempted'].mean()
            }

            # Add FBRef specific features if available
            fbref_features = [
                'player_avg_xG', 'player_avg_xA', 'player_avg_touches',
                'player_avg_progressive_passes', 'expected_involvement',
                'progressive_rate', 'team_possession_style', 'is_possession_team',
                'is_key_player', 'is_playmaker', 'player_quality_score',
                'expected_pass_volume'
            ]

            for feat in fbref_features:
                if feat in latest_stats.index:
                    features[feat] = latest_stats[feat]
                else:
                    # Use sensible defaults
                    features[feat] = 0 if feat.startswith('is_') else 1.0

            return features

        except Exception as e:
            logger.error(f"Error creating features for {player_name}: {e}")
            return None

    def predict_custom_match(
        self,
        home_team: str,
        away_team: str,
        player_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Predict for a custom match.

        Args:
            home_team: Home team name
            away_team: Away team name
            player_names: Specific players to predict (optional)

        Returns:
            DataFrame with predictions
        """
        fixture = pd.Series({
            'home_team': home_team,
            'away_team': away_team,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'league': 'Custom'
        })

        predictions = self._predict_fixture(fixture)

        # Filter to specific players if requested
        if player_names:
            predictions = [p for p in predictions if p['player_name'] in player_names]

        return pd.DataFrame(predictions)


def main():
    """Main entry point for prediction service."""
    parser = argparse.ArgumentParser(description="Predict player passes for upcoming matches")
    parser.add_argument('--days-ahead', type=int, default=7, help='Days ahead to predict')
    parser.add_argument('--leagues', nargs='+', help='Specific leagues to predict')
    parser.add_argument('--match', nargs=2, metavar=('HOME', 'AWAY'), help='Predict specific match')
    parser.add_argument('--players', nargs='+', help='Specific players to predict')
    parser.add_argument('--output', help='Output CSV file')
    parser.add_argument('--update-data', action='store_true', help='Update historical data first')

    args = parser.parse_args()

    # Initialize service
    service = PredictionService()

    # Update data if requested
    if args.update_data:
        logger.info("Updating historical data...")
        from train_fbref import collect_fbref_data, process_fbref_data
        raw_data = collect_fbref_data(seasons=["2024-2025"])
        if not raw_data.empty:
            process_fbref_data(raw_data)

    # Generate predictions
    if args.match:
        logger.info(f"Predicting for {args.match[0]} vs {args.match[1]}")
        predictions = service.predict_custom_match(
            home_team=args.match[0],
            away_team=args.match[1],
            player_names=args.players
        )
    else:
        predictions = service.predict_upcoming_matches(
            days_ahead=args.days_ahead,
            leagues=args.leagues
        )

    # Display results
    if not predictions.empty:
        print("\n" + "="*80)
        print("PASS PREDICTIONS FOR UPCOMING MATCHES")
        print("="*80)

        # Group by fixture
        for fixture in predictions['fixture'].unique():
            fixture_data = predictions[predictions['fixture'] == fixture]
            print(f"\n{fixture} - {fixture_data.iloc[0]['date']}")
            print("-" * 60)

            # Sort by team and predicted passes
            fixture_data = fixture_data.sort_values(['team', 'predicted_passes'], ascending=[True, False])

            for _, player in fixture_data.iterrows():
                print(f"  {player['player_name']:25} {player['position']:3} "
                      f"| Predicted: {player['predicted_passes']:5.1f} "
                      f"| Recent avg: {player['recent_avg']:5.1f} "
                      f"| CI: {player['confidence_interval']}")

        # Save if requested
        if args.output:
            predictions.to_csv(args.output, index=False)
            logger.info(f"Predictions saved to {args.output}")

    else:
        logger.warning("No predictions generated")


if __name__ == "__main__":
    main()
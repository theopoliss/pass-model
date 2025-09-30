"""Training script for FBRef data pipeline - separate from StatsBomb legacy pipeline."""

import argparse
import logging
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

from src.data.collectors.fbref import FBRefCollector
from src.data.processors.fbref_processor import FBRefProcessor
from src.features.fbref_features import FBRefFeatureEngineer
from src.models.baseline import PoissonRegression, NegativeBinomialRegression
from src.models.advanced_models import XGBoostRegressor, WeightedEnsemble
from src.evaluation.metrics import PassPredictionEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FBRefConfig:
    """Configuration for FBRef pipeline."""

    # Data collection - Extended for 5 years of data
    SEASONS = [
        "2019-2020",
        "2020-2021",
        "2021-2022",
        "2022-2023",
        "2023-2024",
        "2024-2025"  # Current season (partial)
    ]
    LEAGUES = ["Premier-League", "La-Liga", "Serie-A", "Bundesliga", "Ligue-1"]

    # Model features - optimized for FBRef data
    FEATURES = [
        # Core features
        'position_encoded',
        'minutes_played',
        'is_home',
        'team_strength_diff',
        'opponent_strength',

        # Player features
        'player_career_avg_passes',
        'player_career_passes_per90',
        'player_pass_consistency',
        'player_games_played',

        # FBRef advanced metrics
        'player_avg_xG',
        'player_avg_xA',
        'player_avg_touches',
        'player_avg_progressive_passes',
        'expected_involvement',
        'progressive_rate',

        # Rolling features
        'passes_attempted_rolling_5',
        'form_trend',

        # Team style
        'team_possession_style',
        'is_possession_team',

        # Player role
        'is_key_player',
        'is_playmaker',

        # Composite scores
        'player_quality_score',
        'expected_pass_volume'
    ]

    # Training parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MIN_GAMES_FOR_PREDICTION = 3  # Minimum games to make predictions

    # Model paths
    MODEL_DIR = Path("data/models/fbref")
    DATA_DIR = Path("data/processed/fbref")


def collect_fbref_data(
    seasons: Optional[List[str]] = None,
    leagues: Optional[List[str]] = None,
    force_refresh: bool = False
) -> pd.DataFrame:
    """Collect data from FBRef.

    Args:
        seasons: List of seasons to collect
        leagues: List of leagues to collect
        force_refresh: Whether to force refresh cached data

    Returns:
        Combined DataFrame with all data
    """
    seasons = seasons or FBRefConfig.SEASONS
    leagues = leagues or FBRefConfig.LEAGUES

    logger.info(f"Collecting FBRef data for seasons: {seasons}, leagues: {leagues}")

    collector = FBRefCollector()
    all_data = []

    for season in seasons:
        season_data = collector.collect_season_data(season, leagues)
        if not season_data.empty:
            all_data.append(season_data)
            logger.info(f"Collected {len(season_data)} records for {season}")

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total records collected: {len(combined_data)}")

        # Save raw data
        FBRefConfig.DATA_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = FBRefConfig.DATA_DIR / f"fbref_raw_{timestamp}.parquet"
        combined_data.to_parquet(raw_file, index=False)
        logger.info(f"Raw data saved to {raw_file}")

        return combined_data

    logger.warning("No data collected")
    return pd.DataFrame()


def process_fbref_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, FBRefProcessor, FBRefFeatureEngineer]:
    """Process raw FBRef data for modeling.

    Args:
        raw_df: Raw data from FBRef

    Returns:
        Tuple of (processed_df, processor, feature_engineer)
    """
    logger.info("Processing FBRef data...")

    # Initialize processors
    processor = FBRefProcessor()
    feature_engineer = FBRefFeatureEngineer()

    # Process data
    processed_df = processor.process_data(raw_df, train=True)
    logger.info(f"Processed {len(processed_df)} records")

    # Engineer features
    featured_df = feature_engineer.engineer_features(processed_df)
    logger.info(f"Created {len(featured_df.columns)} features")

    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_file = FBRefConfig.DATA_DIR / f"fbref_processed_{timestamp}.parquet"
    featured_df.to_parquet(processed_file, index=False)
    logger.info(f"Processed data saved to {processed_file}")

    return featured_df, processor, feature_engineer


def train_models(
    df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> Dict[str, object]:
    """Train multiple models on FBRef data.

    Args:
        df: Processed DataFrame
        features: List of features to use

    Returns:
        Dictionary of trained models
    """
    features = features or FBRefConfig.FEATURES

    # Filter to available features
    available_features = [f for f in features if f in df.columns]
    logger.info(f"Using {len(available_features)} features for training")

    # Remove players with too few games
    player_games = df.groupby('player_name')['match_id'].nunique()
    valid_players = player_games[player_games >= FBRefConfig.MIN_GAMES_FOR_PREDICTION].index
    df = df[df['player_name'].isin(valid_players)]
    logger.info(f"Training on {len(valid_players)} players with >= {FBRefConfig.MIN_GAMES_FOR_PREDICTION} games")

    # Prepare data
    X = df[available_features]
    y = df['passes_attempted']

    # Time-based split (more recent data in test)
    df_sorted = df.sort_values('date')
    split_idx = int(len(df_sorted) * (1 - FBRefConfig.TEST_SIZE))

    train_indices = df_sorted.index[:split_idx]
    test_indices = df_sorted.index[split_idx:]

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    # Minutes as exposure
    exposure_train = df.loc[train_indices, 'minutes_played'] / 90
    exposure_test = df.loc[test_indices, 'minutes_played'] / 90

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    models = {}

    # 1. Poisson Regression (baseline)
    logger.info("Training Poisson Regression...")
    poisson = PoissonRegression()
    poisson.fit(X_train, y_train, exposure=exposure_train)
    models['poisson_fbref'] = poisson

    # 2. Negative Binomial (handles overdispersion)
    logger.info("Training Negative Binomial...")
    negbin = NegativeBinomialRegression()
    negbin.fit(X_train, y_train, exposure=exposure_train)
    models['negbin_fbref'] = negbin

    # 3. XGBoost (best performer)
    logger.info("Training XGBoost...")
    xgb_model = XGBoostRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    xgb_model.fit(X_train, y_train, exposure=exposure_train)
    models['xgboost_fbref'] = xgb_model

    # 4. Ensemble
    logger.info("Creating ensemble...")
    ensemble = WeightedEnsemble(
        models={
            'poisson': poisson,
            'negbin': negbin,
            'xgboost': xgb_model
        }
    )
    # Fit the ensemble to calculate weights
    ensemble.fit(X_train, y_train, exposure=exposure_train)
    models['ensemble_fbref'] = ensemble

    # Evaluate models
    evaluator = PassPredictionEvaluator()
    results = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test, exposure=exposure_test)
        metrics = evaluator.evaluate(y_test, y_pred)
        results[model_name] = metrics
        logger.info(f"{model_name} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")

    # Save models
    save_models(models, results)

    return models


def save_models(models: Dict, results: Dict):
    """Save trained models with versioning.

    Args:
        models: Dictionary of trained models
        results: Dictionary of evaluation results
    """
    FBRefConfig.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")

    for model_name, model in models.items():
        # Save model with timestamp
        model_file = FBRefConfig.MODEL_DIR / f"{model_name}_{timestamp}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved {model_name} to {model_file}")

        # Also save as 'latest' for easy access
        latest_file = FBRefConfig.MODEL_DIR / f"{model_name}_latest.pkl"
        with open(latest_file, 'wb') as f:
            pickle.dump(model, f)

    # Save results
    results_df = pd.DataFrame(results).T
    results_file = FBRefConfig.MODEL_DIR / f"model_results_{timestamp}.csv"
    results_df.to_csv(results_file)
    logger.info(f"Results saved to {results_file}")

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'models': list(models.keys()),
        'features_used': FBRefConfig.FEATURES,
        'leagues': FBRefConfig.LEAGUES,
        'seasons': FBRefConfig.SEASONS
    }

    metadata_file = FBRefConfig.MODEL_DIR / f"metadata_{timestamp}.json"
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Main training pipeline for FBRef data."""
    parser = argparse.ArgumentParser(description="Train pass prediction models on FBRef data")
    parser.add_argument('--collect-data', action='store_true', help='Collect fresh data from FBRef')
    parser.add_argument('--use-cached', action='store_true', help='Use most recent cached data')
    parser.add_argument('--seasons', nargs='+', help='Seasons to collect (e.g., 2023-2024 2024-2025)')
    parser.add_argument('--leagues', nargs='+', help='Leagues to collect')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with limited data')

    args = parser.parse_args()

    # Collect or load data
    if args.collect_data:
        raw_data = collect_fbref_data(
            seasons=args.seasons,
            leagues=args.leagues
        )
    elif args.use_cached:
        # Find most recent cached file - check multiple locations
        cache_locations = [
            FBRefConfig.DATA_DIR,
            Path("data/raw/fbref"),
            Path("data/raw/fbref_cache")
        ]

        cache_files = []
        for location in cache_locations:
            if location.exists():
                cache_files.extend(list(location.glob("*.parquet")))

        if cache_files:
            # Prioritize larger files (full datasets) over single league files
            cache_files_with_size = [(f, f.stat().st_size) for f in cache_files]
            cache_files_with_size.sort(key=lambda x: x[1], reverse=True)
            latest_cache = cache_files_with_size[0][0]
            logger.info(f"Loading cached data from {latest_cache}")
            raw_data = pd.read_parquet(latest_cache)
        else:
            logger.error("No cached data found. Run with --collect-data first.")
            return
    else:
        # Create sample data for testing
        logger.info("Creating sample data for testing...")
        raw_data = create_sample_data()

    if raw_data.empty:
        logger.error("No data available for training")
        return

    # Process data
    processed_data, processor, feature_engineer = process_fbref_data(raw_data)

    # Train models
    models = train_models(processed_data)

    logger.info("Training complete!")
    logger.info(f"Models saved to {FBRefConfig.MODEL_DIR}")


def create_sample_data() -> pd.DataFrame:
    """Create sample data for testing the pipeline."""
    np.random.seed(42)

    n_players = 50
    n_matches = 20

    data = []
    for player_id in range(n_players):
        player_name = f"Player_{player_id}"
        position = np.random.choice(['DF', 'MF', 'FW'])

        for match in range(n_matches):
            data.append({
                'player_name': player_name,
                'match_id': f"match_{match}",
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=match * 3),
                'position': position,
                'minutes_played': np.random.uniform(30, 90),
                'passes_attempted': np.random.poisson(30 if position == 'MF' else 20),
                'passes_completed': np.random.uniform(0.7, 0.95) * 30,
                'pass_accuracy': np.random.uniform(70, 95),
                'xG': np.random.uniform(0, 0.5),
                'xA': np.random.uniform(0, 0.3),
                'touches': np.random.uniform(20, 80),
                'progressive_passes': np.random.poisson(3),
                'home_team': f"Team_{np.random.randint(0, 10)}",
                'away_team': f"Team_{np.random.randint(10, 20)}",
                'league': 'Premier-League',
                'season': '2024-2025'
            })

    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
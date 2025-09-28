"""Main training script for pass prediction models."""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config.config import config
from src.data.collectors.statsbomb import StatsBombCollector
from src.data.processors.processor import PassDataProcessor
from src.features.feature_engineering import FeatureEngineer
from src.models.baseline import (
    HistoricalAverageBaseline,
    PoissonRegression,
    NegativeBinomialRegression,
    EnsembleBaseline
)
from src.models.advanced_models import (
    PositionSpecificModel,
    XGBoostRegressor,
    WeightedEnsemble,
    StackingEnsemble
)
from src.evaluation.metrics import PassPredictionEvaluator

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_data(competitions: list = None) -> pd.DataFrame:
    """Collect data from StatsBomb.

    Args:
        competitions: List of competition/season pairs to collect

    Returns:
        DataFrame with player-match pass data
    """
    if competitions is None:
        competitions = config.data.competitions

    collector = StatsBombCollector(cache_dir=config.raw_data_dir)
    all_data = []
    stats = []

    logger.info(f"Starting data collection for {len(competitions)} competitions")
    logger.info("=" * 60)

    for i, comp in enumerate(competitions, 1):
        comp_name = f"Competition {comp['competition_id']}, Season {comp['season_id']}"
        logger.info(f"[{i}/{len(competitions)}] Collecting: {comp_name}")

        try:
            matches, events, lineups = collector.collect_competition_data(
                comp["competition_id"],
                comp["season_id"]
            )

            # Aggregate player passes
            player_data = collector.aggregate_player_passes(events, matches, lineups)
            player_data["competition_id"] = comp["competition_id"]
            player_data["season_id"] = comp["season_id"]

            all_data.append(player_data)

            # Track statistics
            stats.append({
                "competition": comp_name,
                "matches": len(matches),
                "events": len(events),
                "player_records": len(player_data)
            })

            logger.info(f"  ✓ Collected: {len(matches)} matches, {len(player_data)} player records")

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            stats.append({
                "competition": comp_name,
                "matches": 0,
                "events": 0,
                "player_records": 0
            })
            continue

    logger.info("=" * 60)
    logger.info("Collection Summary:")
    for stat in stats:
        if stat["player_records"] > 0:
            logger.info(f"  • {stat['competition']}: {stat['matches']} matches, {stat['player_records']} records")

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info("=" * 60)
        logger.info(f"Total collected: {len(combined_data)} player-match records")

        # Print position distribution
        position_counts = combined_data['position'].value_counts().head(5)
        logger.info("\nTop 5 positions:")
        for pos, count in position_counts.items():
            logger.info(f"  • {pos}: {count} records")

        return combined_data
    else:
        raise ValueError("No data collected")


def prepare_features(raw_data: pd.DataFrame) -> tuple:
    """Prepare features for modeling.

    Args:
        raw_data: Raw player-match data

    Returns:
        Tuple of (processed_data, feature_matrix, feature_names, target)
    """
    # Process data
    processor = PassDataProcessor(
        min_minutes=config.data.min_minutes_played,
        exclude_goalkeepers=config.data.exclude_goalkeepers
    )

    processed_data, feature_matrix, feature_names = processor.process_data(raw_data)

    # Additional feature engineering
    engineer = FeatureEngineer()
    processed_data = engineer.engineer_features(processed_data, feature_set="intermediate")

    # Update feature matrix with new features
    feature_matrix, feature_names = processor.create_model_features(processed_data)

    # Extract target
    target = processed_data[config.model.target]

    # Extract exposure (minutes played)
    exposure = processed_data["minutes_played"] if "minutes_played" in processed_data.columns else None

    return processed_data, feature_matrix, feature_names, target, exposure


def tune_xgboost_hyperparameters(X_train, y_train, exposure_train=None, n_iter=20):
    """Tune XGBoost hyperparameters with proper exposure handling.

    Args:
        X_train: Training features
        y_train: Training targets
        exposure_train: Exposure values for training
        n_iter: Number of parameter settings sampled

    Returns:
        Best parameters found
    """
    logger.info("Starting XGBoost hyperparameter tuning...")

    # Define parameter distributions
    param_distributions = {
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }

    # Manual randomized search with proper exposure handling
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error
    import random

    best_score = float('inf')
    best_params = None

    # Random sampling of parameter combinations
    random.seed(42)
    for i in range(n_iter):
        # Sample parameters
        params = {
            key: random.choice(values) for key, values in param_distributions.items()
        }

        # Cross-validation with exposure
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            if exposure_train is not None:
                exp_cv_train = exposure_train.iloc[train_idx]
                exp_cv_val = exposure_train.iloc[val_idx]
            else:
                exp_cv_train = exp_cv_val = None

            # Train model with current params
            model = XGBoostRegressor(
                use_exposure=config.model.use_exposure,
                base_params={'objective': 'count:poisson', 'seed': 42, **params}
            )
            model.fit(X_cv_train, y_cv_train, exposure=exp_cv_train)

            # Predict and score
            y_pred = model.predict(X_cv_val, exposure=exp_cv_val)
            cv_scores.append(mean_absolute_error(y_cv_val, y_pred))

        # Average CV score
        avg_score = np.mean(cv_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            logger.info(f"  New best params (iter {i+1}): MAE={avg_score:.3f}")

    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best CV score (MAE): {best_score:.3f}")

    return best_params

def train_models(X_train, y_train, X_test, y_test, exposure_train=None, exposure_test=None, tune_xgb=False, ensemble_type='weighted') -> dict:
    """Train multiple models and evaluate them.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        exposure_train, exposure_test: Exposure variables

    Returns:
        Dictionary with trained models and results
    """
    models = {}
    results = {}
    evaluator = PassPredictionEvaluator()

    # 1. Historical Average Baseline
    logger.info("Training Historical Average Baseline")
    hist_model = HistoricalAverageBaseline(group_columns=["position_encoded"])
    hist_model.fit(X_train, y_train)
    y_pred_hist = hist_model.predict(X_test)
    results["historical_average"] = evaluator.evaluate(y_test, y_pred_hist)
    models["historical_average"] = hist_model

    # 2. Poisson Regression
    logger.info("Training Poisson Regression")
    poisson_model = PoissonRegression(use_exposure=config.model.use_exposure)
    poisson_model.fit(X_train, y_train, exposure=exposure_train)
    y_pred_poisson = poisson_model.predict(X_test, exposure=exposure_test)
    results["poisson"] = evaluator.evaluate(y_test, y_pred_poisson)
    models["poisson"] = poisson_model

    # 3. Negative Binomial Regression (if overdispersion detected)
    if config.model.overdispersion:
        logger.info("Training Negative Binomial Regression")
        nb_model = NegativeBinomialRegression(use_exposure=config.model.use_exposure)
        nb_model.fit(X_train, y_train, exposure=exposure_train)
        y_pred_nb = nb_model.predict(X_test, exposure=exposure_test)
        results["negative_binomial"] = evaluator.evaluate(y_test, y_pred_nb)
        models["negative_binomial"] = nb_model

    # 4. Position-Specific Poisson
    logger.info("Training Position-Specific Poisson Model")
    pos_model = PositionSpecificModel(
        base_model_class=PoissonRegression,
        position_column="position_encoded",
        model_params={"use_exposure": config.model.use_exposure}
    )
    pos_model.fit(X_train, y_train, exposure=exposure_train)
    y_pred_pos = pos_model.predict(X_test, exposure=exposure_test)
    results["position_specific"] = evaluator.evaluate(y_test, y_pred_pos)
    models["position_specific"] = pos_model

    # 5. XGBoost
    logger.info("Training XGBoost Model")

    # Tune hyperparameters if requested
    if tune_xgb:
        best_params = tune_xgboost_hyperparameters(X_train, y_train, exposure_train)
        # Merge with base params
        xgb_params = {
            'objective': 'count:poisson',
            'seed': 42,
            **best_params
        }
        xgb_model = XGBoostRegressor(use_exposure=config.model.use_exposure, base_params=xgb_params)
    else:
        xgb_model = XGBoostRegressor(use_exposure=config.model.use_exposure)

    xgb_model.fit(X_train, y_train, exposure=exposure_train)
    y_pred_xgb = xgb_model.predict(X_test, exposure=exposure_test)
    results["xgboost"] = evaluator.evaluate(y_test, y_pred_xgb)
    models["xgboost"] = xgb_model

    # 6. Ensemble (using already-trained models)
    logger.info(f"Training {ensemble_type.capitalize()} Ensemble Model")

    # Use the already-trained models for ensemble
    ensemble_base_models = {
        "poisson": models["poisson"],
        "position_specific": models["position_specific"],
        "xgboost": models["xgboost"]
    }

    if ensemble_type == 'stacking':
        # Stacking ensemble with meta-learner
        ensemble_model = StackingEnsemble(
            base_models=ensemble_base_models,
            use_cv=True
        )
        ensemble_model.fit(X_train, y_train, exposure=exposure_train)
        y_pred_ensemble = ensemble_model.predict(X_test, exposure=exposure_test)
        results["ensemble_stacking"] = evaluator.evaluate(y_test, y_pred_ensemble)
        models["ensemble_stacking"] = ensemble_model

    elif ensemble_type == 'weighted':
        # Weighted ensemble based on validation performance
        ensemble_model = WeightedEnsemble(
            models=ensemble_base_models,
            weight_method='inverse_mae'
        )
        ensemble_model.fit(X_train, y_train, exposure=exposure_train)
        y_pred_ensemble = ensemble_model.predict(X_test, exposure=exposure_test)

        # Log the weights
        if hasattr(ensemble_model, 'weights_'):
            logger.info(f"Ensemble weights: {ensemble_model.weights_}")
            logger.info(f"Validation scores: {ensemble_model.validation_scores_}")

        results["ensemble_weighted"] = evaluator.evaluate(y_test, y_pred_ensemble)
        models["ensemble_weighted"] = ensemble_model

    else:
        # Simple averaging ensemble (fallback)
        ensemble_model = EnsembleBaseline(models=ensemble_base_models)
        ensemble_model.fit(X_train, y_train, exposure=exposure_train)
        y_pred_ensemble = ensemble_model.predict(X_test, exposure=exposure_test)
        results["ensemble_simple"] = evaluator.evaluate(y_test, y_pred_ensemble)
        models["ensemble_simple"] = ensemble_model

    return models, results


def save_models(models: dict, results: dict, output_dir: Path = None):
    """Save trained models and results.

    Args:
        models: Dictionary of trained models
        results: Dictionary of evaluation results
        output_dir: Directory to save models
    """
    if output_dir is None:
        output_dir = config.model_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    for name, model in models.items():
        model_path = output_dir / f"{name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved {name} model to {model_path}")

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / "model_results.csv")
    logger.info(f"Saved results to {output_dir / 'model_results.csv'}")

    # Print results
    print("\n" + "="*60)
    print("Model Performance Summary")
    print("="*60)
    print(results_df[["mae", "rmse", "r2"]].round(3))


def main(args):
    """Main training pipeline."""
    logger.info("Starting pass prediction model training")

    # Step 1: Collect or load data
    data_path = config.processed_data_dir / "player_pass_data.csv"

    if args.collect_data:
        logger.info("Collecting fresh data from StatsBomb API")
        raw_data = collect_data()

        # Save with timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_path = config.processed_data_dir / f"player_pass_data_{timestamp}.csv"
        raw_data.to_csv(versioned_path, index=False)
        logger.info(f"Saved versioned data to {versioned_path}")

        # Also save as main file
        raw_data.to_csv(data_path, index=False)
        logger.info(f"Saved main data to {data_path}")
    elif not data_path.exists():
        logger.info("No existing data found. Collecting from StatsBomb API")
        raw_data = collect_data()
        raw_data.to_csv(data_path, index=False)
    else:
        logger.info(f"Loading data from {data_path}")
        raw_data = pd.read_csv(data_path)

    # Step 2: Prepare features
    logger.info("Preparing features")
    processed_data, X, feature_names, y, exposure = prepare_features(raw_data)

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Features: {feature_names}")

    # Step 3: Split data
    if exposure is not None:
        X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(
            X, y, exposure, test_size=config.model.test_size, random_state=config.model.random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.model.test_size, random_state=config.model.random_state
        )
        exposure_train = exposure_test = None

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Step 4: Train models
    logger.info("Training models")
    models, results = train_models(
        X_train, y_train, X_test, y_test,
        exposure_train, exposure_test,
        tune_xgb=args.tune,
        ensemble_type=args.ensemble
    )

    # Step 5: Save models and results
    save_models(models, results)

    # Step 6: Generate visualizations (optional)
    if args.plot:
        logger.info("Generating plots")
        evaluator = PassPredictionEvaluator()

        best_model_name = min(results.keys(), key=lambda k: results[k]["mae"])
        best_model = models[best_model_name]

        if exposure_test is not None and hasattr(best_model, 'predict'):
            if "exposure" in best_model.predict.__code__.co_varnames:
                y_pred_best = best_model.predict(X_test, exposure=exposure_test)
            else:
                y_pred_best = best_model.predict(X_test)
        else:
            y_pred_best = best_model.predict(X_test)

        fig = evaluator.plot_predictions(y_test.values, y_pred_best, title=f"Best Model: {best_model_name}")
        fig.savefig(config.model_dir / "predictions_plot.png")
        logger.info(f"Saved plot to {config.model_dir / 'predictions_plot.png'}")

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pass prediction models")
    parser.add_argument("--collect-data", action="store_true", help="Collect fresh data from StatsBomb")
    parser.add_argument("--plot", action="store_true", help="Generate evaluation plots")
    parser.add_argument("--tune", action="store_true", help="Tune XGBoost hyperparameters")
    parser.add_argument("--ensemble", type=str, default="weighted", choices=["simple", "weighted", "stacking"],
                       help="Type of ensemble to use (default: weighted)")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--competitions", type=str, help="Comma-separated list of competition_id:season_id pairs (e.g., '43:3,55:43')")
    parser.add_argument("--list-competitions", action="store_true", help="List available competitions and exit")

    args = parser.parse_args()

    # List competitions if requested
    if args.list_competitions:
        from src.data.collectors.statsbomb import StatsBombCollector
        collector = StatsBombCollector()
        competitions = collector.get_competitions()
        print("\nAvailable StatsBomb Competitions:")
        print("=" * 60)
        print(competitions[['competition_id', 'competition_name', 'season_id', 'season_name']].to_string())
        sys.exit(0)

    # Parse custom competitions if provided
    if args.competitions:
        custom_comps = []
        for comp_str in args.competitions.split(','):
            comp_id, season_id = comp_str.split(':')
            custom_comps.append({
                "competition_id": int(comp_id),
                "season_id": int(season_id)
            })
        config.data.competitions = custom_comps

    main(args)
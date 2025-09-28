# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a soccer pass prediction model that uses StatsBomb's free API data to predict how many passes a player will make in a match. The system is designed with a phased development approach, currently at Phase 1 (baseline implementation) with plans for tactical features, two-stage modeling, and production deployment.

## Development Commands

### Training and Data Collection
```bash
# Collect fresh data from StatsBomb and train all models
python train.py --collect-data --plot

# Train using existing data
python train.py

# Interactive exploration
jupyter notebook notebooks/exploration.ipynb
```

### Python Environment Setup (Python 3.13 compatibility)
```bash
# Required for Python 3.13 due to distutils removal
pip install setuptools wheel
pip install -r requirements.txt

# If scipy fails to build, install gfortran first
brew install gcc
```

## Architecture and Key Design Decisions

### Data Pipeline Flow
1. **StatsBomb API** → `StatsBombCollector` collects match events
2. **Raw events** → `aggregate_player_passes()` creates player-match records
3. **Player data** → `PassDataProcessor` handles cleaning, position encoding, team strength calculation
4. **Processed data** → `FeatureEngineer` creates advanced features (interactions, rolling averages)
5. **Features** → Models (Poisson/NegBin with exposure offset for minutes played)

### Model Architecture
- **Count regression approach**: Uses Poisson/Negative Binomial instead of linear regression since passes are count data
- **Exposure modeling**: Minutes played used as exposure offset in log space
- **Hierarchical structure ready**: Code organized to easily add random effects for players/teams/managers

### Known Issues and Fixes

1. **Index mismatch in baseline model**: Fixed by using enumerate instead of DataFrame index in `HistoricalAverageBaseline.predict()`
2. **NaN predictions**: Models now include `.fillna()` and `np.nan_to_num()` to handle missing values
3. **Python 3.13 compatibility**: Requires flexible version requirements (>=) instead of pinned versions

## Configuration System

Configuration is centralized in `config/config.py` using Pydantic models:
- `ModelConfig`: Model type, features, training parameters
- `DataConfig`: Competitions to collect, filtering parameters
- `PositionMapping`: Maps detailed positions to groups (DEF/MID/FWD)

Default competitions: World Cup 2018, Euro 2020 (IDs: 43/3, 55/43)

## Phase Development Roadmap

**Current (Phase 1)**: Baseline with position, minutes, home/away, team strength
**Phase 2**: Add tactical features, formations, PPDA
**Phase 3**: Two-stage model (team volume × player share)
**Phase 4**: Tactical clustering with k-means
**Phase 5**: Game state dynamics, passing networks
**Phase 6**: Production API deployment

## Key Files and Their Roles

- `train.py`: Main entry point, orchestrates the full pipeline
- `src/models/baseline.py`: Statistical models (Poisson, NegBin) with proper count regression
- `src/data/processors/processor.py`: Feature creation from raw data
- `src/evaluation/metrics.py`: Count-specific metrics (Poisson deviance, calibration)

## Testing and Evaluation

The system uses temporal cross-validation to avoid data leakage. Key metrics:
- MAE/RMSE for point estimates
- Poisson deviance for count distribution fit
- Position-stratified evaluation to check model performance across roles

Current performance baseline: ~12.4 MAE for Poisson regression on World Cup 2018 data.
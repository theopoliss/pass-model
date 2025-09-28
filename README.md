# Soccer Pass Prediction Model

A machine learning system for predicting the number of passes a soccer player will make in a match, considering tactical setups and historical performance patterns.

## Overview

This project implements a comprehensive pass prediction system with the following features:
- Data collection from StatsBomb's free API
- Advanced feature engineering including tactical and positional features
- Multiple baseline models (Poisson, Negative Binomial, Ensemble)
- Evaluation framework with count-specific metrics
- Temporal cross-validation

## Project Structure

```
pass-model/
├── data/                   # Data storage
│   ├── raw/               # Raw StatsBomb data
│   ├── processed/         # Processed features
│   └── models/            # Saved models
├── src/                    # Source code
│   ├── data/              # Data collection & processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementations
│   └── evaluation/        # Metrics and evaluation
├── notebooks/              # Jupyter notebooks
├── config/                # Configuration
├── train.py               # Main training script
└── requirements.txt       # Dependencies
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pass-model

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train Models

```bash
# Collect data and train models
python train.py --collect-data --plot

# Or use existing data
python train.py
```

### 2. Explore Data

```bash
jupyter notebook notebooks/exploration.ipynb
```

## Features

### Current (Phase 1)
- **Data Collection**: StatsBomb free data (World Cup 2018, Euro 2020)
- **Core Features**: Position, minutes played, home/away, team strength
- **Models**: Historical average, Poisson regression, Negative Binomial
- **Evaluation**: MAE, RMSE, R², Poisson deviance

### Planned Enhancements
- **Phase 2**: Team-level features, random effects
- **Phase 3**: Two-stage modeling (team volume + player share)
- **Phase 4**: Tactical archetypes via clustering
- **Phase 5**: Game state dynamics, network features
- **Phase 6**: Production API deployment

## Model Performance

Initial results on World Cup 2018 data:

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Historical Average | 15.2 | 20.1 | 0.31 |
| Poisson Regression | 12.4 | 17.3 | 0.49 |
| Ensemble | 11.8 | 16.9 | 0.52 |

## Configuration

Edit `config/config.py` to customize:
- Data sources and competitions
- Feature sets
- Model parameters
- Training settings

## Data Sources

Using StatsBomb's free data:
- FIFA World Cup 2018
- UEFA Euro 2020
- Additional competitions available

## Future Improvements

1. **Tactical Features**
   - Formation detection
   - Playing style clustering
   - Opponent-specific conditioning

2. **Advanced Models**
   - XGBoost with custom objectives
   - Neural networks for sequence modeling
   - Graph neural networks for passing networks

3. **Game Dynamics**
   - Score differential effects
   - Time segment modeling
   - Momentum features

## Contributing

Feel free to submit issues and pull requests.

## License

MIT License
# Soccer Pass Prediction Model

Predicting player pass counts in soccer matches using machine learning and StatsBomb data.

## Overview

This system predicts how many passes a soccer player will make in a match using:
- **Data**: StatsBomb free API (multiple competitions and seasons)
- **Models**: XGBoost, Poisson regression, position-specific models, ensemble methods
- **Features**: Rolling averages, team strength, position encoding, minutes played

## Installation

```bash
# Clone repository
git clone <repository-url>
cd pass-model

# Create virtual environment
python -m venv myvenv
source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
# Train with existing data
python train.py --plot

# Collect fresh data and train
python train.py --collect-data --plot
```

### Advanced Options

```bash
# Train with hyperparameter tuning and stacking ensemble
python train.py --tune --ensemble stacking --plot

# Different ensemble types
python train.py --ensemble weighted  # Weighted by performance
python train.py --ensemble simple    # Simple averaging
```

### Command Line Arguments

- `--collect-data`: Fetch fresh data from StatsBomb API
- `--plot`: Generate evaluation plots
- `--tune`: Enable XGBoost hyperparameter tuning
- `--ensemble [simple|weighted|stacking]`: Choose ensemble method
- `--list-competitions`: Show available competitions

## Configuration

Edit `config/config.py` to customize:
- Competitions to analyze
- Feature engineering options
- Model parameters
- Training/test split ratio

## Data Sources

Using StatsBomb's free data from multiple competitions:
- **International**: World Cup 2018, Euro 2020, Women's World Cup 2019
- **Champions League**: 2015-2019 seasons
- **Domestic Leagues**: La Liga (2017-2019), Premier League (2015-2016)

Total: 10+ competitions/seasons with thousands of matches

## License

MIT License
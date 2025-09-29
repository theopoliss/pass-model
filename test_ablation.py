#!/usr/bin/env python
"""Test ablation study with synthetic formation data."""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from train import run_ablation_study
import argparse

# Load existing data
data = pd.read_csv('data/processed/player_pass_data.csv')

# Add synthetic formation data to demonstrate ablation
np.random.seed(42)
formations = ['433', '442', '352', '4231', '343', '451']

# Add formations based on some pattern
data['team_formation'] = np.random.choice(formations, len(data))
data['opponent_formation'] = np.random.choice(formations, len(data))

# Add realistic effects: 433 vs 442 midfielders get more passes
for idx in data.index:
    if data.loc[idx, 'position'] in ['Center Midfield', 'Left Midfield', 'Right Midfield']:
        if data.loc[idx, 'team_formation'] == '433' and data.loc[idx, 'opponent_formation'] == '442':
            # Add bonus passes for midfield advantage
            data.loc[idx, 'passes_attempted'] *= 1.15
        elif data.loc[idx, 'team_formation'] == '442' and data.loc[idx, 'opponent_formation'] == '433':
            # Reduce passes for being outnumbered
            data.loc[idx, 'passes_attempted'] *= 0.9

# Create args object
args = argparse.Namespace(ensemble='weighted')

# Run ablation study
print("Running ablation study with synthetic formation data...")
print("=" * 70)
run_ablation_study(data, args)
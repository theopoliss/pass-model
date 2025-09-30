"""
Use the soccerdata library to collect FBRef data.
This library handles rate limiting and anti-detection automatically!

Install: pip install soccerdata
Docs: https://github.com/probberechts/soccerdata
"""

import soccerdata as sd
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_fbref_data():
    """Collect player passing data using soccerdata library."""

    # Output directory
    output_dir = Path("data/raw/soccerdata")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Leagues to collect
    leagues = {
        'ENG-Premier League': 'Premier League',
        'ESP-La Liga': 'La Liga',
        'ITA-Serie A': 'Serie A',
        'GER-Bundesliga': 'Bundesliga',
        'FRA-Ligue 1': 'Ligue 1'
    }

    # Seasons to collect
    seasons = ['2223', '2324', '2425']  # 2022-23, 2023-24, 2024-25

    all_data = []

    for league_code, league_name in leagues.items():
        for season in seasons:
            try:
                print(f"\n{'='*60}")
                print(f"Collecting {league_name} {season}")
                print('='*60)

                # Create FBRef scraper
                fbref = sd.FBref(leagues=league_code, seasons=season)

                # Get player passing stats
                print("Fetching passing stats...")
                passing_stats = fbref.read_player_season_stats(stat_type="passing")

                if not passing_stats.empty:
                    # Add metadata
                    passing_stats['league'] = league_name
                    passing_stats['season'] = season

                    # Get additional stats if needed
                    print("Fetching general stats...")
                    general_stats = fbref.read_player_season_stats(stat_type="standard")

                    # Merge stats
                    if not general_stats.empty:
                        combined = pd.merge(
                            passing_stats,
                            general_stats[['player', 'team', 'minutes_90s', 'goals', 'assists', 'xg', 'xa']],
                            on=['player', 'team'],
                            how='left',
                            suffixes=('', '_general')
                        )
                    else:
                        combined = passing_stats

                    all_data.append(combined)
                    print(f"✓ Collected {len(combined)} player records")

                    # Save intermediate results
                    intermediate_file = output_dir / f"{league_code.replace(' ', '_')}_{season}.parquet"
                    combined.to_parquet(intermediate_file)

                else:
                    print(f"✗ No data found for {league_name} {season}")

            except Exception as e:
                print(f"✗ Error collecting {league_name} {season}: {e}")
                continue

    # Combine all data
    if all_data:
        final_data = pd.concat(all_data, ignore_index=True)

        # Save final dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"fbref_complete_{timestamp}.parquet"
        final_data.to_parquet(output_file)

        csv_file = output_dir / f"fbref_complete_{timestamp}.csv"
        final_data.to_csv(csv_file, index=False)

        print(f"\n{'='*60}")
        print("COLLECTION COMPLETE!")
        print('='*60)
        print(f"Total records: {len(final_data):,}")
        print(f"Unique players: {final_data['player'].nunique():,}")
        print(f"\nData saved to:")
        print(f"  {output_file}")
        print(f"  {csv_file}")

        return final_data

    return pd.DataFrame()


def test_single_league():
    """Test with just Premier League current season."""

    print("\n" + "="*60)
    print("TESTING SOCCERDATA LIBRARY")
    print("="*60)

    try:
        # Test with Premier League 2024-25
        print("\nTesting with Premier League 2024-25...")
        # Fix: Use 'leagues' parameter (plural) not 'league'
        fbref = sd.FBref(leagues='ENG-Premier League', seasons='2425')

        # Try to get passing stats
        print("Fetching passing stats...")
        passing = fbref.read_player_season_stats(stat_type="passing")

        if not passing.empty:
            print(f"\n✓ Success! Got {len(passing)} player records")

            print("\nColumns available:")
            print(passing.columns.tolist())

            print("\nSample data:")
            print(passing[['player', 'team', 'passes_completed', 'passes']].head())

            # Check what passing metrics we have
            pass_cols = [col for col in passing.columns if 'pass' in col.lower()]
            print(f"\nPassing columns found: {pass_cols}")

            # Save test data
            output_dir = Path("data/raw/soccerdata")
            output_dir.mkdir(parents=True, exist_ok=True)
            test_file = output_dir / "test_epl_2425.csv"
            passing.to_csv(test_file, index=False)
            print(f"\nTest data saved to: {test_file}")

            return True
        else:
            print("\n✗ No data returned")
            return False

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPossible issues:")
        print("1. Library not installed: pip install soccerdata")
        print("2. Network/connection issues")
        print("3. FBRef may have changed structure")
        return False


def process_for_training(df):
    """Process soccerdata output for our training pipeline."""

    # Rename columns to match our pipeline
    column_mapping = {
        'player': 'player_name',
        'team': 'team_name',
        'passes': 'passes_attempted',
        'passes_completed': 'passes_completed',
        'passes_pct': 'pass_accuracy',
        'passes_progressive': 'progressive_passes',
        'minutes_90s': 'minutes_played_90s',
        'xg': 'xG',
        'xa': 'xA'
    }

    df_renamed = df.rename(columns=column_mapping)

    # Convert minutes_90s to total minutes
    if 'minutes_played_90s' in df_renamed.columns:
        df_renamed['minutes_played'] = df_renamed['minutes_played_90s'] * 90

    # Select relevant columns for our model
    relevant_cols = [
        'player_name', 'team_name', 'league', 'season',
        'passes_attempted', 'passes_completed', 'pass_accuracy',
        'progressive_passes', 'minutes_played', 'xG', 'xA'
    ]

    available_cols = [col for col in relevant_cols if col in df_renamed.columns]

    return df_renamed[available_cols]


def main():
    """Main function."""
    import sys

    print("\n" + "="*60)
    print("SOCCERDATA FBREF COLLECTOR")
    print("="*60)
    print("\nThis uses the soccerdata library to collect FBRef data.")
    print("It handles rate limiting and anti-detection automatically!\n")

    print("Options:")
    print("1. Quick test (Premier League only)")
    print("2. Collect all leagues (takes time)")
    print("3. Exit")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = '1'  # Default to test

    if choice == '1':
        print("\nRunning quick test...")
        success = test_single_league()

        if success:
            print("\n✓ The soccerdata library works!")
            print("Run with option 2 to collect all leagues.")

    elif choice == '2':
        print("\nCollecting all leagues...")
        print("⚠️  This will take 30-60 minutes due to rate limiting")
        print("The library will pause between requests automatically.\n")

        data = collect_fbref_data()

        if not data.empty:
            # Process for our pipeline
            processed = process_for_training(data)

            print(f"\n✓ Collection complete!")
            print(f"Ready to train: python train_fbref.py --use-cached")

    else:
        print("Exiting")


if __name__ == "__main__":
    main()
"""
Full FBRef scraping script for collecting real match data.
Run this to collect data from top 5 leagues for training.
"""

from scrape_fbref import FBRefScraper
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def scrape_recent_data():
    """Scrape recent season data for model training."""

    print("\n" + "="*60)
    print("FBREF DATA COLLECTION FOR PASS PREDICTION MODEL")
    print("="*60)

    scraper = FBRefScraper()

    # Configuration
    leagues_to_scrape = {
        'Premier-League': 'Premier League',
        'La-Liga': 'La Liga',
        'Serie-A': 'Serie A',
        'Bundesliga': 'Bundesliga',
        'Ligue-1': 'Ligue 1'
    }

    seasons_to_scrape = [
        '2023-2024',  # Last complete season
        '2024-2025'   # Current season (partial)
    ]

    print("\nConfiguration:")
    print(f"  Leagues: {', '.join(leagues_to_scrape.values())}")
    print(f"  Seasons: {', '.join(seasons_to_scrape)}")

    # Ask for confirmation
    print("\n⚠️  WARNING: This will take several hours due to rate limiting!")
    print("The script will pause 3 seconds between each match to be respectful.")
    print("\nEstimated time: ~2-3 hours per league-season")

    response = input("\nProceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled by user")
        return

    print("\n" + "-"*60)
    print("Starting data collection...")
    print("-"*60)

    all_data = []
    total_leagues = len(leagues_to_scrape)
    total_seasons = len(seasons_to_scrape)
    current = 0

    for league_key, league_name in leagues_to_scrape.items():
        for season in seasons_to_scrape:
            current += 1
            progress = f"[{current}/{total_leagues * total_seasons}]"

            print(f"\n{progress} Scraping {league_name} {season}")
            print("-"*40)

            try:
                # Scrape this league-season
                data = scraper.scrape_league_season(league_key, season)

                if not data.empty:
                    all_data.append(data)
                    print(f"✓ Collected {len(data)} player records")

                    # Save intermediate results
                    intermediate_file = scraper.output_dir / f"{league_key}_{season}_scraped.parquet"
                    data.to_parquet(intermediate_file, index=False)
                else:
                    print(f"✗ No data collected for {league_name} {season}")

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                print("Intermediate results have been saved")
                break
            except Exception as e:
                print(f"✗ Error: {e}")
                continue

    # Combine all data
    if all_data:
        import pandas as pd
        from datetime import datetime

        combined = pd.concat(all_data, ignore_index=True)

        # Save final dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = scraper.output_dir / f"fbref_complete_{timestamp}.parquet"
        combined.to_parquet(output_file, index=False)

        # Also save as CSV for inspection
        csv_file = scraper.output_dir / f"fbref_complete_{timestamp}.csv"
        combined.to_csv(csv_file, index=False)

        print("\n" + "="*60)
        print("DATA COLLECTION COMPLETE!")
        print("="*60)
        print(f"\n📊 Summary:")
        print(f"  Total records: {len(combined):,}")
        print(f"  Unique players: {combined['player_name'].nunique():,}")
        print(f"  Date range: {combined.get('date', ['N/A'])[0]} to {combined.get('date', ['N/A'])[-1]}")
        print(f"\n💾 Data saved to:")
        print(f"  Parquet: {output_file}")
        print(f"  CSV: {csv_file}")
        print(f"\n✅ Next step:")
        print(f"  python train_fbref.py --use-cached")

    else:
        print("\n✗ No data was collected")


def quick_test():
    """Quick test with limited data."""
    print("\n🧪 Running quick test (1 league, current season, 5 matches)...")

    scraper = FBRefScraper()

    # Override to limit matches
    original_method = scraper.get_player_data

    def limited_get_player_data(match_links, league, season):
        print(f"  Limiting to first 5 matches (found {len(match_links)} total)")
        return original_method(match_links[:5], league, season)

    scraper.get_player_data = limited_get_player_data

    # Test with Premier League current season
    data = scraper.scrape_league_season('Premier-League', '2024-2025')

    if not data.empty:
        print(f"\n✓ Test successful! Collected {len(data)} player records")
        print("\nSample data:")
        print(data[['player_name', 'position', 'minutes_played', 'passes_attempted']].head())

        # Check data quality
        if 'passes_attempted' in data.columns:
            valid_passes = data['passes_attempted'].notna().sum()
            print(f"\n✓ Pass data available for {valid_passes}/{len(data)} records")
            print(f"  Average passes: {data['passes_attempted'].mean():.1f}")
        else:
            print("\n⚠️  Warning: No pass data found")

        return True
    else:
        print("\n✗ Test failed - no data collected")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Scrape FBRef data for pass prediction model')
    parser.add_argument('--test', action='store_true', help='Run quick test with limited data')
    parser.add_argument('--full', action='store_true', help='Run full scraping (takes hours)')

    args = parser.parse_args()

    if args.test:
        success = quick_test()
        if success:
            print("\n✓ Test complete! Use --full to scrape all data")
    elif args.full:
        scrape_recent_data()
    else:
        print("\nUsage:")
        print("  python scrape_fbref_full.py --test    # Quick test with 5 matches")
        print("  python scrape_fbref_full.py --full    # Full scraping (takes hours)")


if __name__ == "__main__":
    main()
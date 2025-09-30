"""
Full FBRef data collection using the Sec-CH-UA header fix.
Collects player pass data from top 5 European leagues.
"""

from fbref_fixed_scraper import FBRefFixedScraper
import pandas as pd
import time
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_all_leagues(matches_per_league=10):
    """Collect data from all top 5 leagues."""

    scraper = FBRefFixedScraper()

    # Test connection first
    if not scraper.test_connection():
        logger.error("Connection failed - cannot proceed")
        return None

    leagues = [
        'Premier-League',
        'La-Liga',
        'Serie-A',
        'Bundesliga',
        'Ligue-1'
    ]

    all_data = []

    for i, league in enumerate(leagues):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(leagues)}] Collecting {league}")
        print('='*60)

        try:
            # Scrape matches from this league
            league_data = scraper.scrape_recent_matches(
                league=league,
                num_matches=matches_per_league
            )

            if not league_data.empty:
                all_data.append(league_data)
                print(f"✓ Collected {len(league_data)} records from {league}")
            else:
                print(f"✗ No data collected from {league}")

            # Delay between leagues
            if i < len(leagues) - 1:
                print(f"Waiting 10 seconds before next league...")
                time.sleep(10)

        except Exception as e:
            logger.error(f"Error collecting {league}: {e}")
            continue

    if all_data:
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)

        # Save combined dataset
        output_dir = Path("data/raw/fbref_complete")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"fbref_all_leagues_{timestamp}.parquet"
        combined.to_parquet(output_file, index=False)

        csv_file = output_dir / f"fbref_all_leagues_{timestamp}.csv"
        combined.to_csv(csv_file, index=False)

        print(f"\n{'='*60}")
        print("COLLECTION COMPLETE!")
        print('='*60)
        print(f"Total records: {len(combined):,}")
        print(f"Unique players: {combined['player_name'].nunique():,}")
        print(f"\nData saved to:")
        print(f"  {output_file}")
        print(f"  {csv_file}")

        # Show breakdown by league
        print("\nRecords by league:")
        for league in leagues:
            league_count = len(combined[combined['league'] == league])
            print(f"  {league}: {league_count}")

        return combined

    return None


def quick_test():
    """Quick test with just Premier League."""
    print("\n🧪 Running quick test (Premier League, 5 matches)...")

    scraper = FBRefFixedScraper()

    data = scraper.scrape_recent_matches(
        league='Premier-League',
        num_matches=5
    )

    if not data.empty:
        print(f"\n✓ Test successful! Collected {len(data)} records")

        # Check data quality
        if 'passes_attempted' in data.columns:
            valid_passes = data['passes_attempted'].notna().sum()
            print(f"Pass data: {valid_passes}/{len(data)} records")
            print(f"Average passes: {data['passes_attempted'].mean():.1f}")

        print("\nSample:")
        print(data[['player_name', 'position', 'minutes_played', 'passes_attempted']].head())

        return True

    print("\n✗ Test failed")
    return False


def main():
    """Main collection script."""
    import sys

    print("\n" + "="*60)
    print("FBREF DATA COLLECTION - WORKING VERSION")
    print("="*60)
    print("\nUsing Sec-CH-UA header to bypass 403 errors")
    print("This actually works!\n")

    print("Options:")
    print("1. Quick test (5 Premier League matches)")
    print("2. Collect all leagues (10 matches each)")
    print("3. Full collection (50 matches per league)")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nChoice (1/2/3): ").strip()

    if choice == '1':
        success = quick_test()
        if success:
            print("\n✓ Ready for full collection!")

    elif choice == '2':
        print("\nCollecting 10 matches from each league...")
        print("Estimated time: 15-20 minutes\n")

        data = collect_all_leagues(matches_per_league=10)

        if data is not None:
            print(f"\n✓ Success! Ready to train model")
            print("Run: python train_fbref.py --use-cached")

    elif choice == '3':
        print("\n⚠️  Full collection will take 1-2 hours")
        confirm = input("Proceed? (yes/no): ").strip().lower()

        if confirm in ['yes', 'y']:
            data = collect_all_leagues(matches_per_league=50)

            if data is not None:
                print(f"\n✓ Full dataset collected!")
                print("This is high-quality training data!")
        else:
            print("Cancelled")

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
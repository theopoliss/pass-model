#!/usr/bin/env python3
"""
Collect full season data from FBRef for better model training.
This collects complete seasons instead of just recent matches.
"""

import pandas as pd
from fbref_fixed_scraper import FBRefFixedScraper
import time
from datetime import datetime
import sys
from pathlib import Path

def collect_full_seasons():
    """Collect multiple full seasons of data."""

    # Ensure output directory exists
    Path("data/raw/fbref_seasons").mkdir(parents=True, exist_ok=True)

    scraper = FBRefFixedScraper()

    # Test connection first
    if not scraper.test_connection():
        print("❌ Cannot connect to FBRef")
        return

    print("=" * 60)
    print("FBREF SEASON DATA COLLECTION")
    print("=" * 60)

    # Define what to collect
    leagues = {
        'Premier-League': 'Premier League',
        'La-Liga': 'La Liga',
        'Serie-A': 'Serie A',
        'Bundesliga': 'Bundesliga',
        'Ligue-1': 'Ligue 1'
    }

    # All available seasons
    all_seasons = [
        '2021-2022',  # Complete season
        '2022-2023',  # Complete season
        '2023-2024',  # Complete season
        '2024-2025',  # Last complete season
        '2025-2026'   # Current season (partial)
    ]

    print("\nCollection Plan:")
    print(f"- Leagues: {', '.join(leagues.values())}")
    print(f"- Seasons available: 2021-22 through current (2025-26)")

    print("\n" + "=" * 60)
    print("OPTIONS:")
    print("=" * 60)
    print("1. Quick test")
    print("   → 10 matches from 2024-25 Premier League")
    print("   → ~300 player records, takes ~1 minute")
    print()
    print("2. Last complete season (2024-25)")
    print("   → All 5 leagues × 380 matches = ~1,900 matches")
    print("   → ~57,000 player records")
    print("   → Takes 2-3 hours")
    print()
    print("3. Multiple seasons ⭐ RECOMMENDED")
    print("   → 4 complete seasons: 2021-22, 2022-23, 2023-24, 2024-25")
    print("   → All 5 leagues × 380 matches × 4 seasons = ~7,600 matches")
    print("   → ~228,000 player records")
    print("   → Takes 8-10 hours (can stop/resume)")
    print()
    print("4. Current season only (2025-26)")
    print("   → Partial season so far")
    print("   → All 5 leagues, ~500 matches")
    print("   → ~15,000 player records")
    print("   → Takes ~45 minutes")
    print()
    print("5. Custom")
    print("   → Choose your own league/season/matches")

    choice = input("\nEnter choice (1-5): ").strip()

    all_data = []

    if choice == '1':
        # Quick test
        print("\n🧪 Running quick test...")
        data = scraper.collect_matches('Premier-League', season='2024-2025', max_matches=10)
        if data:
            all_data.extend(data)

    elif choice == '2':
        # Last complete season (2024-25)
        print("\n📊 Collecting 2024-25 season (all leagues)...")
        season = '2024-2025'
        for league_key, league_name in leagues.items():
            print(f"\n[{league_name}]")
            data = scraper.collect_matches(league_key, season=season, max_matches=None)
            if data:
                all_data.extend(data)
                print(f"  ✓ Collected {len(data)} records")

            # Rate limiting between leagues
            if league_key != list(leagues.keys())[-1]:
                print("  Waiting 30s before next league...")
                time.sleep(30)

    elif choice == '3':
        # Multiple complete seasons - THE BIG ONE!
        print("\n📊 Collecting 4 complete seasons (2021-22 through 2024-25)...")
        print("⚠️  This will take 8-10 hours. You can stop (Ctrl+C) and resume later.")

        complete_seasons = ['2021-2022', '2022-2023', '2023-2024', '2024-2025']

        for season in complete_seasons:
            print(f"\n{'='*60}")
            print(f"SEASON {season}")
            print(f"{'='*60}")

            for league_key, league_name in leagues.items():
                print(f"\n[{league_name} - {season}]")
                try:
                    data = scraper.collect_matches(league_key, season=season, max_matches=None)
                    if data:
                        all_data.extend(data)
                        print(f"  ✓ Collected {len(data)} records")
                        print(f"  Total so far: {len(all_data)} records")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    print(f"  Continuing with next league...")

                # Rate limiting between leagues
                if league_key != list(leagues.keys())[-1]:
                    print("  Waiting 30s before next league...")
                    time.sleep(30)

            # Longer break between seasons
            if season != complete_seasons[-1]:
                print(f"\n✅ Completed {season}. Total records: {len(all_data)}")
                print("Taking 60s break before next season...")
                time.sleep(60)

    elif choice == '4':
        # Current season only
        print("\n📊 Collecting 2025-26 season (current/partial)...")
        season = '2025-2026'
        for league_key, league_name in leagues.items():
            print(f"\n[{league_name} - {season}]")
            data = scraper.collect_matches(league_key, season=season, max_matches=None)
            if data:
                all_data.extend(data)
                print(f"  ✓ Collected {len(data)} records")

            # Rate limiting between leagues
            if league_key != list(leagues.keys())[-1]:
                print("  Waiting 30s before next league...")
                time.sleep(30)

    elif choice == '5':
        # Custom
        print("\nAvailable leagues:", ', '.join(leagues.keys()))
        league = input("Enter league key: ").strip()
        season = input("Enter season (e.g., 2023-2024): ").strip()
        max_matches = input("Max matches (or 'all'): ").strip()
        max_matches = None if max_matches == 'all' else int(max_matches)

        data = scraper.collect_matches(league, season=season, max_matches=max_matches)
        if data:
            all_data.extend(data)

    # Save consolidated data
    if all_data:
        df = pd.DataFrame(all_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/raw/fbref_seasons/fbref_seasons_{timestamp}.parquet"
        df.to_parquet(output_file, index=False)

        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        print(f"✓ Total records: {len(df)}")
        print(f"✓ Unique players: {df['player_name'].nunique()}")
        print(f"✓ Unique matches: {df['match_url'].nunique() if 'match_url' in df.columns else 'N/A'}")
        print(f"✓ Saved to: {output_file}")

        # Data quality check
        print("\n📊 Data Quality:")
        print(f"- Pass completion: {df['passes_completed'].notna().mean():.1%}")
        print(f"- Minutes played: {df['minutes_played'].notna().mean():.1%}")
        print(f"- Avg passes/game: {df['passes_completed'].mean():.1f}")
        print(f"- Avg accuracy: {df['pass_accuracy'].mean():.1f}%")

        print("\n✅ Ready to train model with:")
        print(f"   python train_fbref.py --data {output_file}")

    else:
        print("\n❌ No data collected")

if __name__ == "__main__":
    collect_full_seasons()
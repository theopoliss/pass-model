"""Test scraping a single match page directly."""

import pandas as pd
import logging
from scrape_fbref import FBRefScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_match():
    """Test scraping the Liverpool vs Bournemouth match."""

    # Direct match URL
    match_url = "https://fbref.com/en/matches/a071faa8/Liverpool-Bournemouth-August-15-2025-Premier-League"

    logger.info(f"Testing with match: {match_url}")

    try:
        # Get tables from the page
        tables = pd.read_html(match_url)
        logger.info(f"Found {len(tables)} tables on page")

        # Print info about each table
        for i, table in enumerate(tables):
            try:
                # Get shape and sample columns
                shape = table.shape
                cols = list(table.columns)[:5] if hasattr(table.columns, '__iter__') else []

                print(f"\nTable {i}: {shape}")
                print(f"  Columns (first 5): {cols}")

                # Check if this might be a player stats table
                if shape[0] > 10 and shape[0] < 30:  # Likely player count
                    # Check for player names
                    for col in table.columns:
                        if 'Player' in str(col):
                            print(f"  → Likely player table! Player column: {col}")
                            print(f"  Sample players: {table[col].dropna().head(3).tolist()}")
                            break

                    # Check for passing stats
                    for col in table.columns:
                        if 'Pass' in str(col) or 'Att' in str(col) or 'Cmp' in str(col):
                            print(f"  → Found passing column: {col}")

            except Exception as e:
                print(f"  Error processing table {i}: {e}")

        # Now try the scraper's method
        print("\n" + "="*60)
        print("Testing scraper's extraction method:")
        print("="*60)

        scraper = FBRefScraper()
        player_data = scraper.process_match_tables(tables, game_id=1)

        if player_data:
            df = pd.DataFrame(player_data)
            print(f"\n✓ Extracted {len(df)} player records")
            print("\nColumns:", df.columns.tolist())
            print("\nSample data:")
            print(df[['player_name', 'position', 'minutes_played']].head())

            if 'passes_attempted' in df.columns:
                print(f"\n✓ Pass data found!")
                print(f"Average passes: {df['passes_attempted'].mean():.1f}")

            return df
        else:
            print("\n✗ No player data extracted")
            return None

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    data = test_single_match()
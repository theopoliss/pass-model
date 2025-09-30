"""Test FBRef scraper with a small sample."""

from scrape_fbref import FBRefScraper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_scraper():
    """Test scraper with one league, current season, limited matches."""
    scraper = FBRefScraper()

    # Test with Premier League current season
    logger.info("Testing scraper with Premier League 2024-2025 (first 3 matches)")

    # Modify scraper to limit matches
    original_method = scraper.get_player_data

    def limited_get_player_data(match_links, league, season):
        # Only scrape first 3 matches for testing
        return original_method(match_links[:3], league, season)

    scraper.get_player_data = limited_get_player_data

    # Run scraper
    data = scraper.scrape_league_season('Premier-League', '2024-2025')

    if not data.empty:
        print(f"\n✓ Successfully scraped {len(data)} records")
        print("\nColumns found:")
        print(data.columns.tolist())
        print("\nSample data:")
        print(data.head())
        print("\nData types:")
        print(data.dtypes)

        # Check for pass data
        if 'passes_attempted' in data.columns:
            print("\n✓ Pass data found!")
            print(f"Average passes: {data['passes_attempted'].mean():.1f}")
        else:
            print("\n⚠ Warning: No pass data found. May need to adjust table indices.")

        return data
    else:
        print("\n✗ No data scraped")
        return None


if __name__ == "__main__":
    test_scraper()
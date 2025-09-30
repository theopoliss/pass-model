"""
Simplified FBRef scraper that focuses on getting the data we need.
Works around 403 errors by being more careful with requests.
"""

import pandas as pd
import time
import logging
from pathlib import Path
from datetime import datetime
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleFBRefScraper:
    """Simplified scraper that's more respectful of rate limits."""

    def __init__(self):
        self.output_dir = Path("data/raw/fbref_scraped")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create session with browser-like headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        })

    def scrape_match(self, match_url):
        """Scrape a single match."""
        logger.info(f"Scraping match: {match_url}")

        # Be respectful with delays
        time.sleep(5)  # Longer delay to avoid 403s

        try:
            response = self.session.get(match_url, timeout=30)

            if response.status_code == 403:
                logger.warning("Got 403 - site is blocking requests")
                return None
            elif response.status_code == 429:
                logger.warning("Rate limited - waiting longer")
                time.sleep(30)
                return None

            response.raise_for_status()

            # Parse tables
            tables = pd.read_html(response.text)
            logger.info(f"Found {len(tables)} tables")

            # Extract player data
            player_data = self.extract_players_from_tables(tables)
            return player_data

        except Exception as e:
            logger.error(f"Error scraping match: {e}")
            return None

    def extract_players_from_tables(self, tables):
        """Extract player data from match tables."""
        all_players = []

        # Based on our testing, these are the table indices
        team_configs = [
            {'summary': 3, 'passing': 4, 'is_home': 1},   # Home team
            {'summary': 10, 'passing': 11, 'is_home': 0}  # Away team
        ]

        for config in team_configs:
            try:
                if config['summary'] >= len(tables):
                    continue

                # Get summary table
                summary = tables[config['summary']]

                # Handle multi-level columns
                if isinstance(summary.columns, pd.MultiIndex):
                    summary.columns = ['_'.join(col).strip() if col[1] else col[0]
                                     for col in summary.columns.values]

                # Get passing table if available
                passing = None
                if config['passing'] < len(tables):
                    passing = tables[config['passing']]
                    if isinstance(passing.columns, pd.MultiIndex):
                        passing.columns = ['_'.join(col).strip() if col[1] else col[0]
                                         for col in passing.columns.values]

                # Process players (skip last row which is usually totals)
                for i in range(len(summary) - 1):
                    player_data = self.extract_player_row(
                        summary.iloc[i],
                        passing.iloc[i] if passing is not None and i < len(passing) else None,
                        config['is_home']
                    )
                    if player_data:
                        all_players.append(player_data)

            except Exception as e:
                logger.debug(f"Error processing team: {e}")

        return all_players

    def extract_player_row(self, summary_row, passing_row, is_home):
        """Extract data for a single player."""
        try:
            # Find player name column
            player_name = None
            for col in summary_row.index:
                if 'Player' in str(col):
                    player_name = summary_row[col]
                    break

            if not player_name or pd.isna(player_name):
                return None

            # Extract basic data
            player_dict = {
                'player_name': player_name,
                'is_home': is_home,
                'date': datetime.now().strftime('%Y-%m-%d')
            }

            # Extract position and minutes
            for col in summary_row.index:
                if 'Pos' in str(col):
                    player_dict['position'] = summary_row[col]
                elif 'Min' in str(col):
                    min_val = summary_row[col]
                    if pd.notna(min_val):
                        # Handle "90+3" format
                        player_dict['minutes_played'] = int(str(min_val).split('+')[0])

            # Extract passing data if available
            if passing_row is not None:
                for col in passing_row.index:
                    if 'Total_Att' in str(col) or (col == 'Att'):
                        player_dict['passes_attempted'] = self._to_numeric(passing_row[col])
                    elif 'Total_Cmp' in str(col) or (col == 'Cmp'):
                        player_dict['passes_completed'] = self._to_numeric(passing_row[col])

            return player_dict

        except Exception as e:
            logger.debug(f"Error extracting player: {e}")
            return None

    def _to_numeric(self, val):
        """Convert value to numeric, return 0 if failed."""
        if pd.isna(val):
            return 0
        try:
            return float(val)
        except:
            return 0

    def test_single_match(self):
        """Test with a known match URL."""
        # Use the Liverpool-Bournemouth match as test
        test_url = "https://fbref.com/en/matches/a071faa8/Liverpool-Bournemouth-August-15-2025-Premier-League"

        logger.info("Testing with single match...")
        players = self.scrape_match(test_url)

        if players:
            df = pd.DataFrame(players)
            logger.info(f"✓ Extracted {len(df)} players")

            # Save test data
            test_file = self.output_dir / f"test_match_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(test_file, index=False)

            print("\nSample data:")
            print(df.head())

            if 'passes_attempted' in df.columns:
                print(f"\n✓ Pass data found!")
                print(f"Average passes: {df['passes_attempted'].mean():.1f}")

            return df
        else:
            logger.error("Failed to extract player data")
            return None


def main():
    """Test the simplified scraper."""
    scraper = SimpleFBRefScraper()

    print("\n" + "="*60)
    print("TESTING SIMPLIFIED FBREF SCRAPER")
    print("="*60)

    # Test with single match
    data = scraper.test_single_match()

    if data is not None and not data.empty:
        print(f"\n✓ Success! Ready to scrape more matches")
        print("\nNote: Due to rate limiting, you should:")
        print("1. Use longer delays between requests (5+ seconds)")
        print("2. Consider scraping during off-peak hours")
        print("3. Save progress frequently")
        print("4. Use a VPN if getting consistent 403s")
    else:
        print("\n✗ Test failed")
        print("\nTroubleshooting:")
        print("1. FBRef may be blocking automated requests")
        print("2. Try accessing the URL in a browser first")
        print("3. Consider using Selenium for JavaScript-rendered content")
        print("4. Check if the site structure has changed")


if __name__ == "__main__":
    main()
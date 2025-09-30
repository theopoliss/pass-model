"""
Understat scraper - alternative source that's less aggressive about blocking.
Understat has xG data and some passing statistics.
"""

import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import re
import time
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnderstatScraper:
    """Scraper for Understat.com - typically less blocking than FBRef."""

    BASE_URL = "https://understat.com"

    LEAGUES = {
        'EPL': 'Premier League',
        'La_liga': 'La Liga',
        'Serie_A': 'Serie A',
        'Bundesliga': 'Bundesliga',
        'Ligue_1': 'Ligue 1'
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.output_dir = Path("data/raw/understat")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_league_data(self, league='EPL', season='2024'):
        """Get player data for a league season."""
        url = f"{self.BASE_URL}/league/{league}/{season}"

        logger.info(f"Fetching {league} {season}...")

        try:
            response = self.session.get(url)
            response.raise_for_status()

            # Parse the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Understat stores data in JavaScript variables
            # Look for the script containing player data
            scripts = soup.find_all('script')

            for script in scripts:
                if 'var playersData' in script.text:
                    # Extract JSON from JavaScript
                    json_str = re.search(r'var playersData\s*=\s*JSON\.parse\((.*?)\);', script.text)
                    if json_str:
                        # Parse the escaped JSON
                        data_str = json_str.group(1)
                        # Remove quotes and parse
                        data_str = data_str.strip("'")
                        # Decode the string
                        data = json.loads(data_str.encode().decode('unicode_escape'))

                        return self.process_player_data(data)

            logger.warning("Could not find player data in page")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching {league}: {e}")
            return pd.DataFrame()

    def process_player_data(self, raw_data):
        """Process raw Understat data into DataFrame."""
        players = []

        for player_id, player_info in raw_data.items():
            try:
                player_dict = {
                    'player_id': player_id,
                    'player_name': player_info.get('player_name', ''),
                    'team': player_info.get('team_title', ''),
                    'position': player_info.get('position', ''),

                    # Basic stats
                    'games': int(player_info.get('games', 0)),
                    'minutes_played': int(player_info.get('time', 0)),
                    'goals': int(player_info.get('goals', 0)),
                    'assists': int(player_info.get('assists', 0)),

                    # xG metrics
                    'xG': float(player_info.get('xG', 0)),
                    'xA': float(player_info.get('xA', 0)),
                    'npxG': float(player_info.get('npxG', 0)),

                    # Shooting
                    'shots': int(player_info.get('shots', 0)),
                    'key_passes': int(player_info.get('key_passes', 0)),

                    # Per 90 stats
                    'xG_per90': float(player_info.get('xG90', 0)),
                    'xA_per90': float(player_info.get('xA90', 0))
                }

                # Calculate passes (estimate from key passes and assists)
                # Understat doesn't have detailed passing, but we can estimate
                if player_dict['minutes_played'] > 0:
                    # Rough estimates based on position and involvement
                    if 'M' in player_dict['position'] or 'midfielder' in player_dict['position'].lower():
                        player_dict['passes_estimated'] = player_dict['minutes_played'] * 0.5
                    elif 'D' in player_dict['position'] or 'defender' in player_dict['position'].lower():
                        player_dict['passes_estimated'] = player_dict['minutes_played'] * 0.45
                    else:  # Forwards
                        player_dict['passes_estimated'] = player_dict['minutes_played'] * 0.3

                players.append(player_dict)

            except Exception as e:
                logger.debug(f"Error processing player: {e}")
                continue

        return pd.DataFrame(players)

    def get_match_data(self, match_id):
        """Get detailed match data."""
        url = f"{self.BASE_URL}/match/{match_id}"

        try:
            response = self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract match data from JavaScript
            scripts = soup.find_all('script')

            for script in scripts:
                if 'var match_info' in script.text:
                    # Extract and parse match data
                    json_str = re.search(r'var match_info\s*=\s*JSON\.parse\((.*?)\);', script.text)
                    if json_str:
                        data_str = json_str.group(1).strip("'")
                        data = json.loads(data_str.encode().decode('unicode_escape'))
                        return data

        except Exception as e:
            logger.error(f"Error fetching match {match_id}: {e}")

        return None

    def collect_recent_data(self):
        """Collect recent data from all major leagues."""
        all_data = []

        for league_key, league_name in self.LEAGUES.items():
            logger.info(f"\nCollecting {league_name}...")

            # Get current season data
            df = self.get_league_data(league_key, '2024')

            if not df.empty:
                df['league'] = league_name
                all_data.append(df)
                logger.info(f"  ✓ Collected {len(df)} players")

            # Be respectful with rate limiting
            time.sleep(3)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)

            # Save data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"understat_{timestamp}.parquet"
            combined.to_parquet(output_file, index=False)

            csv_file = self.output_dir / f"understat_{timestamp}.csv"
            combined.to_csv(csv_file, index=False)

            logger.info(f"\n✓ Total: {len(combined)} player records")
            logger.info(f"Saved to: {output_file}")

            return combined

        return pd.DataFrame()


def test_understat():
    """Test if Understat scraping works."""
    print("\n" + "="*60)
    print("TESTING UNDERSTAT SCRAPER")
    print("="*60)
    print("\nUnderstat is generally less aggressive about blocking.")
    print("It has xG data but limited passing statistics.\n")

    scraper = UnderstatScraper()

    # Test with Premier League
    print("Testing with Premier League 2024...")
    df = scraper.get_league_data('EPL', '2024')

    if not df.empty:
        print(f"\n✓ Success! Scraped {len(df)} players")
        print("\nSample data:")
        print(df[['player_name', 'position', 'games', 'xG', 'xA']].head())

        print("\nNote: Understat doesn't have detailed pass data,")
        print("but we can use xG/xA and position to estimate involvement.")

        # Save test data
        test_file = scraper.output_dir / f"test_epl_2024.csv"
        df.to_csv(test_file, index=False)
        print(f"\nTest data saved to: {test_file}")

        return True
    else:
        print("\n✗ Scraping failed")
        print("Possible reasons:")
        print("1. Site structure changed")
        print("2. Network issues")
        print("3. Rate limiting")
        return False


def main():
    """Main function."""
    import sys

    print("\nOptions:")
    print("1. Test scraper (Premier League only)")
    print("2. Collect all leagues")
    print("3. Exit")

    choice = input("\nChoice (1/2/3): ").strip()

    if choice == '1':
        success = test_understat()
        if success:
            print("\n✓ Understat scraping works!")
            print("Use option 2 to collect all leagues.")

    elif choice == '2':
        print("\nCollecting data from all leagues...")
        print("This will take a few minutes...\n")

        scraper = UnderstatScraper()
        data = scraper.collect_recent_data()

        if not data.empty:
            print(f"\n✓ Collection complete!")
            print(f"Total players: {len(data)}")
            print("\nLeague breakdown:")
            print(data.groupby('league').size())

    else:
        print("Exiting")
        sys.exit(0)


if __name__ == "__main__":
    main()
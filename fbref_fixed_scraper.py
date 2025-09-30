"""
FBRef scraper that works by including the Sec-CH-UA header.
This bypasses the 403 error by making requests look like they come from a real browser.
"""

import tls_client
import pandas as pd
import time
import random
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FBRefFixedScraper:
    """FBRef scraper using the Sec-CH-UA header fix."""

    BASE_URL = "https://fbref.com"

    LEAGUE_URLS = {
        'Premier-League': '/en/comps/9/Premier-League-Stats',
        'La-Liga': '/en/comps/12/La-Liga-Stats',
        'Serie-A': '/en/comps/11/Serie-A-Stats',
        'Bundesliga': '/en/comps/20/Bundesliga-Stats',
        'Ligue-1': '/en/comps/13/Ligue-1-Stats'
    }

    def __init__(self):
        """Initialize with tls_client session."""
        # Use tls_client instead of requests
        self.session = tls_client.Session(
            client_identifier="chrome_120",
            random_tls_extension_order=True
        )

        self.output_dir = Path("data/raw/fbref_fixed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _make_request(self, url):
        """Make a request with the Sec-CH-UA header."""
        headers = {
            # The magic header that bypasses the 403
            "sec-ch-ua": '"Not)A;Brand";v="99", "Google Chrome";v="120", "Chromium";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

        try:
            response = self.session.get(url, headers=headers)
            # tls_client response doesn't have raise_for_status, check manually
            if response.status_code != 200:
                logger.error(f"Got status code {response.status_code} for {url}")
                if response.status_code == 403:
                    logger.error("Still getting 403 - FBRef may have updated their detection")
                return None
            return response
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def test_connection(self):
        """Test if we can access FBRef."""
        test_url = "https://fbref.com/en/comps/"
        logger.info(f"Testing connection to {test_url}")

        response = self._make_request(test_url)

        if response and response.status_code == 200:
            logger.info("✓ Connection successful! No 403 error!")
            return True
        else:
            logger.error(f"✗ Connection failed. Status: {response.status_code if response else 'No response'}")
            return False

    def get_league_fixtures(self, league='Premier-League', season='2024-2025'):
        """Get fixtures/matches for a league season."""
        # Build URL for specific season
        league_id = {
            'Premier-League': '9',
            'La-Liga': '12',
            'Serie-A': '11',
            'Bundesliga': '20',
            'Ligue-1': '13'
        }.get(league)

        if not league_id:
            logger.error(f"Unknown league: {league}")
            return []

        # For any season, use the schedule/fixtures page which has all match links
        if season != '2024-2025':
            # Historical season - use schedule page
            url = f'{self.BASE_URL}/en/comps/{league_id}/{season}/schedule/{season}-{league}-Scores-and-Fixtures'
        else:
            # Current season - use main stats page
            league_path = self.LEAGUE_URLS.get(league, '')
            url = self.BASE_URL + league_path
        logger.info(f"Fetching fixtures from: {url}")

        response = self._make_request(url)
        if not response:
            return []

        # Parse the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find match links - look for specific match report URLs
        match_links = []
        seen_urls = set()

        for link in soup.find_all('a', href=True):
            href = link['href']
            # Look for match report URLs (format: /en/matches/{match-id}/Team-vs-Team-Date)
            # Skip date-only links like /en/matches/2023-08-11
            if '/en/matches/' in href and len(href.split('/')) >= 4:
                parts = href.split('/')
                # Must have a match ID (not just a date) - match IDs are typically 8 chars
                if len(parts) >= 4 and parts[3] and len(parts[3]) == 8:
                    full_url = self.BASE_URL + href if not href.startswith('http') else href
                    if full_url not in seen_urls:
                        seen_urls.add(full_url)
                        match_links.append((full_url, link.text))

        logger.info(f"Found {len(match_links)} unique match links")
        return match_links

    def scrape_match(self, match_url):
        """Scrape player data from a match page."""
        logger.info(f"Scraping match: {match_url.split('/')[-1][:40]}...")

        response = self._make_request(match_url)
        if not response:
            return []

        # Parse tables
        try:
            tables = pd.read_html(response.text)
            logger.info(f"  Found {len(tables)} tables")

            # Extract player data using our existing logic
            player_data = self._extract_player_data(tables)
            return player_data

        except Exception as e:
            logger.error(f"  Error parsing tables: {e}")
            return []

    def _extract_player_data(self, tables):
        """Extract player passing data from match tables."""
        all_players = []

        # Based on FBRef structure:
        # Tables 3-5: Home team stats
        # Tables 10-12: Away team stats
        team_configs = [
            {'summary': 3, 'passing': 4, 'is_home': 1},
            {'summary': 10, 'passing': 11, 'is_home': 0}
        ]

        for config in team_configs:
            try:
                if config['summary'] >= len(tables):
                    continue

                summary_df = tables[config['summary']].copy()
                passing_df = tables[config['passing']].copy() if config['passing'] < len(tables) else None

                # Handle multi-level columns
                if isinstance(summary_df.columns, pd.MultiIndex):
                    summary_df.columns = ['_'.join(col).strip() if col[1] else col[0]
                                         for col in summary_df.columns.values]

                if passing_df is not None and isinstance(passing_df.columns, pd.MultiIndex):
                    passing_df.columns = ['_'.join(col).strip() if col[1] else col[0]
                                        for col in passing_df.columns.values]

                # Extract player data (skip last row which is usually totals)
                for i in range(len(summary_df) - 1):
                    player_dict = self._parse_player_row(
                        summary_df.iloc[i],
                        passing_df.iloc[i] if passing_df is not None and i < len(passing_df) else None,
                        config['is_home']
                    )
                    if player_dict:
                        all_players.append(player_dict)

            except Exception as e:
                logger.debug(f"  Error processing team: {e}")

        logger.info(f"  Extracted {len(all_players)} players")
        return all_players

    def _parse_player_row(self, summary_row, passing_row, is_home):
        """Parse a single player's data."""
        try:
            # Find player name
            player_name = None
            for col in summary_row.index:
                if 'Player' in str(col):
                    player_name = summary_row[col]
                    break

            if not player_name or pd.isna(player_name):
                return None

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
                        player_dict['minutes_played'] = int(str(min_val).split('+')[0])

            # Extract passing data if available
            if passing_row is not None:
                for col in passing_row.index:
                    if 'Total_Att' in str(col) or col == 'Att':
                        player_dict['passes_attempted'] = self._to_numeric(passing_row[col])
                    elif 'Total_Cmp' in str(col) or col == 'Cmp':
                        player_dict['passes_completed'] = self._to_numeric(passing_row[col])
                    elif 'Total_Cmp%' in str(col) or 'Cmp%' in str(col):
                        player_dict['pass_accuracy'] = self._to_numeric(passing_row[col])
                    elif 'PrgP' in str(col):
                        player_dict['progressive_passes'] = self._to_numeric(passing_row[col])

            return player_dict

        except Exception as e:
            logger.debug(f"    Error parsing player: {e}")
            return None

    def _to_numeric(self, val):
        """Convert to numeric, return 0 if failed."""
        if pd.isna(val):
            return 0
        try:
            return float(val)
        except:
            return 0

    def collect_matches(self, league='Premier-League', season='2024-2025', max_matches=None):
        """Collect matches from a specific league and season."""
        # Get match links for the season
        match_links = self.get_league_fixtures(league, season)

        if not match_links:
            logger.error(f"No matches found for {league} {season}")
            return []

        # Limit to requested number if specified
        if max_matches:
            match_links = match_links[:max_matches]

        logger.info(f"Found {len(match_links)} matches to collect")

        all_players = []
        for i, (match_url, match_info) in enumerate(match_links, 1):
            logger.info(f"\n[{i}/{len(match_links)}] Processing match...")

            players = self.scrape_match(match_url)
            if players:
                # Add match metadata
                for player in players:
                    player['match_url'] = match_url
                    player['league'] = league
                    player['season'] = season
                all_players.extend(players)

            # Rate limiting
            if i < len(match_links):
                wait = random.uniform(2, 4)
                logger.info(f"  Waiting {wait:.1f}s...")
                time.sleep(wait)

        return all_players

    def scrape_recent_matches(self, league='Premier-League', num_matches=5):
        """Scrape recent matches from a league (backward compatibility)."""
        # Get match links
        match_links = self.get_league_fixtures(league)

        if not match_links:
            logger.error("No matches found")
            return pd.DataFrame()

        # Limit to requested number
        match_links = match_links[:num_matches]

        all_player_data = []

        for i, match_url in enumerate(match_links):
            logger.info(f"\n[{i+1}/{len(match_links)}] Processing match...")

            player_data = self.scrape_match(match_url)

            if player_data:
                for player in player_data:
                    player['match_url'] = match_url
                    player['league'] = league

                all_player_data.extend(player_data)

            # Respectful delay
            if i < len(match_links) - 1:
                delay = random.uniform(2, 4)
                logger.info(f"  Waiting {delay:.1f}s...")
                time.sleep(delay)

        if all_player_data:
            df = pd.DataFrame(all_player_data)

            # Save data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"fbref_{league}_{timestamp}.parquet"
            df.to_parquet(output_file, index=False)

            csv_file = self.output_dir / f"fbref_{league}_{timestamp}.csv"
            df.to_csv(csv_file, index=False)

            logger.info(f"\n✓ Saved {len(df)} records to {output_file}")

            return df

        return pd.DataFrame()


def main():
    """Test the fixed scraper."""
    print("\n" + "="*60)
    print("FBREF FIXED SCRAPER - Using Sec-CH-UA Header")
    print("="*60)
    print("\nThis uses the fix discovered on GitHub:")
    print("FBRef checks for Sec-CH-UA header, we provide it.\n")

    scraper = FBRefFixedScraper()

    # Test connection first
    if not scraper.test_connection():
        print("\n✗ Connection test failed")
        return

    print("\n✓ Connection test passed! Let's scrape some data.\n")

    # Scrape recent Premier League matches
    print("Scraping 3 recent Premier League matches...")
    data = scraper.scrape_recent_matches(league='Premier-League', num_matches=3)

    if not data.empty:
        print(f"\n✓ SUCCESS! Scraped {len(data)} player records")

        print("\nSample data:")
        print(data[['player_name', 'position', 'minutes_played', 'passes_attempted']].head(10))

        if 'passes_attempted' in data.columns:
            valid_passes = data['passes_attempted'].notna().sum()
            print(f"\n✓ Pass data available for {valid_passes}/{len(data)} records")
            print(f"  Average passes: {data['passes_attempted'].mean():.1f}")

        print(f"\n✓ Data saved to: data/raw/fbref_fixed/")
        print("\nYou can now train the model with this real FBRef data!")
        print("Run: python train_fbref.py --use-cached")
    else:
        print("\n✗ No data scraped")


if __name__ == "__main__":
    main()
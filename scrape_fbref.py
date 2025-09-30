"""
Adapted FBRef scraper for pass prediction model.
Based on existing scraper but modified to collect passing-specific data.
"""

from bs4 import BeautifulSoup as soup
import requests
import pandas as pd
import time
import re
from functools import reduce
import sys
from urllib.error import HTTPError
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class FBRefScraper:
    """Scraper for FBRef player match data."""

    # League mappings
    LEAGUES = {
        'Premier-League': '9',
        'La-Liga': '12',
        'Serie-A': '11',
        'Ligue-1': '13',
        'Bundesliga': '20'
    }

    # Recent seasons to scrape
    SEASONS = ['2022-2023', '2023-2024', '2024-2025']

    def __init__(self, output_dir='data/raw/fbref_scraped'):
        """Initialize scraper."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        # More realistic headers to avoid 403 errors
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    def scrape_all_leagues(self, seasons=None, leagues=None):
        """Scrape data for all specified leagues and seasons."""
        seasons = seasons or self.SEASONS
        leagues = leagues or list(self.LEAGUES.keys())

        all_data = []

        for league in leagues:
            for season in seasons:
                logger.info(f"Scraping {league} {season}")
                try:
                    league_data = self.scrape_league_season(league, season)
                    if not league_data.empty:
                        all_data.append(league_data)
                        # Save intermediate results
                        self._save_data(league_data, f"{league}_{season}")
                except Exception as e:
                    logger.error(f"Error scraping {league} {season}: {e}")
                    continue

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            self._save_data(combined, "combined_data")
            return combined
        return pd.DataFrame()

    def scrape_league_season(self, league, season):
        """Scrape a specific league season."""
        league_id = self.LEAGUES.get(league)
        if not league_id:
            logger.error(f"Unknown league: {league}")
            return pd.DataFrame()

        # Get fixture list URL
        url = f'https://fbref.com/en/comps/{league_id}/{season}/schedule/{season}-{league}-Scores-and-Fixtures'

        # Get fixture data
        fixture_data = self.get_fixture_data(url, league, season)

        # Get match links
        match_links = self.get_match_links(url, league)
        logger.info(f"Found {len(match_links)} matches")

        # Get player data for each match
        player_data = self.get_player_data(match_links[:10], league, season)  # Limit to 10 for testing

        # Merge fixture and player data if both exist
        if not fixture_data.empty and not player_data.empty:
            # Merge on game_id
            merged = pd.merge(player_data, fixture_data, on='game_id', how='left')
            return merged

        return player_data if not player_data.empty else fixture_data

    def get_fixture_data(self, url, league, season):
        """Get fixture/match metadata."""
        logger.info('Getting fixture data...')
        try:
            tables = pd.read_html(url)

            # Get fixtures with relevant columns
            fixtures = tables[0][['Wk', 'Day', 'Date', 'Home', 'Away']].dropna()

            # Try to get additional columns if they exist
            for col in ['xG', 'xG.1', 'Score']:
                if col in tables[0].columns:
                    fixtures[col] = tables[0][col]

            fixtures['season'] = season
            fixtures['league'] = league
            fixtures["game_id"] = fixtures.index

            logger.info(f'Found {len(fixtures)} fixtures')
            return fixtures

        except Exception as e:
            logger.error(f"Error getting fixture data: {e}")
            return pd.DataFrame()

    def get_match_links(self, url, league):
        """Extract individual match links from fixture page."""
        logger.info('Getting match links...')
        match_links = []

        try:
            # Add delay before request to be respectful
            time.sleep(2)
            response = self.session.get(url)
            response.raise_for_status()

            page_soup = soup(response.content, "html.parser")

            # Look for all links that match the pattern /en/matches/[8-char-id]/Team-Team-Date
            links = page_soup.find_all('a', href=re.compile(r'/en/matches/[a-f0-9]{8}/'))

            for link in links:
                href = link.get('href', '')
                # Make sure it's a match page (has team names)
                if '-' in href and 'Premier-League' in href or 'La-Liga' in href or 'Serie-A' in href or 'Bundesliga' in href or 'Ligue-1' in href:
                    full_link = 'https://fbref.com' + href if not href.startswith('http') else href
                    if full_link not in match_links:
                        match_links.append(full_link)
                        logger.debug(f"Found match link: {full_link}")

        except Exception as e:
            logger.error(f"Error getting match links: {e}")

        logger.info(f"Found {len(match_links)} match links")
        return match_links

    def get_player_data(self, match_links, league, season):
        """Scrape player data from match pages."""
        all_player_data = []

        for count, link in enumerate(match_links):
            logger.info(f'Scraping match {count+1}/{len(match_links)}')

            try:
                # Get page tables
                tables = pd.read_html(link)

                # Process tables to extract player stats
                match_data = self.process_match_tables(tables, count)

                if match_data:
                    for player in match_data:
                        player['league'] = league
                        player['season'] = season
                        player['match_url'] = link

                    all_player_data.extend(match_data)

            except Exception as e:
                logger.error(f'Error scraping {link}: {e}')

            # Rate limiting
            time.sleep(3)

        return pd.DataFrame(all_player_data)

    def process_match_tables(self, tables, game_id):
        """Extract player statistics from match tables.

        Current FBRef structure (2024-2025):
        - Table 3: Home team summary stats
        - Table 4: Home team passing details
        - Table 10: Away team summary stats
        - Table 11: Away team passing details
        """
        player_data = []

        try:
            # Updated indices based on actual FBRef structure
            team_configs = [
                {'summary': 3, 'passing': 4, 'is_home': 1, 'team': 'home'},   # Home team
                {'summary': 10, 'passing': 11, 'is_home': 0, 'team': 'away'}  # Away team
            ]

            for config in team_configs:
                if config['summary'] < len(tables):
                    team_data = self.extract_team_data(tables, config, game_id)
                    player_data.extend(team_data)
                    logger.debug(f"Extracted {len(team_data)} players from {config['team']} team")

        except Exception as e:
            logger.debug(f"Error processing tables: {e}")

        return player_data

    def extract_team_data(self, tables, config, game_id):
        """Extract data for one team from tables."""
        team_data = []

        try:
            # Get summary table
            summary_table = tables[config['summary']].copy()

            # Handle multi-level columns (FBRef uses these)
            if isinstance(summary_table.columns, pd.MultiIndex):
                # Flatten column names
                summary_table.columns = ['_'.join(col).strip() if col[1] else col[0]
                                        for col in summary_table.columns.values]

            # Get passing table
            passing_table = None
            if config['passing'] < len(tables):
                passing_table = tables[config['passing']].copy()
                if isinstance(passing_table.columns, pd.MultiIndex):
                    # For passing table, extract the important columns
                    passing_table.columns = ['_'.join(col).strip() if col[1] else col[0]
                                            for col in passing_table.columns.values]

            # Find the player column (might be different names)
            player_col = None
            for col in summary_table.columns:
                if 'Player' in col:
                    player_col = col
                    break

            if not player_col:
                logger.debug("No player column found")
                return team_data

            # Process each player row (skip last row which is usually totals)
            for idx in range(len(summary_table) - 1):
                row = summary_table.iloc[idx]

                player_name = row.get(player_col, '')
                if pd.isna(player_name) or player_name == '':
                    continue

                # Extract basic info
                player_dict = {
                    'player_name': player_name,
                    'position': self._extract_value(row, ['Pos', 'Position']),
                    'minutes_played': self._parse_minutes(self._extract_value(row, ['Min', 'Minutes'])),
                    'game_id': game_id,
                    'is_home': config['is_home']
                }

                # Extract passing stats from passing table
                if passing_table is not None:
                    # Find player in passing table
                    pass_player_col = None
                    for col in passing_table.columns:
                        if 'Player' in col:
                            pass_player_col = col
                            break

                    if pass_player_col:
                        player_pass_rows = passing_table[passing_table[pass_player_col] == player_name]
                        if not player_pass_rows.empty:
                            pass_row = player_pass_rows.iloc[0]

                            # Extract passing columns (handle different naming)
                            player_dict.update({
                                'passes_attempted': self._extract_numeric(pass_row, ['Total_Att', 'Att', 'Passes_Att']),
                                'passes_completed': self._extract_numeric(pass_row, ['Total_Cmp', 'Cmp', 'Passes_Cmp']),
                                'pass_accuracy': self._extract_numeric(pass_row, ['Total_Cmp%', 'Cmp%', 'Pass%']),
                                'progressive_passes': self._extract_numeric(pass_row, ['PrgP', 'Progressive', 'Prog']),
                            })

                # Extract other stats from summary table
                player_dict.update({
                    'goals': self._extract_numeric(row, ['Gls', 'Goals']),
                    'assists': self._extract_numeric(row, ['Ast', 'Assists']),
                    'shots': self._extract_numeric(row, ['Sh', 'Shots']),
                    'xG': self._extract_numeric(row, ['xG', 'Expected_xG']),
                    'xA': self._extract_numeric(row, ['xAG', 'xA', 'Expected_xAG']),
                    'touches': self._extract_numeric(row, ['Touches', 'Touches_Touches'])
                })

                team_data.append(player_dict)

        except Exception as e:
            logger.debug(f"Error extracting team data: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return team_data

    def _extract_value(self, row, possible_columns):
        """Extract value from row trying multiple column names."""
        for col in possible_columns:
            if col in row.index:
                return row[col]
            # Also try with underscores
            for idx in row.index:
                if col.lower() in idx.lower():
                    return row[idx]
        return None

    def _extract_numeric(self, row, possible_columns):
        """Extract numeric value from row trying multiple column names."""
        val = self._extract_value(row, possible_columns)
        if pd.isna(val):
            return 0
        try:
            return float(val)
        except:
            return 0

    def _parse_minutes(self, min_str):
        """Parse minutes played from string format."""
        if pd.isna(min_str):
            return 0
        if isinstance(min_str, (int, float)):
            return int(min_str)
        # Handle format like "90" or "90+3"
        min_str = str(min_str).split('+')[0]
        try:
            return int(min_str)
        except:
            return 0

    def _save_data(self, df, name):
        """Save dataframe to parquet and CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as parquet (efficient)
        parquet_file = self.output_dir / f"{name}_{timestamp}.parquet"
        df.to_parquet(parquet_file, index=False)

        # Save as CSV (human-readable)
        csv_file = self.output_dir / f"{name}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)

        logger.info(f"Saved {len(df)} records to {parquet_file.name}")


def main():
    """Main function to run scraper."""
    scraper = FBRefScraper()

    # Interactive mode
    print("\n=== FBRef Data Scraper ===\n")
    print("Available leagues: Premier-League, La-Liga, Serie-A, Ligue-1, Bundesliga")
    print("Available seasons: 2022-2023, 2023-2024, 2024-2025\n")

    # Get user input
    league_input = input("Enter league (or 'all' for all leagues): ").strip()
    season_input = input("Enter season (or 'all' for all seasons): ").strip()

    leagues = None if league_input.lower() == 'all' else [league_input]
    seasons = None if season_input.lower() == 'all' else [season_input]

    print(f"\nStarting scrape...")
    print(f"Leagues: {leagues or 'All'}")
    print(f"Seasons: {seasons or 'All'}")
    print("\nThis may take a while due to rate limiting (3 seconds between requests)...\n")

    # Run scraper
    data = scraper.scrape_all_leagues(seasons=seasons, leagues=leagues)

    if not data.empty:
        print(f"\n✓ Successfully scraped {len(data)} player-match records")
        print(f"Data saved to: {scraper.output_dir}")

        # Show sample
        print("\nSample data:")
        print(data[['player_name', 'position', 'minutes_played', 'passes_attempted', 'league', 'season']].head())

        print("\n✓ You can now train the model with: python train_fbref.py --use-cached")
    else:
        print("\n✗ No data was scraped. Check logs for errors.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
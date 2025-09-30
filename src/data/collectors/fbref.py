"""FBRef data collector for recent match data from top European leagues."""

import time
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class FBRefCollector:
    """Collect player match data from FBRef for recent seasons."""

    BASE_URL = "https://fbref.com"

    # League URLs for Big 5 European leagues (2024-2025 season specific)
    LEAGUE_URLS = {
        "Premier-League": "/en/comps/9/2024-2025/2024-2025-Premier-League-Stats",
        "La-Liga": "/en/comps/12/2024-2025/2024-2025-La-Liga-Stats",
        "Serie-A": "/en/comps/11/2024-2025/2024-2025-Serie-A-Stats",
        "Bundesliga": "/en/comps/20/2024-2025/2024-2025-Bundesliga-Stats",
        "Ligue-1": "/en/comps/13/2024-2025/2024-2025-Ligue-1-Stats"
    }

    # League IDs for dynamic URL construction
    LEAGUE_IDS = {
        "Premier-League": 9,
        "La-Liga": 12,
        "Serie-A": 11,
        "Bundesliga": 20,
        "Ligue-1": 13
    }

    # Rate limiting
    REQUEST_DELAY = 3  # seconds between requests to be respectful

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize FBRef collector.

        Args:
            cache_dir: Directory to cache scraped data
        """
        self.cache_dir = cache_dir or Path("data/raw/fbref_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; PassPredictionBot/1.0)'
        })

    def collect_season_data(
        self,
        season: str = "2024-2025",
        leagues: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Collect player match data for a season.

        Args:
            season: Season in YYYY-YYYY format
            leagues: List of leagues to collect. If None, collects all Big 5

        Returns:
            DataFrame with player match statistics
        """
        leagues = leagues or list(self.LEAGUE_URLS.keys())
        all_data = []

        for league in leagues:
            logger.info(f"Collecting {league} data for {season}")

            # Check cache first
            cache_file = self.cache_dir / f"{league}_{season}.parquet"
            if cache_file.exists():
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < timedelta(days=7):
                    logger.info(f"Using cached data for {league} {season}")
                    league_data = pd.read_parquet(cache_file)
                    all_data.append(league_data)
                    continue

            # Scrape league data
            league_data = self._scrape_league_season(league, season)
            if not league_data.empty:
                league_data['league'] = league
                league_data['season'] = season

                # Cache the data
                league_data.to_parquet(cache_file, index=False)
                all_data.append(league_data)

            time.sleep(self.REQUEST_DELAY)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def _scrape_league_season(self, league: str, season: str) -> pd.DataFrame:
        """Scrape all matches for a league season.

        Args:
            league: League name
            season: Season string

        Returns:
            DataFrame with player match data
        """
        # Build season-specific URL
        league_id = self.LEAGUE_IDS.get(league)
        if not league_id:
            logger.error(f"Unknown league: {league}")
            return pd.DataFrame()

        # Construct URL based on season
        league_url = f"/en/comps/{league_id}/{season}/{season}-{league.replace('-', '-')}-Stats"
        full_url = self.BASE_URL + league_url
        logger.debug(f"Fetching URL: {full_url}")

        try:
            response = self.session.get(full_url)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error fetching {league}: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find matches/fixtures table
        matches_data = self._extract_matches(soup, league)

        # For each match, get player statistics
        all_players_data = []
        for match_id, match_info in matches_data.items():
            player_data = self._scrape_match_players(match_id)
            if player_data:
                for player in player_data:
                    player.update(match_info)  # Add match context
                all_players_data.extend(player_data)

            time.sleep(self.REQUEST_DELAY)

            # Limit for testing (remove in production)
            if len(all_players_data) > 100:
                break

        return pd.DataFrame(all_players_data)

    def _extract_matches(self, soup: BeautifulSoup, league: str) -> Dict:
        """Extract match information from league page.

        Args:
            soup: BeautifulSoup of league page
            league: League name

        Returns:
            Dictionary mapping match_id to match info
        """
        matches = {}

        # Find the scores & fixtures table - try multiple possible table IDs
        scores_table = None
        for table_id in ['scores', 'sched', 'fixtures_results', 'matchlogs']:
            scores_table = soup.find('table', {'id': table_id})
            if scores_table:
                logger.debug(f"Found table with id: {table_id}")
                break

        # Also try finding by class
        if not scores_table:
            scores_table = soup.find('table', class_='stats_table')
            if scores_table:
                logger.debug("Found table with class: stats_table")

        if not scores_table:
            # Log the page structure to help debug
            logger.warning(f"No scores table found for {league}")
            tables = soup.find_all('table')
            logger.debug(f"Found {len(tables)} tables on page")
            for table in tables[:3]:  # Log first 3 tables
                table_id = table.get('id', 'no-id')
                table_class = table.get('class', ['no-class'])
                logger.debug(f"Table: id={table_id}, class={table_class}")
            return matches

        rows = scores_table.find_all('tr')
        for row in rows:
            # Skip header rows
            if row.find('th'):
                continue

            # Extract match data
            date_cell = row.find('td', {'data-stat': 'date'})
            home_team_cell = row.find('td', {'data-stat': 'home_team'})
            away_team_cell = row.find('td', {'data-stat': 'away_team'})
            score_cell = row.find('td', {'data-stat': 'score'})

            if date_cell and home_team_cell and away_team_cell:
                match_link = row.find('a', string='Match Report')
                if match_link:
                    match_id = match_link['href'].split('/')[-1]

                    matches[match_id] = {
                        'date': date_cell.text.strip(),
                        'home_team': home_team_cell.text.strip(),
                        'away_team': away_team_cell.text.strip(),
                        'score': score_cell.text.strip() if score_cell else None
                    }

        return matches

    def _scrape_match_players(self, match_id: str) -> List[Dict]:
        """Scrape player statistics from a match.

        Args:
            match_id: FBRef match ID

        Returns:
            List of player statistics dictionaries
        """
        url = f"{self.BASE_URL}/en/matches/{match_id}"

        try:
            response = self.session.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error fetching match {match_id}: {e}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        players_data = []

        # Find player stats tables (both teams)
        stats_tables = soup.find_all('table', {'class': 'stats_table'})

        for table in stats_tables:
            # Check if this is a relevant table (passing, summary, etc.)
            caption = table.find('caption')
            if not caption or 'Passing' not in caption.text:
                continue

            rows = table.find_all('tr')
            for row in rows:
                player_cell = row.find('th', {'data-stat': 'player'})
                if not player_cell:
                    continue

                player_data = {
                    'player_name': player_cell.text.strip(),
                    'match_id': match_id
                }

                # Extract relevant stats
                stat_mapping = {
                    'minutes': 'minutes_played',
                    'passes': 'passes_attempted',
                    'passes_completed': 'passes_completed',
                    'passing_accuracy': 'pass_accuracy',
                    'xg': 'xG',
                    'xa': 'xA',
                    'position': 'position',
                    'touches': 'touches',
                    'progressive_passes': 'progressive_passes'
                }

                for stat_name, field_name in stat_mapping.items():
                    stat_cell = row.find('td', {'data-stat': stat_name})
                    if stat_cell:
                        value = stat_cell.text.strip()
                        # Convert numeric values
                        try:
                            if field_name in ['minutes_played', 'passes_attempted',
                                            'passes_completed', 'touches',
                                            'progressive_passes']:
                                value = int(value) if value else 0
                            elif field_name in ['pass_accuracy', 'xG', 'xA']:
                                value = float(value) if value else 0.0
                        except (ValueError, TypeError):
                            pass

                        player_data[field_name] = value

                if player_data.get('minutes_played', 0) > 0:
                    players_data.append(player_data)

        return players_data

    def collect_recent_matches(self, days_back: int = 30) -> pd.DataFrame:
        """Collect matches from the last N days.

        Args:
            days_back: Number of days to look back

        Returns:
            DataFrame with recent player match data
        """
        # For current season, this would filter by date
        # For now, just get current season data
        return self.collect_season_data("2024-2025")

    def get_upcoming_fixtures(self, days_ahead: int = 7) -> pd.DataFrame:
        """Get upcoming fixtures for prediction.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            DataFrame with upcoming fixtures and expected lineups
        """
        fixtures = []

        for league, league_url in self.LEAGUE_URLS.items():
            full_url = self.BASE_URL + league_url

            try:
                response = self.session.get(full_url)
                response.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"Error fetching fixtures for {league}: {e}")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find upcoming matches
            scores_table = soup.find('table', {'id': 'scores'})
            if scores_table:
                rows = scores_table.find_all('tr')
                for row in rows:
                    date_cell = row.find('td', {'data-stat': 'date'})
                    score_cell = row.find('td', {'data-stat': 'score'})

                    # If no score, it's upcoming
                    if date_cell and (not score_cell or not score_cell.text.strip()):
                        home_team = row.find('td', {'data-stat': 'home_team'})
                        away_team = row.find('td', {'data-stat': 'away_team'})

                        if home_team and away_team:
                            fixtures.append({
                                'date': date_cell.text.strip(),
                                'league': league,
                                'home_team': home_team.text.strip(),
                                'away_team': away_team.text.strip()
                            })

            time.sleep(self.REQUEST_DELAY)

        return pd.DataFrame(fixtures)


def test_collector():
    """Test the FBRef collector."""
    collector = FBRefCollector()

    # Test collecting recent data
    logger.info("Testing FBRef collector...")

    # Get a small sample
    data = collector.collect_season_data(
        season="2024-2025",
        leagues=["Premier-League"]
    )

    if not data.empty:
        logger.info(f"Collected {len(data)} player-match records")
        logger.info(f"Columns: {data.columns.tolist()}")
        logger.info(f"Sample:\n{data.head()}")

        # Save sample
        data.to_csv("data/raw/fbref_sample.csv", index=False)
        logger.info("Sample saved to data/raw/fbref_sample.csv")
    else:
        logger.warning("No data collected")

    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_collector()
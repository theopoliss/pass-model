"""
API-Football collector for recent match data.
Free tier: 100 requests/day, includes player statistics.

Get your free API key at: https://www.api-football.com/
"""

import requests
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIFootballCollector:
    """Collect player match data from API-Football."""

    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

    # League IDs in API-Football
    LEAGUES = {
        'Premier-League': 39,
        'La-Liga': 140,
        'Serie-A': 135,
        'Bundesliga': 78,
        'Ligue-1': 61
    }

    def __init__(self, api_key=None):
        """Initialize with API key.

        Get free key at: https://rapidapi.com/api-sports/api/api-football
        """
        self.api_key = api_key or "YOUR_API_KEY_HERE"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }

        self.output_dir = Path("data/raw/api_football")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_connection(self):
        """Test API connection."""
        url = f"{self.BASE_URL}/status"

        try:
            response = requests.get(url, headers=self.headers)
            data = response.json()

            if response.status_code == 200:
                logger.info("✓ API connection successful")
                if 'response' in data:
                    logger.info(f"Account type: {data['response'].get('account', {}).get('type', 'Unknown')}")
                    logger.info(f"Requests remaining today: {data['response'].get('requests', {}).get('limit_day', 'Unknown')}")
                return True
            else:
                logger.error(f"API error: {data.get('message', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def get_recent_fixtures(self, league='Premier-League', last_days=30):
        """Get recent fixtures for a league."""
        league_id = self.LEAGUES.get(league)
        if not league_id:
            logger.error(f"Unknown league: {league}")
            return []

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=last_days)

        url = f"{self.BASE_URL}/fixtures"
        params = {
            "league": league_id,
            "season": 2024,  # Current season
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d")
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()

            if response.status_code == 200 and 'response' in data:
                fixtures = data['response']
                logger.info(f"Found {len(fixtures)} fixtures")
                return fixtures
            else:
                logger.error(f"Failed to get fixtures: {data.get('message', 'Unknown error')}")
                return []

        except Exception as e:
            logger.error(f"Error getting fixtures: {e}")
            return []

    def get_match_statistics(self, fixture_id):
        """Get player statistics for a match."""
        url = f"{self.BASE_URL}/fixtures/players"
        params = {"fixture": fixture_id}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()

            if response.status_code == 200 and 'response' in data:
                return data['response']
            else:
                logger.error(f"Failed to get match stats: {data.get('message', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error getting match stats: {e}")
            return None

    def process_match_data(self, match_stats):
        """Convert API response to our format."""
        player_data = []

        for team_data in match_stats:
            team_name = team_data['team']['name']

            for player_info in team_data['players']:
                player = player_info['player']
                stats = player_info['statistics'][0] if player_info['statistics'] else {}

                # Extract relevant data
                player_dict = {
                    'player_name': player['name'],
                    'team': team_name,
                    'position': stats.get('games', {}).get('position', ''),
                    'minutes_played': stats.get('games', {}).get('minutes', 0),

                    # Passing stats
                    'passes_attempted': stats.get('passes', {}).get('total', 0),
                    'passes_completed': stats.get('passes', {}).get('accuracy', 0),
                    'pass_accuracy': self._calculate_accuracy(
                        stats.get('passes', {}).get('accuracy', 0),
                        stats.get('passes', {}).get('total', 0)
                    ),
                    'key_passes': stats.get('passes', {}).get('key', 0),

                    # Other stats
                    'shots': stats.get('shots', {}).get('total', 0),
                    'goals': stats.get('goals', {}).get('total', 0),
                    'assists': stats.get('goals', {}).get('assists', 0),
                    'touches': stats.get('touches', {}).get('total', 0),
                    'dribbles_attempted': stats.get('dribbles', {}).get('attempts', 0),
                    'dribbles_successful': stats.get('dribbles', {}).get('success', 0)
                }

                player_data.append(player_dict)

        return player_data

    def _calculate_accuracy(self, completed, attempted):
        """Calculate pass accuracy percentage."""
        if attempted > 0:
            return (completed / attempted) * 100
        return 0

    def collect_recent_data(self, leagues=None, last_days=30, max_matches=10):
        """Collect recent match data for specified leagues."""
        leagues = leagues or list(self.LEAGUES.keys())
        all_data = []

        for league in leagues:
            logger.info(f"\nCollecting {league} data...")

            # Get recent fixtures
            fixtures = self.get_recent_fixtures(league, last_days)

            if not fixtures:
                logger.warning(f"No fixtures found for {league}")
                continue

            # Limit matches to avoid hitting API limits
            fixtures = fixtures[:max_matches]

            # Get player stats for each match
            for i, fixture in enumerate(fixtures):
                fixture_id = fixture['fixture']['id']
                home_team = fixture['teams']['home']['name']
                away_team = fixture['teams']['away']['name']
                date = fixture['fixture']['date']

                logger.info(f"  [{i+1}/{len(fixtures)}] {home_team} vs {away_team}")

                # Get match statistics
                match_stats = self.get_match_statistics(fixture_id)

                if match_stats:
                    player_data = self.process_match_data(match_stats)

                    # Add match metadata
                    for player in player_data:
                        player['fixture_id'] = fixture_id
                        player['date'] = date
                        player['home_team'] = home_team
                        player['away_team'] = away_team
                        player['league'] = league

                    all_data.extend(player_data)

                # Rate limiting (free tier)
                time.sleep(1)

        if all_data:
            df = pd.DataFrame(all_data)

            # Save data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"api_football_{timestamp}.parquet"
            df.to_parquet(output_file, index=False)

            csv_file = self.output_dir / f"api_football_{timestamp}.csv"
            df.to_csv(csv_file, index=False)

            logger.info(f"\n✓ Collected {len(df)} player records")
            logger.info(f"Data saved to: {output_file}")

            return df

        return pd.DataFrame()


def main():
    """Main function."""
    print("\n" + "="*60)
    print("API-FOOTBALL DATA COLLECTOR")
    print("="*60)

    print("\n⚠️  IMPORTANT: You need an API key from API-Football")
    print("Get your free key at: https://rapidapi.com/api-sports/api/api-football")
    print("Free tier: 100 requests/day\n")

    api_key = input("Enter your API key (or press Enter to use mock data): ").strip()

    if not api_key:
        print("\n📊 Generating mock data for testing...")
        # Generate mock data similar to API structure
        mock_data = []
        for i in range(100):
            mock_data.append({
                'player_name': f"Player_{i}",
                'team': f"Team_{i % 20}",
                'position': ['GK', 'DF', 'MF', 'FW'][i % 4],
                'minutes_played': 70 + (i % 20),
                'passes_attempted': 20 + (i % 30),
                'passes_completed': 15 + (i % 25),
                'pass_accuracy': 70 + (i % 20),
                'goals': i % 3,
                'assists': i % 2,
                'league': 'Premier-League',
                'date': datetime.now().isoformat()
            })

        df = pd.DataFrame(mock_data)
        print(f"\n✓ Generated {len(df)} mock records")
        print("\nSample data:")
        print(df.head())

        # Save mock data
        output_dir = Path("data/raw/api_football")
        output_dir.mkdir(parents=True, exist_ok=True)
        mock_file = output_dir / f"mock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(mock_file, index=False)
        print(f"\nMock data saved to: {mock_file}")

    else:
        collector = APIFootballCollector(api_key)

        # Test connection
        if collector.test_connection():
            print("\n✓ API connection successful!")

            # Collect recent data (limited to avoid hitting limits)
            print("\nCollecting recent match data...")
            print("(Limited to 10 matches to stay within free tier)\n")

            data = collector.collect_recent_data(
                leagues=['Premier-League'],  # Start with one league
                last_days=7,  # Last week
                max_matches=5  # Stay well within limits
            )

            if not data.empty:
                print("\n📊 Sample of collected data:")
                print(data[['player_name', 'position', 'minutes_played', 'passes_attempted']].head())
                print(f"\n✓ Ready to train model with this data!")
        else:
            print("\n✗ API connection failed. Please check your API key.")


if __name__ == "__main__":
    main()
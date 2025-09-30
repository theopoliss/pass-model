"""Generate realistic synthetic FBRef-style data for testing the pipeline.

This creates data that mimics real FBRef structure while the scraper is being fixed.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Real team names from top 5 leagues
TEAMS = {
    "Premier-League": [
        "Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United",
        "Tottenham", "Newcastle", "Brighton", "Aston Villa", "West Ham",
        "Fulham", "Brentford", "Crystal Palace", "Wolves", "Everton",
        "Leicester", "Nottingham Forest", "Bournemouth", "Luton", "Burnley"
    ],
    "La-Liga": [
        "Real Madrid", "Barcelona", "Atletico Madrid", "Real Sociedad", "Athletic Bilbao",
        "Villarreal", "Real Betis", "Valencia", "Sevilla", "Girona",
        "Rayo Vallecano", "Osasuna", "Getafe", "Celta Vigo", "Cadiz",
        "Mallorca", "Las Palmas", "Alaves", "Granada", "Almeria"
    ],
    "Serie-A": [
        "Inter Milan", "AC Milan", "Juventus", "Napoli", "Roma",
        "Lazio", "Atalanta", "Fiorentina", "Bologna", "Torino",
        "Monza", "Udinese", "Sassuolo", "Empoli", "Salernitana",
        "Lecce", "Verona", "Cagliari", "Frosinone", "Genoa"
    ],
    "Bundesliga": [
        "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Union Berlin",
        "Freiburg", "Eintracht Frankfurt", "Wolfsburg", "Mainz", "Borussia Monchengladbach",
        "Hoffenheim", "Werder Bremen", "Augsburg", "Stuttgart", "Bochum",
        "Heidenheim", "Darmstadt", "FC Koln"
    ],
    "Ligue-1": [
        "PSG", "Monaco", "Marseille", "Lille", "Nice",
        "Lyon", "Lens", "Rennes", "Reims", "Toulouse",
        "Montpellier", "Strasbourg", "Brest", "Nantes", "Le Havre",
        "Metz", "Lorient", "Clermont"
    ]
}

# Common player names by position
PLAYER_NAMES = {
    "GK": ["Alisson", "Ederson", "Ter Stegen", "Courtois", "Donnarumma", "Neuer", "Oblak"],
    "DF": ["Van Dijk", "Ramos", "Marquinhos", "Dias", "Stones", "Saliba", "Gabriel", "White",
           "Walker", "Robertson", "Cancelo", "Davies", "Hakimi", "Theo Hernandez"],
    "MF": ["De Bruyne", "Modric", "Kroos", "Bellingham", "Rice", "Rodri", "Casemiro",
           "Bruno Fernandes", "Odegaard", "Pedri", "Gavi", "Valverde", "Camavinga"],
    "FW": ["Haaland", "Mbappe", "Vinicius Jr", "Salah", "Saka", "Son", "Kane", "Lewandowski",
           "Osimhen", "Lautaro Martinez", "Griezmann", "Rashford"]
}

def generate_player_match_data(n_seasons=5, matches_per_season=380):
    """Generate realistic player match data.

    Args:
        n_seasons: Number of seasons to generate
        matches_per_season: Matches per league season

    Returns:
        DataFrame with synthetic player match data
    """
    all_data = []

    seasons = [
        "2019-2020", "2020-2021", "2021-2022",
        "2022-2023", "2023-2024", "2024-2025"
    ][:n_seasons]

    for season in seasons:
        season_start = datetime.strptime(f"{season[:4]}-08-15", "%Y-%m-%d")

        for league, teams in TEAMS.items():
            logger.info(f"Generating {league} {season}")

            # Generate matches (each team plays each other twice)
            match_id = 0
            for round_num in range(2):  # Home and away
                for i, home_team in enumerate(teams):
                    for away_team in teams[i+1:]:
                        if round_num == 1:  # Swap for return fixture
                            home_team, away_team = away_team, home_team

                        match_id += 1
                        match_date = season_start + timedelta(days=match_id * 3)

                        # Generate player data for this match
                        for team, is_home in [(home_team, True), (away_team, False)]:
                            # Generate 11 starters + 3 subs
                            positions = ["GK"] + ["DF"] * 4 + ["MF"] * 3 + ["FW"] * 3 + ["MF", "FW", "DF"]

                            for pos_idx, position in enumerate(positions):
                                # Create unique player name
                                player_pool = PLAYER_NAMES.get(position, PLAYER_NAMES["MF"])
                                player_name = f"{random.choice(player_pool)}_{team[:3]}_{pos_idx}"

                                # Determine minutes played
                                if pos_idx < 11:  # Starter
                                    minutes = random.gauss(75, 15)
                                    minutes = max(45, min(90, minutes))
                                else:  # Substitute
                                    minutes = random.gauss(20, 10)
                                    minutes = max(0, min(45, minutes))

                                if minutes < 10:
                                    continue  # Skip players with very few minutes

                                # Generate realistic pass numbers based on position
                                if position == "GK":
                                    base_passes = random.gauss(25, 8)
                                elif position == "DF":
                                    base_passes = random.gauss(45, 12)
                                elif position == "MF":
                                    base_passes = random.gauss(40, 15)
                                else:  # FW
                                    base_passes = random.gauss(25, 10)

                                # Adjust for minutes played
                                passes_attempted = int(base_passes * (minutes / 90))
                                passes_attempted = max(5, passes_attempted)

                                # Calculate other metrics
                                pass_accuracy = random.gauss(82, 8)
                                pass_accuracy = max(60, min(95, pass_accuracy))
                                passes_completed = int(passes_attempted * pass_accuracy / 100)

                                # Advanced metrics
                                xG = random.expovariate(5) if position in ["FW", "MF"] else random.expovariate(20)
                                xA = random.expovariate(7) if position in ["MF", "FW"] else random.expovariate(25)

                                touches = int(passes_attempted * random.gauss(1.8, 0.3))
                                progressive_passes = int(passes_attempted * random.gauss(0.15, 0.05))
                                progressive_passes = max(0, progressive_passes)

                                all_data.append({
                                    'player_name': player_name,
                                    'match_id': f"{league}_{season}_{match_id}",
                                    'date': match_date,
                                    'position': position,
                                    'minutes_played': round(minutes, 0),
                                    'passes_attempted': passes_attempted,
                                    'passes_completed': passes_completed,
                                    'pass_accuracy': round(pass_accuracy, 1),
                                    'xG': round(xG, 3),
                                    'xA': round(xA, 3),
                                    'touches': touches,
                                    'progressive_passes': progressive_passes,
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'league': league,
                                    'season': season,
                                    'is_home': int(is_home)
                                })

                        # Limit matches for faster generation
                        if match_id >= matches_per_season // 10:  # Generate 10% of matches
                            break
                    if match_id >= matches_per_season // 10:
                        break

    df = pd.DataFrame(all_data)

    # Add some consistency to player stats across games
    # Players should have somewhat consistent performance
    player_factors = {}
    for player in df['player_name'].unique():
        player_factors[player] = random.gauss(1.0, 0.15)  # Performance multiplier

    df['player_factor'] = df['player_name'].map(player_factors)
    df['passes_attempted'] = (df['passes_attempted'] * df['player_factor']).astype(int)
    df['passes_attempted'] = df['passes_attempted'].clip(lower=5)
    df = df.drop('player_factor', axis=1)

    return df

def save_synthetic_data():
    """Generate and save synthetic FBRef data."""
    logger.info("Generating synthetic FBRef data...")

    # Generate data
    df = generate_player_match_data(n_seasons=5, matches_per_season=100)

    # Save to FBRef data directory
    output_dir = Path("data/raw/fbref")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parquet_file = output_dir / f"fbref_synthetic_{timestamp}.parquet"
    csv_file = output_dir / f"fbref_synthetic_{timestamp}.csv"

    df.to_parquet(parquet_file, index=False)
    df.to_csv(csv_file, index=False)

    # Also save in cache directory for seamless integration
    cache_dir = Path("data/raw/fbref_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save by league-season for cache compatibility
    for (league, season), group in df.groupby(['league', 'season']):
        cache_file = cache_dir / f"{league}_{season}.parquet"
        group.to_parquet(cache_file, index=False)

    logger.info(f"Generated {len(df):,} player-match records")
    logger.info(f"Unique players: {df['player_name'].nunique():,}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"\nData saved to:")
    logger.info(f"  - {parquet_file}")
    logger.info(f"  - {csv_file}")
    logger.info(f"  - Cache files in {cache_dir}")

    # Print sample
    print("\nSample data:")
    print(df.head())

    print("\nData statistics:")
    print(df[['passes_attempted', 'pass_accuracy', 'minutes_played', 'xG', 'xA']].describe())

    return df

if __name__ == "__main__":
    data = save_synthetic_data()
    print(f"\n✓ Synthetic data generated successfully!")
    print(f"You can now run: python train_fbref.py --use-cached")
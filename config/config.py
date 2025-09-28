"""Configuration management for the pass prediction model."""

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class ModelConfig(BaseModel):
    """Model configuration."""

    model_type: str = Field(default="poisson", description="Type of model to use")
    features: List[str] = Field(
        default=[
            # Position
            "position_encoded",

            # Match context
            "minutes_played",
            "is_home",

            # Team strength
            "team_strength_diff",
            "opponent_strength",

            # Player-specific features
            "player_career_avg_passes",
            "player_career_avg_passes_per90",
            "player_pass_consistency",
            "player_games_played",
            "player_career_completion_rate",

            # Player form
            "player_recent_passes_avg",
            "player_recent_completion_rate",
            "player_form_trend"
        ]
    )
    target: str = Field(default="passes_attempted")

    # Training parameters
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)
    cv_folds: int = Field(default=5, ge=3, le=10)

    # Poisson/NegBin specific
    use_exposure: bool = Field(default=True, description="Use minutes as exposure")
    overdispersion: bool = Field(default=False, description="Use Negative Binomial instead of Poisson")


class DataConfig(BaseModel):
    """Data configuration."""

    # StatsBomb specific
    competitions: List[Dict[str, int]] = Field(
        default=[
            # International Tournaments
            {"competition_id": 43, "season_id": 3},  # FIFA World Cup 2018
            {"competition_id": 55, "season_id": 43},  # UEFA Euro 2020
            {"competition_id": 72, "season_id": 30},  # Women's World Cup 2019

            # Champions League - multiple seasons for more data
            {"competition_id": 16, "season_id": 27},  # Champions League 2018/2019
            {"competition_id": 16, "season_id": 4},   # Champions League 2017/2018
            {"competition_id": 16, "season_id": 3},   # Champions League 2016/2017
            {"competition_id": 16, "season_id": 2},   # Champions League 2015/2016

            # League Data
            {"competition_id": 11, "season_id": 27},  # La Liga 2018/2019
            {"competition_id": 11, "season_id": 4},   # La Liga 2017/2018
            {"competition_id": 2, "season_id": 27},   # Premier League 2015/2016
        ]
    )

    # Data filtering
    min_minutes_played: int = Field(default=15, description="Minimum minutes to include player")
    exclude_goalkeepers: bool = Field(default=True)

    # Feature engineering
    rolling_window_games: int = Field(default=5, description="Games for rolling averages")

    # Database (optional)
    use_database: bool = Field(default=False)
    db_url: Optional[str] = Field(default=os.getenv("DATABASE_URL"))


class PositionMapping(BaseModel):
    """Position group mappings."""

    position_groups: Dict[str, List[str]] = Field(
        default={
            "GK": ["Goalkeeper"],
            "DEF": ["Center Back", "Left Back", "Right Back", "Left Wing Back", "Right Wing Back"],
            "MID": ["Center Defensive Midfield", "Center Midfield", "Center Attacking Midfield",
                    "Left Midfield", "Right Midfield"],
            "FWD": ["Left Wing", "Right Wing", "Center Forward", "Left Center Forward",
                    "Right Center Forward", "Secondary Striker"]
        }
    )

    def get_position_group(self, position: str) -> str:
        """Get position group for a given position."""
        for group, positions in self.position_groups.items():
            if position in positions:
                return group
        return "OTHER"


class Config(BaseModel):
    """Main configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    positions: PositionMapping = Field(default_factory=PositionMapping)

    # Logging
    log_level: str = Field(default="INFO")

    # Paths
    project_root: Path = Field(default=PROJECT_ROOT)
    data_dir: Path = Field(default=DATA_DIR)
    raw_data_dir: Path = Field(default=RAW_DATA_DIR)
    processed_data_dir: Path = Field(default=PROCESSED_DATA_DIR)
    model_dir: Path = Field(default=MODEL_DIR)


# Global config instance
config = Config()
"""StatsBomb data collector."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StatsBombCollector:
    """Collect and process StatsBomb open data."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the collector.

        Args:
            cache_dir: Directory to cache downloaded data.
        """
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def get_competitions(self) -> pd.DataFrame:
        """Get available competitions."""
        return sb.competitions()

    def validate_competition(self, competition_id: int, season_id: int) -> bool:
        """Validate if a competition/season exists.

        Args:
            competition_id: StatsBomb competition ID
            season_id: StatsBomb season ID

        Returns:
            True if competition exists, False otherwise
        """
        competitions = self.get_competitions()
        exists = ((competitions['competition_id'] == competition_id) &
                 (competitions['season_id'] == season_id)).any()
        return exists

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Get matches for a competition and season.

        Args:
            competition_id: StatsBomb competition ID
            season_id: StatsBomb season ID

        Returns:
            DataFrame with match information
        """
        return sb.matches(competition_id=competition_id, season_id=season_id)

    def get_events(self, match_id: int) -> pd.DataFrame:
        """Get events for a match.

        Args:
            match_id: StatsBomb match ID

        Returns:
            DataFrame with match events
        """
        return sb.events(match_id=match_id)

    def get_lineups(self, match_id: int) -> Dict:
        """Get lineups for a match.

        Args:
            match_id: StatsBomb match ID

        Returns:
            Dictionary with team lineups
        """
        return sb.lineups(match_id=match_id)

    def collect_competition_data(
        self, competition_id: int, season_id: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Collect all data for a competition season.

        Args:
            competition_id: StatsBomb competition ID
            season_id: StatsBomb season ID

        Returns:
            Tuple of (matches_df, events_df, lineups_dict)
        """
        logger.info(f"Collecting data for competition {competition_id}, season {season_id}")

        # Validate competition exists
        if not self.validate_competition(competition_id, season_id):
            raise ValueError(f"Competition {competition_id} season {season_id} not found in StatsBomb data")

        # Get matches
        matches = self.get_matches(competition_id, season_id)
        logger.info(f"Found {len(matches)} matches")

        # Collect events and lineups for each match
        all_events = []
        all_lineups = {}

        for _, match in tqdm(matches.iterrows(), total=len(matches), desc="Collecting matches"):
            match_id = match["match_id"]

            try:
                # Get events
                events = self.get_events(match_id)
                events["match_id"] = match_id
                events["competition_id"] = competition_id
                events["season_id"] = season_id
                all_events.append(events)

                # Get lineups
                lineups = self.get_lineups(match_id)
                all_lineups[match_id] = lineups

            except Exception as e:
                logger.warning(f"Error collecting match {match_id}: {e}")
                continue

        # Combine all events
        events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

        # Add game state to events
        if not events_df.empty:
            events_df = self._add_game_state_to_events(events_df)

        return matches, events_df, all_lineups

    def _add_game_state_to_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Add game state (score at time of event) to events.

        Args:
            events_df: DataFrame with match events

        Returns:
            DataFrame with game state columns added
        """
        events_df = events_df.sort_values(['match_id', 'minute', 'second']).copy()

        # Initialize score columns
        events_df['home_score_at_event'] = 0
        events_df['away_score_at_event'] = 0

        for match_id in events_df['match_id'].unique():
            match_events = events_df[events_df['match_id'] == match_id].copy()

            # Find goals
            goals = match_events[match_events['type'] == 'Shot']
            goals = goals[goals['shot_outcome'] == 'Goal'] if 'shot_outcome' in goals.columns else pd.DataFrame()

            if not goals.empty:
                home_score = 0
                away_score = 0

                for idx, event in match_events.iterrows():
                    # Update score if we've passed a goal
                    for _, goal in goals.iterrows():
                        if (goal['minute'] < event['minute'] or
                            (goal['minute'] == event['minute'] and goal['second'] <= event['second'])):

                            if goal['team'] == match_events.iloc[0]['team']:  # Assuming first event team is home
                                if goal['minute'] < event['minute'] or (goal['minute'] == event['minute'] and goal['second'] < event['second']):
                                    home_score = 1  # Simplified - would need actual team mapping
                            else:
                                if goal['minute'] < event['minute'] or (goal['minute'] == event['minute'] and goal['second'] < event['second']):
                                    away_score = 1

                    events_df.loc[idx, 'home_score_at_event'] = home_score
                    events_df.loc[idx, 'away_score_at_event'] = away_score

        # Add derived game state features
        events_df['score_differential'] = events_df['home_score_at_event'] - events_df['away_score_at_event']
        events_df['game_state'] = events_df['score_differential'].apply(
            lambda x: 'winning' if x > 0 else ('losing' if x < 0 else 'drawing')
        )

        return events_df

    def extract_pass_data(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Extract pass data from events.

        Args:
            events_df: DataFrame with match events

        Returns:
            DataFrame with pass information
        """
        # Filter for pass events
        passes = events_df[events_df["type"] == "Pass"].copy()

        # Extract relevant columns
        pass_columns = [
            "match_id",
            "team",
            "player",
            "position",
            "minute",
            "second",
            "location",
            "pass_end_location",
            "pass_outcome",
            "pass_length",
            "pass_angle",
            "pass_height",
            "pass_type"
        ]

        # Keep only existing columns
        existing_columns = [col for col in pass_columns if col in passes.columns]
        passes_clean = passes[existing_columns].copy()

        # Add success indicator
        if "pass_outcome" in passes_clean.columns:
            passes_clean["pass_success"] = passes_clean["pass_outcome"].isna()
        else:
            passes_clean["pass_success"] = True  # Assume success if no outcome

        return passes_clean

    def _classify_pitch_zone(self, x_coord: float, pitch_length: float = 120) -> str:
        """Classify pitch zone based on x-coordinate.

        Args:
            x_coord: X-coordinate of the pass (0-120 typically)
            pitch_length: Length of pitch in StatsBomb coords

        Returns:
            Zone classification: 'defensive', 'middle', 'attacking'
        """
        if x_coord < pitch_length / 3:
            return 'defensive'
        elif x_coord < 2 * pitch_length / 3:
            return 'middle'
        else:
            return 'attacking'

    def _is_progressive_pass(self, start_x: float, end_x: float,
                            start_y: float, end_y: float,
                            attacking_direction: bool = True) -> bool:
        """Determine if a pass is progressive (advances significantly toward goal).

        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            attacking_direction: True if attacking left-to-right

        Returns:
            True if pass advances >25% toward goal
        """
        if attacking_direction:
            progress = end_x - start_x
        else:
            progress = start_x - end_x

        # Progressive if advances >30 units toward goal (25% of pitch)
        return progress > 30

    def _classify_pass_direction(self, angle: float) -> str:
        """Classify pass direction based on angle.

        Args:
            angle: Pass angle in radians

        Returns:
            Direction: 'forward', 'sideways', 'backward'
        """
        import math
        angle_deg = math.degrees(angle) if angle else 0

        # Normalize angle to -180 to 180
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg < -180:
            angle_deg += 360

        # Forward: -45 to 45 degrees
        # Backward: -135 to -180 or 135 to 180
        # Sideways: everything else
        if -45 <= angle_deg <= 45:
            return 'forward'
        elif angle_deg < -135 or angle_deg > 135:
            return 'backward'
        else:
            return 'sideways'

    def extract_formations(self, events_df: pd.DataFrame) -> Dict[str, Dict]:
        """Extract formation data from Starting XI events.

        Args:
            events_df: DataFrame with match events

        Returns:
            Dictionary with match_id -> {team -> formation} mapping
        """
        formations = {}

        # Get Starting XI events which contain tactics/formation info
        starting_xi_events = events_df[events_df["type"] == "Starting XI"].copy()

        for _, event in starting_xi_events.iterrows():
            match_id = event["match_id"]
            team = event["team"]

            if match_id not in formations:
                formations[match_id] = {}

            # Extract formation from tactics if available
            if pd.notna(event.get("tactics")) and isinstance(event["tactics"], dict):
                formation = event["tactics"].get("formation")
                if formation:
                    formations[match_id][team] = str(formation)

        return formations

    def aggregate_player_passes(
        self, events_df: pd.DataFrame, matches_df: pd.DataFrame, lineups: Dict
    ) -> pd.DataFrame:
        """Aggregate passes by player and match with enhanced features.

        Args:
            events_df: DataFrame with match events
            matches_df: DataFrame with match information
            lineups: Dictionary with team lineups

        Returns:
            DataFrame with player-match level pass statistics
        """
        # Extract formations first
        formations = self.extract_formations(events_df)

        # Filter for pass events
        passes = events_df[events_df["type"] == "Pass"].copy()

        # Extract location features if available
        if "location" in passes.columns:
            # Parse location arrays [x, y]
            passes["start_x"] = passes["location"].apply(lambda x: x[0] if isinstance(x, list) and len(x) >= 2 else None)
            passes["start_y"] = passes["location"].apply(lambda x: x[1] if isinstance(x, list) and len(x) >= 2 else None)

            # Classify starting zones
            passes["start_zone"] = passes["start_x"].apply(
                lambda x: self._classify_pitch_zone(x) if x is not None else None
            )

        if "pass_end_location" in passes.columns:
            # Parse end location
            passes["end_x"] = passes["pass_end_location"].apply(lambda x: x[0] if isinstance(x, list) and len(x) >= 2 else None)
            passes["end_y"] = passes["pass_end_location"].apply(lambda x: x[1] if isinstance(x, list) and len(x) >= 2 else None)

            # Classify ending zones
            passes["end_zone"] = passes["end_x"].apply(
                lambda x: self._classify_pitch_zone(x) if x is not None else None
            )

            # Identify progressive passes
            passes["is_progressive"] = passes.apply(
                lambda x: self._is_progressive_pass(
                    x["start_x"], x["end_x"], x["start_y"], x["end_y"]
                ) if all(v is not None for v in [x.get("start_x"), x.get("end_x"),
                                                   x.get("start_y"), x.get("end_y")]) else False,
                axis=1
            )

        # Classify pass direction if angle available
        if "pass_angle" in passes.columns:
            passes["pass_direction"] = passes["pass_angle"].apply(
                lambda x: self._classify_pass_direction(x) if x is not None else None
            )

        # Check for passes under pressure
        if "under_pressure" in passes.columns:
            passes["under_pressure"] = passes["under_pressure"].fillna(False)

        # Add game state features if available
        if "game_state" in passes.columns:
            passes["passes_while_winning"] = passes["game_state"] == "winning"
            passes["passes_while_losing"] = passes["game_state"] == "losing"
            passes["passes_while_drawing"] = passes["game_state"] == "drawing"

        # Group by match and player with enhanced aggregations
        player_passes = (
            passes.groupby(["match_id", "team", "player", "position"])
            .agg(
                passes_attempted=("type", "count"),
                passes_completed=(
                    "pass_outcome",
                    lambda x: x.isna().sum() if "pass_outcome" in passes.columns else len(x)
                ),
                avg_pass_length=("pass_length", "mean") if "pass_length" in passes.columns else ("type", lambda x: None),
                first_pass_minute=("minute", "min"),
                last_pass_minute=("minute", "max"),
                # Zone-based features
                passes_from_defensive=("start_zone", lambda x: (x == "defensive").sum()) if "start_zone" in passes.columns else ("type", lambda x: None),
                passes_from_middle=("start_zone", lambda x: (x == "middle").sum()) if "start_zone" in passes.columns else ("type", lambda x: None),
                passes_from_attacking=("start_zone", lambda x: (x == "attacking").sum()) if "start_zone" in passes.columns else ("type", lambda x: None),
                passes_to_attacking=("end_zone", lambda x: (x == "attacking").sum()) if "end_zone" in passes.columns else ("type", lambda x: None),
                # Progressive passes
                progressive_passes=("is_progressive", "sum") if "is_progressive" in passes.columns else ("type", lambda x: None),
                # Direction features
                forward_passes=("pass_direction", lambda x: (x == "forward").sum()) if "pass_direction" in passes.columns else ("type", lambda x: None),
                backward_passes=("pass_direction", lambda x: (x == "backward").sum()) if "pass_direction" in passes.columns else ("type", lambda x: None),
                sideways_passes=("pass_direction", lambda x: (x == "sideways").sum()) if "pass_direction" in passes.columns else ("type", lambda x: None),
                # Pressure
                passes_under_pressure=("under_pressure", "sum") if "under_pressure" in passes.columns else ("type", lambda x: None),
                # Pass types if available
                through_balls=("pass_type", lambda x: (x == "Through Ball").sum()) if "pass_type" in passes.columns else ("type", lambda x: None),
                long_balls=("pass_height", lambda x: (x == "High Pass").sum()) if "pass_height" in passes.columns else ("type", lambda x: None),
                # Game state features
                passes_while_winning=("passes_while_winning", "sum") if "passes_while_winning" in passes.columns else ("type", lambda x: None),
                passes_while_losing=("passes_while_losing", "sum") if "passes_while_losing" in passes.columns else ("type", lambda x: None),
                passes_while_drawing=("passes_while_drawing", "sum") if "passes_while_drawing" in passes.columns else ("type", lambda x: None)
            )
            .reset_index()
        )

        # Calculate minutes played (approximation from first to last action)
        player_passes["minutes_played"] = (
            player_passes["last_pass_minute"] - player_passes["first_pass_minute"]
        ).clip(lower=1)  # At least 1 minute if they made a pass

        # Add match information
        player_passes = player_passes.merge(
            matches_df[["match_id", "home_team", "away_team", "home_score", "away_score",
                       "match_date", "competition_stage"]],
            on="match_id",
            how="left"
        )

        # Add home/away indicator
        player_passes["is_home"] = player_passes["team"] == player_passes["home_team"]

        # Add result indicator
        player_passes["team_goals"] = player_passes.apply(
            lambda x: x["home_score"] if x["is_home"] else x["away_score"], axis=1
        )
        player_passes["opponent_goals"] = player_passes.apply(
            lambda x: x["away_score"] if x["is_home"] else x["home_score"], axis=1
        )
        player_passes["goal_difference"] = player_passes["team_goals"] - player_passes["opponent_goals"]

        # Add formation data
        for idx, row in player_passes.iterrows():
            match_id = row["match_id"]
            team = row["team"]
            opponent_team = row["away_team"] if row["is_home"] else row["home_team"]

            # Get formations for this match
            if match_id in formations:
                match_formations = formations[match_id]

                # Team formation
                player_passes.at[idx, "team_formation"] = match_formations.get(team, None)

                # Opponent formation
                player_passes.at[idx, "opponent_formation"] = match_formations.get(opponent_team, None)

        return player_passes

    def save_data(self, data: pd.DataFrame, filename: str, output_dir: Path) -> None:
        """Save data to file.

        Args:
            data: DataFrame to save
            filename: Output filename
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        if filename.endswith(".csv"):
            data.to_csv(output_path, index=False)
        elif filename.endswith(".parquet"):
            data.to_parquet(output_path, index=False)
        else:
            data.to_pickle(output_path)

        logger.info(f"Saved data to {output_path}")
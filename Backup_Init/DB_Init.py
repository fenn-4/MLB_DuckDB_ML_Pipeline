"""
MLB Statcast Data Pipeline Initialization Script

- Collects player and Statcast data for MLB seasons 2022-2025
- Stores data in DuckDB
- Updates player info and advanced metrics after data collection
- Calculates rolling statistics for players
"""

import requests
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from io import StringIO
import time
from Player_Tables_Alter import update_player_info
from Statcast_Table_Alter import update_advanced_metrics

# --- Configuration ---
KEEP_COLUMNS = [
    'game_pk', 'game_date', 'pitch_type', 'release_speed', 'release_pos_x', 'release_pos_z',
    'batter', 'pitcher', 'events', 'description', 'zone', 'stand', 'p_throws',
    'home_team', 'away_team', 'hit_location', 'bb_type', 'balls', 'strikes',
    'game_year', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'outs_when_up', 'inning',
    'inning_topbot', 'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'sz_top', 'sz_bot', 'hit_distance_sc', 'launch_speed', 'launch_angle',
    'effective_speed', 'release_spin_rate', 'release_extension', 'release_pos_y',
    'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle',
    'estimated_slg_using_speedangle', 'pitch_name', 'post_away_score',
    'post_home_score', 'if_fielding_alignment', 'of_fielding_alignment',
    'spin_axis', 'delta_home_win_exp', 'delta_run_exp', 'bat_speed',
    'swing_length', 'home_win_exp', 'api_break_z_with_gravity',
    'api_break_x_batter_in', 'arm_angle', 'attack_angle', 'attack_direction',
    'swing_path_tilt', 'intercept_ball_minus_batter_pos_x_inches',
    'intercept_ball_minus_batter_pos_y_inches'
]

SEASON_RANGES = {
    2022: {'start': '2022-04-07', 'end': '2022-10-05'},
    2023: {'start': '2023-03-30', 'end': '2023-10-01'},
    2024: {'start': '2024-03-28', 'end': '2024-09-29'},
    2025: {'start': '2025-03-27', 'end': '2025-09-28'}
}

# --- Database Manager ---
class DatabaseManager:
    def __init__(self, db_path='mlb_statcast.db', schema_path='Schema_Init.sql'):
        self.conn = duckdb.connect(db_path)
        with open(schema_path, 'r') as file:
            self.conn.execute(file.read())

    def close(self):
        self.conn.close()

    def get_latest_date(self):
        result = self.conn.execute("SELECT MAX(game_date) FROM statcast_data").fetchone()
        return result[0] if result[0] is not None else None

    def insert_players(self, player_type, df):
        table_name = f"{player_type}s"
        self.conn.execute(f"""
            INSERT INTO {table_name} (player_id, player_name)
            SELECT player_id, player_name
            FROM df
            WHERE player_id NOT IN (SELECT player_id FROM {table_name})
        """)

    def insert_statcast_data(self, df):
        schema_cols = [col[1] for col in self.conn.execute('PRAGMA table_info(statcast_data)').fetchall()]
        if KEEP_COLUMNS:
            keep = [col for col in KEEP_COLUMNS if col in df.columns and col in schema_cols]
        else:
            keep = [col for col in schema_cols if col in df.columns]
        
        # Create an explicit copy to avoid SettingWithCopyWarning
        df_filtered = df[keep].copy()
        
        # Round float columns using .loc for safer assignment
        float_cols = df_filtered.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            df_filtered.loc[:, float_cols] = df_filtered.loc[:, float_cols].round(4)
        
        # Reindex to match schema
        df_filtered = df_filtered.reindex(columns=schema_cols)
        self.conn.execute("INSERT INTO statcast_data SELECT * FROM df_filtered")

    def count_statcast_rows(self):
        return self.conn.execute("SELECT COUNT(*) FROM statcast_data").fetchone()[0]

# --- Data Fetching Functions ---
def get_player_data(player_type):
    url = f"https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea=2025%7C2024%7C2023%7C2022%7C&hfSit=&player_type={player_type}&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&minors=false"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df[['player_id', 'player_name']].drop_duplicates()
        return None
    except Exception:
        return None

def get_statcast_data(date):
    year = datetime.strptime(date, '%Y-%m-%d').year
    url = f"https://baseballsavant.mlb.com/statcast_search/csv?hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea={year}%7C&hfSit=&player_type=batter&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={date}&game_date_lt={date}&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=is%5C.%5C.tracked%7C&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&type=details&all=true&minors=false"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        return None
    except Exception:
        return None

# --- Main Pipeline ---
def main():
    db = DatabaseManager()
    print("Fetching player data...")
    for player_type in ['pitcher', 'batter']:
        df = get_player_data(player_type)
        if df is not None and not df.empty:
            db.insert_players(player_type, df)
            print(f"Inserted {len(df)} {player_type}s")
        time.sleep(1)

    latest_date = db.get_latest_date()
    if latest_date is None:
        start_date = datetime.strptime(SEASON_RANGES[2022]['start'], '%Y-%m-%d')
    else:
        if isinstance(latest_date, datetime):
            start_date = latest_date + timedelta(days=1)
        else:
            start_date = datetime.combine(latest_date, datetime.min.time()) + timedelta(days=1)
    today = datetime.now().date()
    end_date = today - timedelta(days=1)
    dates_to_fetch = []
    for year in range(start_date.year, 2026):
        season_start = datetime.strptime(SEASON_RANGES[year]['start'], '%Y-%m-%d').date()
        season_end = datetime.strptime(SEASON_RANGES[year]['end'], '%Y-%m-%d').date()
        d = max(start_date.date(), season_start)
        while d <= min(season_end, end_date):
            dates_to_fetch.append(d.strftime('%Y-%m-%d'))
            d += timedelta(days=1)
    for date_str in dates_to_fetch:
        print(f"Fetching data for {date_str}...")
        df = get_statcast_data(date_str)
        if df is not None and not df.empty:
            db.insert_statcast_data(df)
            print(f"Inserted {len(df)} rows for {date_str}")
        else:
            print(f"No data for {date_str}")
        time.sleep(1)
    total_rows = db.count_statcast_rows()
    print(f"\nTotal rows in database: {total_rows}")
    db.close()
    print("Done.")

    # Post-processing: update player info and advanced metrics
    print("\nUpdating player info (p_throws and stand)...")
    update_player_info()
    print("\nCalculating advanced metrics (phi and estimated_ISO)...")
    update_advanced_metrics()

if __name__ == "__main__":
    main() 
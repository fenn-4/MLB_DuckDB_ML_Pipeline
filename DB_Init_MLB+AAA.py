"""Unified MLB and Triple-A Statcast Data Pipeline.

- Downloads MLB and Triple-A Statcast data from Baseball Savant
- Stores data in DuckDB tables: statcast_data (MLB), statcast_minor (Triple-A)
- Updates player info and advanced metrics
"""

import requests
import pandas as pd
import duckdb
import time
from datetime import datetime, timedelta
from io import StringIO
import subprocess

# === CONFIGURATION ===
DB_PATH = 'Full_DB/mlb_statcast.db'
SCHEMA_PATH = 'Helper_Queries/Schema_Init.sql'
REQUEST_DELAY = 1  # seconds between requests
MAX_YEAR = 2026
ROUND_DECIMALS = 4

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

DATA_SOURCES = {
    'mlb': {
        'table_name': 'statcast_major',
        'endpoint': 'statcast_search',
        'url_params': 'minors=false',
        'param_template': [
            "hfInfield=", "hfOutfield=", "hfInn=", "hfBBT=",
            "hfFlag=is%5C.%5C.tracked%7C", "metric_1=", "group_by=name",
            "min_pitches=0", "min_results=0", "min_pas=0", "sort_col=pitches",
            "player_event_sort=api_p_release_speed", "sort_order=desc",
            "type=details", "all=true"
        ],
        'seasons': {
            2021: {'start': '2021-04-01', 'end': '2021-10-03'},
            2022: {'start': '2022-04-07', 'end': '2022-10-05'},
            2023: {'start': '2023-03-30', 'end': '2023-10-01'},
            2024: {'start': '2024-03-28', 'end': '2024-09-29'},
            2025: {'start': '2025-03-27', 'end': '2025-09-28'}
        },
        'description': 'MLB'
    },
    'triple_a': {
        'table_name': 'statcast_minor',
        'endpoint': 'statcast-search-minors',
        'url_params': 'minors=true',
        'param_template': [
            "hfInn=", "hfBBT=", "hfFlag=is%5C.%5C.tracked%7C", "hfLevel=", "metric_1=",
            "hfTeamAffiliate=", "hfOpponentAffiliate=", "group_by=name",
            "min_pitches=0", "min_results=0", "min_pas=0", "sort_col=pitches",
            "player_event_sort=api_p_release_speed", "sort_order=desc",
            "type=details", "all=true"
        ],
        'seasons': {
            2023: {'start': '2023-03-31', 'end': '2023-09-24'},
            2024: {'start': '2024-03-29', 'end': '2024-09-22'},
            2025: {'start': '2025-03-28', 'end': '2025-09-21'}
        },
        'description': 'Triple-A'
    }
}

# === DATABASE ===
class DatabaseManager:
    """Handles DuckDB connection and inserts."""
    def __init__(self, db_path=DB_PATH, schema_path=SCHEMA_PATH):
        self.conn = duckdb.connect(db_path)
        with open(schema_path, 'r') as f:
            self.conn.execute(f.read())

    def close(self):
        self.conn.close()

    def get_latest_date(self, table):
        result = self.conn.execute(f"SELECT MAX(game_date) FROM {table}").fetchone()
        return result[0] if result[0] is not None else None

    def insert_players(self, player_type, df):
        table = f"{player_type}s"
        self.conn.execute(f"""
            INSERT INTO {table} (player_id, player_name)
            SELECT player_id, player_name FROM df
            WHERE player_id NOT IN (SELECT player_id FROM {table})
        """)

    def insert_statcast(self, df, table):
        schema_cols = [col[1] for col in self.conn.execute(f'PRAGMA table_info({table})').fetchall()]
        keep = [col for col in KEEP_COLUMNS if col in df.columns and col in schema_cols]
        df_filtered = df[keep].copy()
        float_cols = df_filtered.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            df_filtered.loc[:, float_cols] = df_filtered.loc[:, float_cols].round(ROUND_DECIMALS)
        df_filtered = df_filtered.reindex(columns=schema_cols)
        self.conn.execute(f"INSERT INTO {table} SELECT * FROM df_filtered")

    def count_rows(self, table):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def update_advanced_metrics(self, table):
        """Update advanced metrics for a specific table"""
        try:
            from Helper_Queries.Statcast_Table_Alter import update_advanced_metrics
            update_advanced_metrics(DB_PATH, table)
            print(f"Advanced metrics calculation completed for {table}")
        except Exception as e:
            print(f"Error calculating advanced metrics for {table}: {e}")
            raise e

    def update_all_advanced_metrics(self):
        """Update advanced metrics for both major and minor league tables"""
        try:
            from Helper_Queries.Statcast_Table_Alter import update_advanced_metrics
            
            # Update both tables
            for table_name in ['statcast_major', 'statcast_minor']:
                print(f"\nCalculating advanced metrics for {table_name}...")
                update_advanced_metrics(DB_PATH, table_name)
                print(f"Advanced metrics calculation completed for {table_name}")
                
        except Exception as e:
            print(f"Error calculating advanced metrics: {e}")
            raise e

# === DATA FETCHING ===
def build_url(date, config):
    year = datetime.strptime(date, '%Y-%m-%d').year
    base = f"https://baseballsavant.mlb.com/{config['endpoint']}/csv"
    params = [
        "hfPT=", "hfAB=", "hfGT=R%7C", "hfPR=", "hfZ=", "hfStadium=", "hfBBL=",
        "hfNewZones=", "hfPull=", "hfC=", f"hfSea={year}%7C", "hfSit=",
        "player_type=batter", "hfOuts=", "hfOpponent=", "pitcher_throws=",
        "batter_stands=", "hfSA=", f"game_date_gt={date}", f"game_date_lt={date}",
        "hfMo=", "hfTeam=", "home_road=", "hfRO=", "position="
    ]
    params.extend(config['param_template'])
    if config['url_params']:
        params.append(config['url_params'])
    return base + "?" + "&".join(params)

def fetch_statcast(date, config):
    url = build_url(date, config)
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return pd.read_csv(StringIO(r.text))
        return None
    except Exception as e:
        print(f"Error fetching {config['description']} for {date}: {e}")
        return None

def fetch_player_data(player_type):
    seasons = list(DATA_SOURCES['mlb']['seasons'].keys())
    seasons_str = '%7C'.join(map(str, seasons))
    url = f"https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea={seasons_str}&hfSit=&player_type={player_type}&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&minors=false"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text))
            return df[['player_id', 'player_name']].drop_duplicates()
        return None
    except Exception as e:
        print(f"Error fetching player data: {e}")
        return None

# === PIPELINE LOGIC ===
def get_dates_to_fetch(seasons, latest):
    if latest is None:
        earliest = min(seasons.keys())
        start = datetime.strptime(seasons[earliest]['start'], '%Y-%m-%d')
    else:
        start = latest + timedelta(days=1) if isinstance(latest, datetime) else datetime.combine(latest, datetime.min.time()) + timedelta(days=1)
    today = datetime.now().date()
    end = today - timedelta(days=1)
    dates = []
    for year in range(start.year, MAX_YEAR):
        if year in seasons:
            s_start = datetime.strptime(seasons[year]['start'], '%Y-%m-%d').date()
            s_end = datetime.strptime(seasons[year]['end'], '%Y-%m-%d').date()
            d = max(start.date(), s_start)
            while d <= min(s_end, end):
                dates.append(d.strftime('%Y-%m-%d'))
                d += timedelta(days=1)
    return dates

def process_source(db, config):
    table = config['table_name']
    print(f"\nProcessing {config['description']}...")
    latest = db.get_latest_date(table)
    dates = get_dates_to_fetch(config['seasons'], latest)
    if not dates:
        print(f"{config['description']} is up to date.")
        return
    print(f"Fetching {len(dates)} dates for {config['description']}...")
    for date in dates:
        df = fetch_statcast(date, config)
        if df is not None and not df.empty:
            db.insert_statcast(df, table)
            print(f"Inserted {len(df)} rows for {date}")
        else:
            print(f"No data for {date}")
        time.sleep(REQUEST_DELAY)
    print(f"Total rows in {table}: {db.count_rows(table)}")

# === MAIN ===
def main():
    print("Starting MLB/Triple-A Statcast Pipeline...")
    caffeinate = subprocess.Popen(['caffeinate', '-i'])
    try:
        db = DatabaseManager()
        print("Fetching player data...")
        for ptype in ['pitcher', 'batter']:
            df = fetch_player_data(ptype)
            if df is not None and not df.empty:
                db.insert_players(ptype, df)
                print(f"Inserted {len(df)} {ptype}s")
            else:
                print(f"No {ptype} data found.")
            time.sleep(REQUEST_DELAY)
        
        # Process each data source
        for source_name, config in DATA_SOURCES.items():
            print(f"\n{'='*50}")
            print(f"Processing {config['description']} ({source_name})...")
            print(f"{'='*50}")
            process_source(db, config)
        
        # Calculate advanced metrics for both tables
        print(f"\n{'='*50}")
        print("Calculating advanced metrics for all tables...")
        print(f"{'='*50}")
        db.update_all_advanced_metrics()
        
        # Verify the advanced metrics were calculated for both tables
        print(f"\n{'='*50}")
        print("Verifying advanced metrics calculation...")
        print(f"{'='*50}")
        for table_name in ['statcast_major', 'statcast_minor']:
            xiso_count = db.conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE HA_Adj_estimated_xISO IS NOT NULL").fetchone()[0]
            total_count = db.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"{table_name}: {xiso_count:,} out of {total_count:,} records have HA_Adj_estimated_xISO calculated")
        
        db.close()
        print("\n" + "="*50)
        print("All data sources processed successfully!")
        print("="*50)
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        caffeinate.terminate()
        caffeinate.wait()

if __name__ == '__main__':
    main() 
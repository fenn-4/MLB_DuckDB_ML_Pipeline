import requests
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from io import StringIO
import time

# Columns to keep from the API data before inserting into the database
# These columns match exactly with schema.sql
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

# Season date ranges
SEASON_RANGES = {
    2022: {'start': '2022-04-07', 'end': '2022-10-05'},
    2023: {'start': '2023-03-30', 'end': '2023-10-01'},
    2024: {'start': '2024-03-28', 'end': '2024-09-29'},
    2025: {'start': '2025-03-27', 'end': '2025-09-28'}
}

def get_player_data(player_type):
    """Fetches player data for a given player type (pitcher or batter).

    Args:
        player_type (str): Either 'pitcher' or 'batter'

    Returns:
        pandas.DataFrame: The fetched data, or None if the request fails.
    """
    url = f"https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea=2025%7C2024%7C2023%7C2022%7C&hfSit=&player_type={player_type}&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&minors=false"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            # Keep only player_id and player_name columns
            return df[['player_id', 'player_name']].drop_duplicates()
        return None
    except:
        return None

def get_statcast_data(date):
    """Fetches Statcast data for a given date from the MLB API.

    Args:
        date (str): The date in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: The fetched data, or None if the request fails.
    """
    year = datetime.strptime(date, '%Y-%m-%d').year
    url = f"https://baseballsavant.mlb.com/statcast_search/csv?hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea={year}%7C&hfSit=&player_type=batter&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={date}&game_date_lt={date}&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=is%5C.%5C.tracked%7C&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&type=details&all=true&minors=false"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        return None
    except:
        return None

def create_database():
    """Creates the DuckDB database and tables if they don't exist.

    Returns:
        duckdb.DuckDBPyConnection: The database connection.
    """
    conn = duckdb.connect('mlb_statcast.db')
    
    # Read and execute schema from file
    with open('schema.sql', 'r') as f:
        schema = f.read()
        conn.execute(schema)
    
    # Create players table if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            player_name VARCHAR
        )
    """)
    
    return conn

def get_latest_date_in_db(conn):
    """Retrieves the latest date in the database.

    Args:
        conn (duckdb.DuckDBPyConnection): The database connection.

    Returns:
        datetime.date: The latest date, or None if the database is empty.
    """
    result = conn.execute("SELECT MAX(game_date) FROM statcast_data").fetchone()
    return result[0] if result[0] is not None else None

def main():
    """Main function to orchestrate the data collection process."""
    conn = create_database()
    
    # Fetch and insert player data
    print("Fetching player data...")
    for player_type in ['pitcher', 'batter']:
        df = get_player_data(player_type)
        if df is not None and not df.empty:
            # Insert only new players
            conn.execute("""
                INSERT INTO players (player_id, player_name)
                SELECT player_id, player_name
                FROM df
                WHERE player_id NOT IN (SELECT player_id FROM players)
            """)
            print(f"Inserted {len(df)} {player_type}s")
        time.sleep(1)
    
    # Continue with statcast data collection
    latest_date = get_latest_date_in_db(conn)
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
            # Get schema columns
            schema_cols = [col[1] for col in conn.execute('PRAGMA table_info(statcast_data)').fetchall()]
            # Determine which columns to keep
            if KEEP_COLUMNS:
                # Only keep columns present in both KEEP_COLUMNS and schema
                keep = [col for col in KEEP_COLUMNS if col in df.columns and col in schema_cols]
            else:
                keep = [col for col in schema_cols if col in df.columns]
            df = df[keep]
            # Round float columns to 4 decimal places
            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = df[float_cols].round(4)
            # Reorder to match schema for insert
            df = df.reindex(columns=schema_cols)
            conn.execute("INSERT INTO statcast_data SELECT * FROM df")
            print(f"Inserted {len(df)} rows for {date_str}")
        else:
            print(f"No data for {date_str}")
        time.sleep(1)
    total_rows = conn.execute("SELECT COUNT(*) FROM statcast_data").fetchone()[0]
    print(f"\nTotal rows in database: {total_rows}")
    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()

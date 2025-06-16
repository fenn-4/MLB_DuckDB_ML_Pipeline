import requests
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from io import StringIO
import time

# Season date ranges
SEASON_RANGES = {
    2022: {'start': '2022-04-07', 'end': '2022-10-05'},
    2023: {'start': '2023-03-30', 'end': '2023-10-01'},
    2024: {'start': '2024-03-28', 'end': '2024-09-29'},
    2025: {'start': '2025-03-27', 'end': '2025-09-28'}
}

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
    """Creates the DuckDB database and table if they don't exist.

    Returns:
        duckdb.DuckDBPyConnection: The database connection.
    """
    conn = duckdb.connect('mlb_statcast.db')
    conn.execute("""
        CREATE TABLE IF NOT EXISTS statcast_data AS SELECT * FROM (SELECT * FROM (SELECT NULL AS pitch_type, NULL AS game_date, NULL AS release_speed, NULL AS release_pos_x, NULL AS release_pos_z, NULL AS player_name, NULL AS batter, NULL AS pitcher, NULL AS events, NULL AS description, NULL AS spin_dir, NULL AS spin_rate_deprecated, NULL AS break_angle_deprecated, NULL AS break_length_deprecated, NULL AS zone, NULL AS des, NULL AS game_type, NULL AS stand, NULL AS p_throws, NULL AS home_team, NULL AS away_team, NULL AS type, NULL AS hit_location, NULL AS bb_type, NULL AS balls, NULL AS strikes, NULL AS game_year, NULL AS pfx_x, NULL AS pfx_z, NULL AS plate_x, NULL AS plate_z, NULL AS on_3b, NULL AS on_2b, NULL AS on_1b, NULL AS outs_when_up, NULL AS inning, NULL AS inning_topbot, NULL AS hc_x, NULL AS hc_y, NULL AS tfs_deprecated, NULL AS tfs_zulu_deprecated, NULL AS umpire, NULL AS sv_id, NULL AS vx0, NULL AS vy0, NULL AS vz0, NULL AS ax, NULL AS ay, NULL AS az, NULL AS sz_top, NULL AS sz_bot, NULL AS hit_distance_sc, NULL AS launch_speed, NULL AS launch_angle, NULL AS effective_speed, NULL AS release_spin_rate, NULL AS release_extension, NULL AS game_pk, NULL AS fielder_2, NULL AS fielder_3, NULL AS fielder_4, NULL AS fielder_5, NULL AS fielder_6, NULL AS fielder_7, NULL AS fielder_8, NULL AS fielder_9, NULL AS release_pos_y, NULL AS estimated_ba_using_speedangle, NULL AS estimated_woba_using_speedangle, NULL AS woba_value, NULL AS woba_denom, NULL AS babip_value, NULL AS iso_value, NULL AS launch_speed_angle, NULL AS at_bat_number, NULL AS pitch_number, NULL AS pitch_name, NULL AS home_score, NULL AS away_score, NULL AS bat_score, NULL AS fld_score, NULL AS post_away_score, NULL AS post_home_score, NULL AS post_bat_score, NULL AS post_fld_score, NULL AS if_fielding_alignment, NULL AS of_fielding_alignment, NULL AS spin_axis, NULL AS delta_home_win_exp, NULL AS delta_run_exp, NULL AS bat_speed, NULL AS swing_length, NULL AS estimated_slg_using_speedangle, NULL AS delta_pitcher_run_exp, NULL AS hyper_speed, NULL AS home_score_diff, NULL AS bat_score_diff, NULL AS home_win_exp, NULL AS bat_win_exp, NULL AS age_pit_legacy, NULL AS age_bat_legacy, NULL AS age_pit, NULL AS age_bat, NULL AS n_thruorder_pitcher, NULL AS n_priorpa_thisgame_player_at_bat, NULL AS pitcher_days_since_prev_game, NULL AS batter_days_since_prev_game, NULL AS pitcher_days_until_next_game, NULL AS batter_days_until_next_game, NULL AS api_break_z_with_gravity, NULL AS api_break_x_arm, NULL AS api_break_x_batter_in, NULL AS arm_angle, NULL AS attack_angle, NULL AS attack_direction, NULL AS swing_path_tilt, NULL AS intercept_ball_minus_batter_pos_x_inches, NULL AS intercept_ball_minus_batter_pos_y_inches) WHERE FALSE)
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

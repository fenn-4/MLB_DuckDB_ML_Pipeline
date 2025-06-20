"""
Rolling Stats Calculator for MLB Statcast Data

This module calculates rolling averages for various metrics from the statcast_data table
and stores them in the rolling_stats table.
"""

import duckdb
import pandas as pd
from datetime import datetime, timedelta

# Define the metrics to track for batters and pitchers
BATTER_METRICS = {
    'attack_angle': 'AVG(attack_angle)',
    'attack_direction': 'AVG(attack_direction)',
    'swing_path_tilt': 'AVG(swing_path_tilt)',
    'bat_speed': 'AVG(bat_speed)',
    'launch_angle': 'AVG(launch_angle)',
    'launch_speed': 'AVG(launch_speed)',
    'estimated_ba': 'AVG(estimated_ba_using_speedangle)',
    'estimated_woba': 'AVG(estimated_woba_using_speedangle)',
    'estimated_slg': 'AVG(estimated_slg_using_speedangle)',
    'estimated_iso': 'AVG(estimated_iso_using_speedangle)'
}

PITCHER_METRICS = {
    'release_speed': 'AVG(release_speed)',
    'release_spin_rate': 'AVG(release_spin_rate)',
    'release_extension': 'AVG(release_extension)',
    'pfx_x': 'AVG(pfx_x)',
    'pfx_z': 'AVG(pfx_z)',
    'arm_angle': 'AVG(arm_angle)'
}

WINDOW_SIZES = [7, 14, 30, 60]  # Rolling windows in days

def calculate_rolling_stats():
    """
    Calculate rolling statistics for both batters and pitchers
    and store them in the rolling_stats table.
    """
    conn = duckdb.connect('Full_DB/mlb_statcast.db')
    
    # Get the date range from the statcast_data table
    date_range = conn.execute("""
        SELECT MIN(game_date) as min_date, MAX(game_date) as max_date 
        FROM statcast_data
    """).fetchone()
    
    if not date_range or not date_range[0] or not date_range[1]:
        print("No data found in statcast_data table")
        return
    
    min_date = date_range[0]
    max_date = date_range[1]
    
    # Calculate rolling stats for batters
    for window_size in WINDOW_SIZES:
        for metric_name, metric_calc in BATTER_METRICS.items():
            query = f"""
            WITH rolling_data AS (
                SELECT 
                    batter as player_id,
                    game_pk,
                    game_date,
                    {metric_calc} as metric_value,
                    COUNT(*) as sample_size
                FROM statcast_data
                WHERE game_date >= ? - INTERVAL {window_size} DAY
                AND game_date <= ?
                AND {metric_name} IS NOT NULL
                GROUP BY batter, game_pk, game_date
            )
            INSERT INTO rolling_stats (player_id, player_type, game_pk, game_date, window_size, metric_name, metric_value, sample_size)
            SELECT 
                player_id,
                'batter',
                game_pk,
                game_date,
                {window_size},
                '{metric_name}',
                metric_value,
                sample_size
            FROM rolling_data
            ON CONFLICT (player_id, player_type, game_pk, window_size, metric_name) 
            DO UPDATE SET metric_value = EXCLUDED.metric_value, sample_size = EXCLUDED.sample_size
            """
            conn.execute(query, [min_date, max_date])
    
    # Calculate rolling stats for pitchers
    for window_size in WINDOW_SIZES:
        for metric_name, metric_calc in PITCHER_METRICS.items():
            query = f"""
            WITH rolling_data AS (
                SELECT 
                    pitcher as player_id,
                    game_pk,
                    game_date,
                    {metric_calc} as metric_value,
                    COUNT(*) as sample_size
                FROM statcast_data
                WHERE game_date >= ? - INTERVAL {window_size} DAY
                AND game_date <= ?
                AND {metric_name} IS NOT NULL
                GROUP BY pitcher, game_pk, game_date
            )
            INSERT INTO rolling_stats (player_id, player_type, game_pk, game_date, window_size, metric_name, metric_value, sample_size)
            SELECT 
                player_id,
                'pitcher',
                game_pk,
                game_date,
                {window_size},
                '{metric_name}',
                metric_value,
                sample_size
            FROM rolling_data
            ON CONFLICT (player_id, player_type, game_pk, window_size, metric_name) 
            DO UPDATE SET metric_value = EXCLUDED.metric_value, sample_size = EXCLUDED.sample_size
            """
            conn.execute(query, [min_date, max_date])
    
    conn.close()
    print("Rolling stats calculation completed")

if __name__ == "__main__":
    calculate_rolling_stats() 
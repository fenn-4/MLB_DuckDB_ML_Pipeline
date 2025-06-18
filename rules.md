# MLB Statcast Data Pipeline Rules

## Overview
This project collects, processes, and stores MLB Statcast data in a DuckDB database. The pipeline includes player and pitch-level data for MLB seasons 2022-2025. The database is initialized and updated using DB_Init.py.

## Database Schema (as of latest update)

### Tables

#### pitchers
- player_id INTEGER PRIMARY KEY
- player_name VARCHAR
- p_throws VARCHAR

#### batters
- player_id INTEGER PRIMARY KEY
- player_name VARCHAR
- stand VARCHAR

#### statcast_data
- game_pk INTEGER
- game_date DATE
- pitch_type VARCHAR
- release_speed DOUBLE
- release_pos_x DOUBLE
- release_pos_z DOUBLE
- batter INTEGER (FK to batters)
- pitcher INTEGER (FK to pitchers)
- events VARCHAR
- description VARCHAR
- zone INTEGER
- stand VARCHAR
- p_throws VARCHAR
- home_team VARCHAR
- away_team VARCHAR
- hit_location INTEGER
- bb_type VARCHAR
- balls INTEGER
- strikes INTEGER
- game_year INTEGER
- pfx_x DOUBLE
- pfx_z DOUBLE
- plate_x DOUBLE
- plate_z DOUBLE
- outs_when_up INTEGER
- inning INTEGER
- inning_topbot VARCHAR
- hc_x DOUBLE
- hc_y DOUBLE
- vx0 DOUBLE
- vy0 DOUBLE
- vz0 DOUBLE
- ax DOUBLE
- ay DOUBLE
- az DOUBLE
- sz_top DOUBLE
- sz_bot DOUBLE
- hit_distance_sc DOUBLE
- launch_speed DOUBLE
- launch_angle DOUBLE
- effective_speed DOUBLE
- release_spin_rate DOUBLE
- release_extension DOUBLE
- release_pos_y DOUBLE
- estimated_ba_using_speedangle DOUBLE
- estimated_woba_using_speedangle DOUBLE
- estimated_slg_using_speedangle DOUBLE
- estimated_iso_using_speedangle DOUBLE
- phi DOUBLE
- pitch_name VARCHAR
- post_away_score INTEGER
- post_home_score INTEGER
- if_fielding_alignment VARCHAR
- of_fielding_alignment VARCHAR
- spin_axis INTEGER
- delta_home_win_exp DOUBLE
- delta_run_exp DOUBLE
- bat_speed DOUBLE
- swing_length DOUBLE
- home_win_exp DOUBLE
- api_break_z_with_gravity DOUBLE
- api_break_x_batter_in DOUBLE
- arm_angle DOUBLE
- attack_angle DOUBLE
- attack_direction DOUBLE
- swing_path_tilt DOUBLE
- intercept_ball_minus_batter_pos_x_inches DOUBLE
- intercept_ball_minus_batter_pos_y_inches DOUBLE
- HA_factor DOUBLE (Hitting Approach factor calculated from phi)

### Notes
- The rolling_stats table and all related logic have been removed from the project as of this update.
- All foreign key constraints are enforced between statcast_data and the player tables.

## Data Collection & Processing
- Player and statcast data are fetched and inserted using DB_Init.py.
- Player info (throwing/batting hand) and advanced metrics (phi, estimated_ISO, HA_factor) are updated after data collection.
- No rolling or windowed statistics are currently calculated or stored.

## Migrations
- If you add or remove columns/tables, update this file and Schema_Init.sql accordingly.

---
_Last updated: [automated update]_

# MLB Statcast Data Pipeline Rules

## Overview
This project collects, processes, and stores MLB and Triple-A Statcast data in a DuckDB database. The pipeline includes player and pitch-level data for MLB seasons 2021-2025 and Triple-A seasons 2023-2025. The database is initialized and updated using the unified `DB_Init_MLB+AAA.py` script.

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

#### statcast_major
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
- HA_Adj_estimated_xISO DOUBLE (HA-adjusted estimated ISO)
- is_zone INTEGER (Strike zone indicator)
- pitch_bucket VARCHAR (Pitch category: Fastball, Breaking Ball, Off-speed, Other)
- pitch_subtype VARCHAR (Specific pitch type)

#### statcast_minor
- Identical structure to statcast_major but for minor league data
- No foreign key constraints (minor league players may not be in major league player tables)
- Contains Triple-A Statcast data from Baseball Savant API
- Populated using the unified `DB_Init_MLB+AAA.py` script

#### start_game
- game_pk INTEGER PRIMARY KEY
- game_date DATE
- home_team VARCHAR
- away_team VARCHAR
- home_starting_pitcher INTEGER (FK to pitchers)
- away_starting_pitcher INTEGER (FK to pitchers)
- home_batter_1 INTEGER
- home_batter_2 INTEGER
- home_batter_3 INTEGER
- home_batter_4 INTEGER
- home_batter_5 INTEGER
- home_batter_6 INTEGER
- home_batter_7 INTEGER
- home_batter_8 INTEGER
- home_batter_9 INTEGER
- away_batter_1 INTEGER
- away_batter_2 INTEGER
- away_batter_3 INTEGER
- away_batter_4 INTEGER
- away_batter_5 INTEGER
- away_batter_6 INTEGER
- away_batter_7 INTEGER
- away_batter_8 INTEGER
- away_batter_9 INTEGER

### Notes
- The rolling_stats table and all related logic have been removed from the project as of this update.
- All foreign key constraints are enforced between statcast_major and the player tables.
- The statcast_minor table has no foreign key constraints to allow for minor league players not in the major league player tables.

## Data Collection & Processing
- **Unified Pipeline**: All data collection is handled by `DB_Init_MLB+AAA.py`
- **MLB Data**: Seasons 2021-2025, stored in `statcast_major` table
- **Triple-A Data**: Seasons 2023-2025, stored in `statcast_minor` table
- **Player Data**: MLB player information fetched and stored in `pitchers` and `batters` tables
- **Advanced Metrics**: Calculated for both MLB and Triple-A data using `Helper_Queries/Statcast_Table_Alter.py`
- **Player Info Updates**: Throwing/batting hand information updated using `Helper_Queries/Player_Tables_Alter.py`
- **Schema Management**: Database schema defined in `Helper_Queries/Schema_Init.sql`

## File Structure
```
MLB_duckDB_ML_Pipeline/
├── DB_Init_MLB+AAA.py          # Main unified data collection script
├── Helper_Queries/             # Helper scripts and schema
│   ├── Schema_Init.sql         # Database schema definition
│   ├── Player_Tables_Alter.py  # Player information updates
│   └── Statcast_Table_Alter.py # Advanced metrics calculations
├── Batter_Pitch_Dashboard.py   # Interactive dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── rules.md                    # This file - detailed rules and schema
```

## Season Date Ranges

### MLB Seasons
- 2021: April 1 - October 3
- 2022: April 7 - October 5
- 2023: March 30 - October 1
- 2024: March 28 - September 29
- 2025: March 27 - September 28

### Triple-A Seasons
- 2023: March 31 - September 24
- 2024: March 29 - September 22
- 2025: March 28 - September 21

## Migrations
- If you add or remove columns/tables, update this file and `Helper_Queries/Schema_Init.sql` accordingly.
- All schema changes should be reflected in both the SQL file and this documentation.

---
_Last updated: [automated update]_

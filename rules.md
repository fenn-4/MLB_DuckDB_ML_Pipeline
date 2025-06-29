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
- blast INTEGER (Power indicator: 1 if HA_Adj_estimated_xISO > 0.3, 0 otherwise)
- is_zone INTEGER (Strike zone indicator: 1=in-zone, 0=out-of-zone)
- pitch_bucket VARCHAR (Pitch category: Fastball, Breaking Ball, Off-speed, Other)
- pitch_subtype VARCHAR (Specific pitch type: Fastball, Slider, Sinker, Curveball, Cutter, Off-speed, Other)

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
├── .gitignore                      # Specifies intentionally untracked files to ignore
├── DB_Init_MLB+AAA.py              # Main unified data collection and database initialization script
├── requirements.txt                # Python dependencies for the project
├── rules.md                        # This file - project documentation, rules, and schema

├── BHM_M&Test/                     # Bayesian Hierarchical Models and Testing
│   ├── combined_whiff_model_bernoulli_enhanced.py      # Enhanced Bernoulli model with 6 whiff groups, 4-way interactions
│   ├── combined_whiff_model_bernoulli_hierarchical.py  # Hierarchical Bernoulli model with 3 whiff groups
│   ├── test_bernoulli_enhanced_model_metrics.py        # Test script for enhanced Bernoulli model
│   └── test_bernoulli_model_metrics.py                 # Test script for hierarchical Bernoulli model

├── Full_DB/
│   ├── mlb_statcast.db             # Main DuckDB database file
│   └── mlb_statcast_backup.db      # Backup of the database

├── Model_Weights/
│   ├── combined_whiff_model_bernoulli_enhanced_trace.nc       # Enhanced Bernoulli model trace (6 groups)
│   ├── combined_whiff_model_bernoulli_enhanced_artifacts.pkl  # Enhanced Bernoulli model artifacts
│   ├── combined_whiff_model_bernoulli_hierarchical_trace.nc   # Hierarchical Bernoulli model trace (3 groups)
│   └── combined_whiff_model_bernoulli_hierarchical_artifacts.pkl # Hierarchical Bernoulli model artifacts

├── Helper_Queries/
│   ├── Schema_Init.sql             # SQL script to define the initial database schema
│   ├── Player_Tables_Alter.py      # Script to update player information (e.g., handedness)
│   ├── Statcast_Table_Alter.py     # Script to calculate advanced metrics and alter statcast tables
│   └── populate_games.py           # Script to populate game-level data

├── Backup_Init/
│   ├── DB_Init.py                  # Older initialization script (likely legacy)
│   └── Minor_League_Data_Init.py   # Older script for minor league data (likely legacy)

└── db/
    └── (empty)                     # Likely placeholder for database-related modules
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

## Current Model Architecture

### Enhanced Bernoulli Model (6 Whiff Groups)
- **Location**: `BHM_M&Test/combined_whiff_model_bernoulli_enhanced.py`
- **Configuration**: 6 whiff groups for both batters and pitchers
- **Key Features**:
  - 4-way interaction: whiff_group × pitcher_whiff_group × zone × pitch_subtype
  - 3-way interaction: zone × pitch_subtype (finer spatial granularity)
  - Individual batter and pitcher deviations
  - Uniform priors (sigma=1) except global intercept (1.5)
- **Test Script**: `BHM_M&Test/test_bernoulli_enhanced_model_metrics.py`

### Hierarchical Bernoulli Model (3 Whiff Groups)
- **Location**: `BHM_M&Test/combined_whiff_model_bernoulli_hierarchical.py`
- **Configuration**: 3 whiff groups (Low, Medium, High)
- **Key Features**:
  - Hierarchical structure from group to individual players
  - 4-way interaction: whiff_group × pitcher_whiff_group × zone × pitch_subtype
  - 3-way interaction: zone × pitch_subtype
  - Hierarchical shrinkage for better parameter estimation
- **Test Script**: `BHM_M&Test/test_bernoulli_model_metrics.py`

## Migrations
- If you add or remove columns/tables, update this file and `Helper_Queries/Schema_Init.sql` accordingly.
- All schema changes should be reflected in both the SQL file and this documentation.
- New machine learning models should be documented in this section with their purpose, features, and dependencies.
- Model performance metrics should be updated as new results become available.

### Recent Updates
- **December 2024**: Streamlined model architecture to two main models
  - Enhanced Bernoulli model with 6 whiff groups for detailed analysis
  - Hierarchical Bernoulli model with 3 whiff groups for computational efficiency
  - Removed redundant models and analysis scripts for cleaner codebase
  - Both models use 4-way whiff_group × pitcher_whiff_group × zone × pitch_subtype interactions
  - Both models retain 3-way zone × pitch_subtype interactions for spatial granularity

- **December 2024**: Updated to 6-bucket whiff group configuration
  - Enhanced Bernoulli model uses 6-bucket configuration for optimal balance of granularity and sample sizes
  - Provides good coverage for 4-way interactions while maintaining reasonable parameter counts
  - Hierarchical model maintains 3-bucket configuration for computational efficiency

- **December 2024**: Removed 2-way whiff group × pitcher whiff group interaction
  - Kept only 4-way interaction: whiff_group × pitcher_whiff_group × zone × pitch_subtype
  - Eliminates multicollinearity with 3-way zone × pitch_subtype interaction
  - Both interactions are complementary: 4-way captures group-level zone/pitch effects, 3-way captures fine spatial granularity

- **December 2024**: Added blast column to Statcast tables
  - Added calculated column `blast` to both `statcast_major` and `statcast_minor` tables
  - `blast` = 1 if `HA_Adj_estimated_xISO` > 0.3, 0 otherwise
  - Provides binary indicator for high-power contact based on analysis showing 39.7% HR rate for ISO > 0.3
  - Updated via `Helper_Queries/Statcast_Table_Alter.py` script

---
_Last updated: December 2024_

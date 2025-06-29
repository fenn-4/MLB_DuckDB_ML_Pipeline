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
├── bayesian_whiff_model.py         # Bayesian hierarchical model for whiff prediction
├── clustered_whiff_model.py        # Clustered batter whiff prediction model
├── clustered_pitcher_model.py     # Clustered pitcher whiff prediction model
├── combined_whiff_model.py         # Combined batter-pitcher meta-model
├── combined_whiff_model_bernoulli.py # Basic Bernoulli model for whiff prediction
├── combined_whiff_model_bernoulli_enhanced.py # Enhanced Bernoulli model with zone parameters and player x zone interactions
├── combined_whiff_model_bernoulli_3way.py # 3-way interaction Bernoulli model
├── routinized_whiff_model.py       # Computationally-efficient hierarchical binomial whiff model
├── evaluate_clustered_model.py     # Evaluation script for clustered batter model
├── evaluate_clustered_pitcher_model.py # Evaluation script for clustered pitcher model
├── test_whiff_model.py             # Test suite for the Bayesian whiff model
├── test_bernoulli_model_metrics.py # Test script for basic Bernoulli model with Brier score, log loss, and AUC
├── test_bernoulli_enhanced_model_metrics.py # Test script for enhanced Bernoulli model with comprehensive metrics

├── Full_DB/
│   ├── mlb_statcast.db             # Main DuckDB database file
│   └── mlb_statcast_backup.db      # Backup of the database

├── Model_Weights/
│   ├── bayesian_whiff_model_2024_2025_trace.nc                # Trained Bayesian model trace
│   ├── bayesian_whiff_model_2024_2025_encoders.pkl            # Feature encoders for the model
│   ├── clustered_whiff_model_2024_2025_trace.nc               # Clustered batter model trace
│   ├── clustered_whiff_model_2024_2025_artifacts.pkl          # Clustered batter model artifacts
│   ├── clustered_pitcher_whiff_model_2024_2025_trace.nc       # Clustered pitcher model trace
│   ├── clustered_pitcher_whiff_model_2024_2025_artifacts.pkl  # Clustered pitcher model artifacts
│   ├── combined_whiff_model_trace.nc                          # Combined model trace
│   ├── combined_whiff_model_artifacts.pkl                     # Combined model artifacts
│   ├── combined_whiff_model_bernoulli_enhanced_trace.nc       # Enhanced Bernoulli model trace
│   ├── combined_whiff_model_bernoulli_enhanced_artifacts.pkl  # Enhanced Bernoulli model artifacts
│   ├── routinized_whiff_model_trace.nc                        # Trace for the routinized binomial whiff model
│   ├── routinized_whiff_model_artifacts.pkl                   # Artifacts for the routinized binomial whiff model
│   └── whiff_model_diagnostics.png                            # Model diagnostic plots

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

## Migrations
- If you add or remove columns/tables, update this file and `Helper_Queries/Schema_Init.sql` accordingly.
- All schema changes should be reflected in both the SQL file and this documentation.
- New machine learning models should be documented in this section with their purpose, features, and dependencies.
- Model performance metrics should be updated as new results become available.

### Recent Updates
- **December 2024**: Updated to 8-bucket whiff group configuration
  - Enhanced Bernoulli model now uses 8-bucket configuration: Very_Low, Low, Low_Med, Medium, Med_High, High, Very_High, Extreme
  - Provides good granularity for detailed player analysis while maintaining reasonable parameter counts
  - Hierarchical model maintains both 3-bucket (Low, Medium, High) and 8-bucket configurations for flexibility
  - Standardized bucket configurations across all models for consistency

- **December 2024**: Updated to 10-bucket (decile) whiff group configuration for high granularity
  - Enhanced Bernoulli model now uses 10-bucket configuration: Decile_1 through Decile_10
  - Provides maximum granularity for detailed player analysis while maintaining reasonable parameter counts
  - Hierarchical model maintains both 3-bucket (Low, Medium, High) and 8-bucket configurations for flexibility
  - Added show_10_bucket_breakdown.py script for detailed decile analysis

- **December 2024**: Added blast column to Statcast tables
  - Added calculated column `blast` to both `statcast_major` and `statcast_minor` tables
  - `blast` = 1 if `HA_Adj_estimated_xISO` > 0.3, 0 otherwise
  - Provides binary indicator for high-power contact based on analysis showing 39.7% HR rate for ISO > 0.3
  - Updated via `Helper_Queries/Statcast_Table_Alter.py` script

- **December 2024**: Added Enhanced Bernoulli Model with Zone Parameters and Player x Zone Interactions
  - Implemented advanced hierarchical Bernoulli model for whiff prediction with enhanced features
  - Includes zone parameters (in-zone vs out-of-zone) and specific zone number effects (zones 1-9, 11-14)
  - Models player x zone interactions for both batters and pitchers
  - Incorporates league-level zone x pitch type interactions
  - Trained on 2024-2025 data with comprehensive test suite including Brier score, log loss, and AUC metrics
  - Provides detailed diagnostics by pitch type and zone location

- **December 2024**: Added Routinized Hierarchical Binomial Whiff Model
  - Implemented a computationally efficient and stable "workhorse" model for whiff prediction.
  - Uses a Binomial likelihood on data aggregated by `(batter, pitcher, p_throws, pitch_subtype, is_zone)`.
  - Models whiff probability via hierarchical effects for players, pitch types, zone location, and their interactions.
  - Includes batter platoon splits and plate discipline (in-zone vs. out-of-zone) effects.
  - Trained on 2024-2025 data for rapid, repeatable analysis.

- **December 2024**: Added Bayesian Hierarchical Whiff Prediction Model
  - Implemented hierarchical Bayesian model for whiff probability prediction
  - Uses batter random effects and fixed effects for pitch characteristics
  - Trained on 2024-2025 swing data with 714 qualified batters
  - Achieves proper convergence with R-hat < 1.1 for all parameters

---
_Last updated: December 2024_

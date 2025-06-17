# MLB Statcast Data Pipeline

## Project Overview
This project collects MLB Statcast data from 2022-2025 and stores it in a DuckDB database for analysis. The data is collected day by day during the regular season for each year.

## Season Date Ranges
- 2022: April 7 - October 5
- 2023: March 30 - October 1
- 2024: March 28 - September 29
- 2025: March 27 - September 28

## Database Schema

### Table: pitchers
A reference table containing pitcher information:
- player_id (INTEGER PRIMARY KEY): Unique identifier for each pitcher
- player_name (VARCHAR): Pitcher's full name

### Table: batters
A reference table containing batter information:
- player_id (INTEGER PRIMARY KEY): Unique identifier for each batter
- player_name (VARCHAR): Batter's full name

### Table: statcast_data
The main table containing all Statcast data with the following columns:

#### Game Information
- game_pk (INTEGER): Unique game identifier
- game_date (DATE): Date of the game
- game_year (INTEGER): Year of the game
- home_team (VARCHAR): Home team
- away_team (VARCHAR): Away team
- post_away_score (INTEGER): Final away team score
- post_home_score (INTEGER): Final home team score
- home_win_exp (DOUBLE): Home team win expectancy
- delta_home_win_exp (DOUBLE): Change in home win expectancy
- delta_run_exp (DOUBLE): Change in run expectancy

#### Player Information
- batter (INTEGER): Batter ID (references batters.player_id)
- pitcher (INTEGER): Pitcher ID (references pitchers.player_id)
- stand (VARCHAR): Batter's stance
- p_throws (VARCHAR): Pitcher's throwing hand

#### Pitch Information
- pitch_type (VARCHAR): Type of pitch thrown
- pitch_name (VARCHAR): Name of the pitch
- release_speed (DOUBLE): Speed of the pitch at release
- release_pos_x (DOUBLE): X position of release
- release_pos_y (DOUBLE): Y position of release
- release_pos_z (DOUBLE): Z position of release
- release_spin_rate (DOUBLE): Spin rate of the pitch
- release_extension (DOUBLE): Extension of the pitch
- effective_speed (DOUBLE): Effective speed of the pitch
- spin_axis (INTEGER): Spin axis of the pitch
- arm_angle (DOUBLE): Arm angle
- attack_angle (DOUBLE): Attack angle
- attack_direction (DOUBLE): Attack direction
- swing_path_tilt (DOUBLE): Swing path tilt
- api_break_z_with_gravity (DOUBLE): Break Z with gravity
- api_break_x_batter_in (DOUBLE): Break X from batter's perspective
- pfx_x (DOUBLE): Horizontal movement
- pfx_z (DOUBLE): Vertical movement
- plate_x (DOUBLE): Horizontal plate location
- plate_z (DOUBLE): Vertical plate location
- sz_top (DOUBLE): Strike zone top
- sz_bot (DOUBLE): Strike zone bottom
- vx0, vy0, vz0 (DOUBLE): Initial velocity components
- ax, ay, az (DOUBLE): Acceleration components

#### Pitch Outcome
- events (VARCHAR): Result of the pitch
- description (VARCHAR): Description of the pitch
- hit_location (INTEGER): Location of hit
- bb_type (VARCHAR): Type of batted ball
- balls (INTEGER): Ball count
- strikes (INTEGER): Strike count
- zone (INTEGER): Pitch zone
- outs_when_up (INTEGER): Outs when up
- inning (INTEGER): Inning number
- inning_topbot (VARCHAR): Top or bottom of inning

#### Batted Ball Information
- launch_speed (DOUBLE): Exit velocity
- launch_angle (DOUBLE): Launch angle
- hit_distance_sc (DOUBLE): Hit distance
- hc_x (DOUBLE): Hit coordinate X
- hc_y (DOUBLE): Hit coordinate Y
- estimated_ba_using_speedangle (DOUBLE): Estimated batting average
- estimated_woba_using_speedangle (DOUBLE): Estimated wOBA
- estimated_slg_using_speedangle (DOUBLE): Estimated slugging percentage
- bat_speed (DOUBLE): Bat speed
- swing_length (DOUBLE): Swing length

#### Fielding Information
- if_fielding_alignment (VARCHAR): Infield alignment
- of_fielding_alignment (VARCHAR): Outfield alignment

#### Additional Metrics
- intercept_ball_minus_batter_pos_x_inches (DOUBLE): Ball-batter intercept X
- intercept_ball_minus_batter_pos_y_inches (DOUBLE): Ball-batter intercept Y

## Data Collection Process
1. Data is collected day by day during the regular season
2. Player data is fetched first for both pitchers and batters into their respective tables
3. Each day's Statcast data is fetched from the MLB Statcast API
4. Data is immediately inserted into the DuckDB database
5. A 1-second delay is implemented between requests to avoid overwhelming the server

## Usage
Run the db_init.py script to begin data collection:
```bash
python db_init.py
```

The script will:
1. Create the DuckDB database if it doesn't exist
2. Create the pitchers, batters, and statcast_data tables with the defined schemas
3. Collect player data for both pitchers and batters into their respective tables
4. Collect data for each day in the specified season ranges
5. Store the data in the database
6. After data collection, update player info (handedness, stance) and calculate advanced metrics (phi, estimated ISO, etc.)

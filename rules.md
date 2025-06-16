# MLB Statcast Data Pipeline

## Project Overview
This project collects MLB Statcast data from 2022-2025 and stores it in a DuckDB database for analysis. The data is collected day by day during the regular season for each year.

## Season Date Ranges
- 2022: April 7 - October 5
- 2023: March 30 - October 1
- 2024: March 28 - September 29
- 2025: March 27 - September 28

## Database Schema

### Table: statcast_data
The main table containing all Statcast data with the following columns:

#### Pitch Information
- pitch_type (VARCHAR): Type of pitch thrown
- pitch_name (VARCHAR): Name of the pitch
- release_speed (DOUBLE): Speed of the pitch at release
- release_pos_x (DOUBLE): X position of release
- release_pos_z (DOUBLE): Z position of release
- release_pos_y (DOUBLE): Y position of release
- release_spin_rate (DOUBLE): Spin rate of the pitch
- release_extension (DOUBLE): Extension of the pitch
- effective_speed (DOUBLE): Effective speed of the pitch

#### Game Information
- game_date (DATE): Date of the game
- game_type (VARCHAR): Type of game
- game_year (INTEGER): Year of the game
- game_pk (INTEGER): Unique game identifier
- inning (INTEGER): Inning number
- inning_topbot (VARCHAR): Top or bottom of inning
- home_team (VARCHAR): Home team
- away_team (VARCHAR): Away team

#### Player Information
- player_name (VARCHAR): Name of the player
- batter (INTEGER): Batter ID
- pitcher (INTEGER): Pitcher ID
- stand (VARCHAR): Batter's stance
- p_throws (VARCHAR): Pitcher's throwing hand

#### Pitch Outcome
- events (VARCHAR): Result of the pitch
- description (VARCHAR): Description of the pitch
- type (VARCHAR): Type of outcome
- hit_location (INTEGER): Location of hit
- bb_type (VARCHAR): Type of batted ball
- balls (INTEGER): Ball count
- strikes (INTEGER): Strike count

#### Advanced Metrics
- launch_speed (DOUBLE): Exit velocity
- launch_angle (DOUBLE): Launch angle
- hit_distance_sc (DOUBLE): Hit distance
- estimated_ba_using_speedangle (DOUBLE): Estimated batting average
- estimated_woba_using_speedangle (DOUBLE): Estimated wOBA
- estimated_slg_using_speedangle (DOUBLE): Estimated slugging percentage

#### Fielding Information
- fielder_2 through fielder_9 (INTEGER): Fielders involved
- if_fielding_alignment (VARCHAR): Infield alignment
- of_fielding_alignment (VARCHAR): Outfield alignment

#### Additional Metrics
- spin_axis (INTEGER): Spin axis of the pitch
- arm_angle (DOUBLE): Arm angle
- attack_angle (DOUBLE): Attack angle
- attack_direction (DOUBLE): Attack direction
- swing_path_tilt (DOUBLE): Swing path tilt

## Data Collection Process
1. Data is collected day by day during the regular season
2. Each day's data is fetched from the MLB Statcast API
3. Data is immediately inserted into the DuckDB database
4. A 1-second delay is implemented between requests to avoid overwhelming the server

## Usage
Run the start.py script to begin data collection:
```bash
python start.py
```

The script will:
1. Create the DuckDB database if it doesn't exist
2. Create the statcast_data table with the defined schema
3. Collect data for each day in the specified season ranges
4. Store the data in the database

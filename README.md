# MLB Statcast Data Pipeline

## Project Overview
This project collects MLB Statcast data from 2022-2025 and stores it in a DuckDB database for analysis. The data is collected day by day during the regular season for each year.

## Features
- Fetches Statcast data from the MLB API.
- Stores data in a DuckDB database.
- Resumes data collection from the last date if interrupted.
- Automatically updates the database up to yesterday.

## Requirements
- Python 3.6+
- DuckDB
- Pandas
- Requests

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MLB_duckDB_ML_Pipeline.git
   cd MLB_duckDB_ML_Pipeline
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script to begin data collection:
```bash
python start.py
```

The script will:
1. Create the DuckDB database if it doesn't exist.
2. Fetch data for each day in the specified season ranges.
3. Store the data in the database.

## Database Schema
The database contains a table `statcast_data` with 118 columns, including:
- `pitch_type`: Type of pitch (e.g., "FF" for fastball).
- `game_date`: Date of the game.
- `release_speed`: Speed of the pitch at release.
- `player_name`: Name of the player.
- And many more advanced metrics.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 
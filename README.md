# MLB Statcast Data Pipeline

## Project Overview
This project collects MLB and Triple-A Statcast data from 2022-2025 and stores it in a DuckDB database for analysis. The data is collected day by day during the regular season for each year using a unified pipeline that handles both major and minor league data. The project also includes an interactive dashboard for analyzing pitch data.

## Features
- Fetches Statcast data from the MLB API for both MLB and Triple-A leagues
- Stores data in separate DuckDB tables (`statcast_data` for MLB, `statcast_minor` for Triple-A)
- Resumes data collection from the last date if interrupted
- Automatically updates the database up to yesterday
- Calculates advanced metrics (phi, estimated_ISO, HA_factor, etc.)
- Updates player information (throwing/batting hand)
- Supports configurable data sources and date ranges
- **Interactive Dashboard**: Web-based interface for analyzing pitch data with advanced filtering

## Requirements
- Python 3.6+
- DuckDB
- Pandas
- Requests
- Streamlit (for dashboard)
- Plotly (for visualizations)
- NumPy

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

### Data Collection
Run the unified script to begin data collection for both MLB and Triple-A:
```bash
python DB_Init_MLB+AAA.py
```

The script will:
1. Create the DuckDB database if it doesn't exist
2. Initialize the database schema from `Helper_Queries/Schema_Init.sql`
3. Fetch player data for MLB (pitchers and batters)
4. Process MLB data (2022-2025 seasons)
5. Process Triple-A data (2023-2025 seasons)
6. Calculate advanced metrics for both datasets
7. Update player information

### Interactive Dashboard
Run the dashboard to analyze pitch data:
```bash
streamlit run Batter_Pitch_Dashboard.py
```

The dashboard will open in your browser (typically at http://localhost:8501) and provides:
- **Interactive Filters**: Toggle by year, pitcher throwing arm (L/R), and ball-strike count
- **Color-coded Pitch Plots**: Visualize pitch locations by subtype
- **Batter-specific Strike Zone**: Uses the same zone calculation logic as the data pipeline
- **Comprehensive Analytics**: Speed distributions, pitch summaries, and raw data tables
- **Data Export**: Download filtered data as CSV

#### Dashboard Features
- **Main Pitch Plot**: Shows pitch locations relative to home plate, color-coded by pitch subtype
- **Summary Statistics**: Total pitches faced, unique pitchers, pitch type diversity
- **Pitch Summary Table**: Count, average speed, and zone rate by pitch type
- **Speed Distribution**: Histogram showing speed distribution by pitch type
- **Raw Data Table**: Detailed view of individual pitches

## Database Schema
The database contains multiple tables:

### Main Data Tables
- `statcast_data`: MLB Statcast data with 118 columns
- `statcast_minor`: Triple-A Statcast data (identical structure to MLB)

### Player Tables
- `pitchers`: MLB pitcher information (player_id, player_name, p_throws)
- `batters`: MLB batter information (player_id, player_name, stand)

### Game Information
- `start_game`: Game-level information including starting lineups

### Key Columns in Statcast Tables
- `pitch_type`: Type of pitch (e.g., "FF" for fastball)
- `game_date`: Date of the game
- `release_speed`: Speed of the pitch at release
- `phi`: Launch angle calculation
- `estimated_iso_using_speedangle`: Estimated ISO using speed and angle
- `HA_factor`: Hitting Approach factor
- `pitch_bucket`: Pitch category (Fastball, Breaking Ball, Off-speed, Other)
- `pitch_subtype`: Specific pitch type
- And many more advanced metrics

## Project Structure
```
MLB_duckDB_ML_Pipeline/
├── DB_Init_MLB+AAA.py          # Main unified data collection script
├── Batter_Pitch_Dashboard.py   # Interactive dashboard
├── Helper_Queries/             # Helper scripts and schema
│   ├── Schema_Init.sql         # Database schema definition
│   ├── Player_Tables_Alter.py  # Player information updates
│   └── Statcast_Table_Alter.py # Advanced metrics calculations
├── requirements.txt            # Python dependencies (consolidated)
├── README.md                   # This file
└── rules.md                    # Detailed project rules and schema
```

## Data Sources
- **MLB Data**: 2022-2025 regular seasons
- **Triple-A Data**: 2023-2025 regular seasons
- **Source**: Baseball Savant API

## Technical Details

### Data Pipeline
- **Database**: DuckDB for efficient storage and querying
- **Advanced Metrics**: Calculated using the same logic as Statcast
- **Resume Capability**: Automatically resumes from last collected date

### Dashboard
- **Framework**: Streamlit for web interface
- **Visualization**: Plotly for interactive charts
- **Zone Calculation**: Implements the same logic as Statcast_Table_Alter.py:
  - Horizontal: ±10/12 feet from center
  - Vertical: Batter-specific boundaries using sz_top and sz_bot

## License
This project is licensed under the MIT License - see the LICENSE file for details. 
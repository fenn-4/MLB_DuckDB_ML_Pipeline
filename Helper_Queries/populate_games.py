"""
Populate Games Table Script

This script creates a start_game table that records every unique game from the statcast_data table,
including starting pitchers and the first 9 batters for each team in order of appearance.
"""

import duckdb
import pandas as pd

def populate_games_table():
    """
    Populate the start_game table with unique game information from statcast_data.
    """
    conn = duckdb.connect('Full_DB/mlb_statcast.db')
    
    print("Creating start_game table...")
    
    # Get unique games with basic info
    games_query = """
    SELECT DISTINCT 
        game_pk,
        game_date,
        home_team,
        away_team
    FROM statcast_data 
    WHERE game_pk IS NOT NULL
    ORDER BY game_pk
    """
    
    games_df = conn.execute(games_query).df()
    print(f"Found {len(games_df)} unique games")
    
    # Process each game to get starting pitchers and batters
    game_records = []
    
    for _, game in games_df.iterrows():
        game_pk = game['game_pk']
        
        # Get starting pitchers using inning_topbot
        # Home pitcher: first pitcher in 'top' inning (home team pitches in top)
        # Away pitcher: first pitcher in 'bot' inning (away team pitches in bottom)
        pitchers_query = """
        SELECT 
            pitcher,
            inning_topbot,
            ROW_NUMBER() OVER (PARTITION BY inning_topbot ORDER BY game_date, inning, outs_when_up) as rn
        FROM statcast_data 
        WHERE game_pk = ? AND pitcher IS NOT NULL AND inning_topbot IS NOT NULL
        """
        
        pitchers_df = conn.execute(pitchers_query, [game_pk]).df()
        
        home_starting_pitcher = None
        away_starting_pitcher = None
        
        for _, pitcher_row in pitchers_df.iterrows():
            if pitcher_row['inning_topbot'] == 'Top' and pitcher_row['rn'] == 1:
                home_starting_pitcher = pitcher_row['pitcher']
            elif pitcher_row['inning_topbot'] == 'Bot' and pitcher_row['rn'] == 1:
                away_starting_pitcher = pitcher_row['pitcher']
        
        # Get first 9 batters for home team (bat in bottom of inning)
        home_batters_query = """
        SELECT DISTINCT batter
        FROM statcast_data 
        WHERE game_pk = ? AND inning_topbot = 'Bot' AND batter IS NOT NULL
        ORDER BY game_date, inning, outs_when_up
        LIMIT 9
        """
        
        home_batters_df = conn.execute(home_batters_query, [game_pk]).df()
        home_batters = home_batters_df['batter'].tolist() if not home_batters_df.empty else []
        
        # Pad with None if less than 9 batters
        while len(home_batters) < 9:
            home_batters.append(None)
        
        # Get first 9 batters for away team (bat in top of inning)
        away_batters_query = """
        SELECT DISTINCT batter
        FROM statcast_data 
        WHERE game_pk = ? AND inning_topbot = 'Top' AND batter IS NOT NULL
        ORDER BY game_date, inning, outs_when_up
        LIMIT 9
        """
        
        away_batters_df = conn.execute(away_batters_query, [game_pk]).df()
        away_batters = away_batters_df['batter'].tolist() if not away_batters_df.empty else []
        
        # Pad with None if less than 9 batters
        while len(away_batters) < 9:
            away_batters.append(None)
        
        # Create game record
        game_record = {
            'game_pk': game_pk,
            'game_date': game['game_date'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'home_starting_pitcher': home_starting_pitcher,
            'away_starting_pitcher': away_starting_pitcher,
            'home_batter_1': home_batters[0] if len(home_batters) > 0 else None,
            'home_batter_2': home_batters[1] if len(home_batters) > 1 else None,
            'home_batter_3': home_batters[2] if len(home_batters) > 2 else None,
            'home_batter_4': home_batters[3] if len(home_batters) > 3 else None,
            'home_batter_5': home_batters[4] if len(home_batters) > 4 else None,
            'home_batter_6': home_batters[5] if len(home_batters) > 5 else None,
            'home_batter_7': home_batters[6] if len(home_batters) > 6 else None,
            'home_batter_8': home_batters[7] if len(home_batters) > 7 else None,
            'home_batter_9': home_batters[8] if len(home_batters) > 8 else None,
            'away_batter_1': away_batters[0] if len(away_batters) > 0 else None,
            'away_batter_2': away_batters[1] if len(away_batters) > 1 else None,
            'away_batter_3': away_batters[2] if len(away_batters) > 2 else None,
            'away_batter_4': away_batters[3] if len(away_batters) > 3 else None,
            'away_batter_5': away_batters[4] if len(away_batters) > 4 else None,
            'away_batter_6': away_batters[5] if len(away_batters) > 5 else None,
            'away_batter_7': away_batters[6] if len(away_batters) > 6 else None,
            'away_batter_8': away_batters[7] if len(away_batters) > 7 else None,
            'away_batter_9': away_batters[8] if len(away_batters) > 8 else None
        }
        
        game_records.append(game_record)
    
    # Convert to DataFrame and insert into start_game table
    games_insert_df = pd.DataFrame(game_records)
    
    # Insert into start_game table
    conn.execute("DELETE FROM start_game")  # Clear existing data
    conn.execute("INSERT INTO start_game SELECT * FROM games_insert_df")
    
    print(f"Successfully populated start_game table with {len(game_records)} games")
    
    # Show sample of populated data
    print("\nSample start_game data:")
    sample_games = conn.execute("""
        SELECT g.game_pk, g.game_date, g.home_team, g.away_team,
               hp.player_name as home_pitcher, ap.player_name as away_pitcher
        FROM start_game g
        LEFT JOIN pitchers hp ON g.home_starting_pitcher = hp.player_id
        LEFT JOIN pitchers ap ON g.away_starting_pitcher = ap.player_id
        ORDER BY g.game_pk
        LIMIT 5
    """).df()
    
    print(sample_games)
    
    conn.close()

if __name__ == "__main__":
    populate_games_table() 
import duckdb

def update_player_info():
    # Connect to the database
    conn = duckdb.connect('Full_DB/mlb_statcast.db')
    
    # Add columns if they don't exist
    print("Adding columns if they don't exist...")
    conn.execute("ALTER TABLE pitchers ADD COLUMN IF NOT EXISTS p_throws VARCHAR")
    conn.execute("ALTER TABLE batters ADD COLUMN IF NOT EXISTS stand VARCHAR")
    
    # Update pitchers table with p_throws
    print("Updating pitchers table with throwing information...")
    conn.execute("""
        UPDATE pitchers p
        SET p_throws = (
            SELECT DISTINCT p_throws 
            FROM statcast_data s 
            WHERE s.pitcher = p.player_id 
            LIMIT 1
        )
    """)
    # Reformat player_name for pitchers
    print("Reformatting pitcher player names...")
    conn.execute("""
        UPDATE pitchers
        SET player_name =
            CASE 
                WHEN player_name LIKE '%,%' 
                THEN ltrim(split_part(player_name, ',', 2)) || ' ' || split_part(player_name, ',', 1)
                ELSE player_name 
            END
    """)
    
    # Update batters table with stand
    print("Updating batters table with batting stance information...")
    conn.execute("""
        UPDATE batters b
        SET stand = (
            SELECT DISTINCT stand 
            FROM statcast_data s 
            WHERE s.batter = b.player_id 
            LIMIT 1
        )
    """)
    # Reformat player_name for batters
    print("Reformatting batter player names...")
    conn.execute("""
        UPDATE batters
        SET player_name =
            CASE 
                WHEN player_name LIKE '%,%' 
                THEN ltrim(split_part(player_name, ',', 2)) || ' ' || split_part(player_name, ',', 1)
                ELSE player_name 
            END
    """)
    
    # Close the connection
    conn.close()

if __name__ == "__main__":
    update_player_info() 
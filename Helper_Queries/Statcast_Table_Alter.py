"""
Update advanced metrics in Statcast tables.
"""

import duckdb
import numpy as np

def update_advanced_metrics(db_path='Full_DB/mlb_statcast.db', table_name='statcast_major'):
    """
    Calculate and update advanced metrics in the specified table.
    
    Args:
        db_path (str): Path to the database file
        table_name (str): Name of the table to update (default: statcast_major)
    """
    conn = duckdb.connect(db_path)
    
    print(f"Adding new columns to {table_name} if they don't exist...")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS phi DOUBLE")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS estimated_iso_using_speedangle DOUBLE")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS HA_factor DOUBLE")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS HA_Adj_estimated_xISO DOUBLE")
    
    print(f"Updating advanced metrics in {table_name}...")
    
    # Calculate phi (hit angle) - only for records with hit coordinates
    print(f"Calculating phi (hit angle) for {table_name}...")
    conn.execute(f"""
        UPDATE {table_name}
        SET phi = CASE 
            WHEN hc_x IS NOT NULL AND hc_y IS NOT NULL AND (198.27 - hc_y) != 0
            THEN ATAN((hc_x - 125.42) / (198.27 - hc_y))
            ELSE NULL 
        END
        WHERE hc_x IS NOT NULL AND hc_y IS NOT NULL
    """)
    
    # Calculate estimated_iso_using_speedangle - only for records with both BA and SLG
    print(f"Calculating estimated_iso_using_speedangle for {table_name}...")
    conn.execute(f"""
        UPDATE {table_name}
        SET estimated_iso_using_speedangle = CASE 
            WHEN estimated_ba_using_speedangle IS NOT NULL AND estimated_slg_using_speedangle IS NOT NULL 
            THEN (estimated_slg_using_speedangle - estimated_ba_using_speedangle)
            ELSE NULL 
        END
        WHERE estimated_ba_using_speedangle IS NOT NULL AND estimated_slg_using_speedangle IS NOT NULL
    """)
    
    # Calculate HA_factor - only for records with phi
    print(f"Calculating HA_factor for {table_name}...")
    conn.execute(f"""
        UPDATE {table_name}
        SET HA_factor = CASE 
            WHEN phi IS NOT NULL 
            THEN 0.569 * POWER(
                CASE 
                    WHEN phi < -1 THEN -1
                    WHEN phi > 1 THEN 1
                    ELSE phi
                END, 
                2
            ) + 0.569
            ELSE NULL 
        END
        WHERE phi IS NOT NULL
    """)
    
    # Calculate HA_Adj_estimated_xISO - only for records with both components
    print(f"Calculating HA_Adj_estimated_xISO for {table_name}...")
    conn.execute(f"""
        UPDATE {table_name}
        SET HA_Adj_estimated_xISO = estimated_iso_using_speedangle * HA_factor
        WHERE estimated_iso_using_speedangle IS NOT NULL AND HA_factor IS NOT NULL
    """)
    
    # Add blast column if it doesn't exist
    print(f"Adding blast column to {table_name} if it doesn't exist...")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS blast INTEGER")
    
    # Calculate blast - 1 if HA_Adj_estimated_xISO > 0.3, 0 otherwise
    print(f"Calculating blast for {table_name}...")
    conn.execute(f"""
        UPDATE {table_name}
        SET blast = CASE 
            WHEN HA_Adj_estimated_xISO > 0.3 THEN 1
            ELSE 0
        END
        WHERE HA_Adj_estimated_xISO IS NOT NULL
    """)
    
    # Add is_zone column if it doesn't exist
    print(f"Adding is_zone column to {table_name} if it doesn't exist...")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS is_zone INTEGER")
    
    # Update is_zone column with new strike zone boundaries
    print(f"Updating is_zone column in {table_name} with new strike zone boundaries...")
    conn.execute(f"""
        UPDATE {table_name}
        SET is_zone = CASE
            WHEN plate_x BETWEEN -10/12 AND 10/12 
            AND plate_z BETWEEN (sz_bot - 3/12 - (sz_bot - 1.69)/4) AND (sz_top - (sz_top - 3.5)/3)
            THEN 1
            ELSE 0
        END
        WHERE plate_x IS NOT NULL AND plate_z IS NOT NULL
    """)
    
    # Add pitch_bucket and pitch_subtype columns if they don't exist
    print(f"Adding pitch_bucket and pitch_subtype columns to {table_name} if they don't exist...")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS pitch_bucket VARCHAR")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS pitch_subtype VARCHAR")

    # Update pitch_bucket and pitch_subtype based on pitch_type
    print(f"Updating pitch_bucket and pitch_subtype columns in {table_name}...")
    conn.execute(f"""
        UPDATE {table_name}
        SET 
            pitch_bucket = CASE pitch_type
                WHEN 'FF' THEN 'Fastball'
                WHEN 'FA' THEN 'Fastball'
                WHEN 'SI' THEN 'Fastball'
                WHEN 'FT' THEN 'Fastball'
                WHEN 'FC' THEN 'Fastball'
                WHEN 'SL' THEN 'Breaking Ball'
                WHEN 'ST' THEN 'Breaking Ball'
                WHEN 'SV' THEN 'Breaking Ball'
                WHEN 'CU' THEN 'Breaking Ball'
                WHEN 'KC' THEN 'Breaking Ball'
                WHEN 'CS' THEN 'Breaking Ball'
                WHEN 'CH' THEN 'Off-speed'
                WHEN 'FS' THEN 'Off-speed'
                WHEN 'FO' THEN 'Off-speed'
                WHEN 'SC' THEN 'Other'
                WHEN 'KN' THEN 'Other'
                WHEN 'EP' THEN 'Other'
                WHEN 'UN' THEN 'Other'
                WHEN 'XX' THEN 'Other'
                WHEN 'PO' THEN 'Other'
                ELSE NULL
            END,
            pitch_subtype = CASE pitch_type
                WHEN 'FF' THEN 'Fastball'
                WHEN 'FA' THEN 'Fastball'
                WHEN 'SI' THEN 'Sinker'
                WHEN 'FT' THEN 'Sinker'
                WHEN 'FC' THEN 'Cutter'
                WHEN 'SL' THEN 'Slider'
                WHEN 'ST' THEN 'Slider'
                WHEN 'SV' THEN 'Curveball'
                WHEN 'CU' THEN 'Curveball'
                WHEN 'KC' THEN 'Curveball'
                WHEN 'CS' THEN 'Curveball'
                WHEN 'CH' THEN 'Changeup'
                WHEN 'FS' THEN 'Splitter'
                WHEN 'FO' THEN 'Splitter'
                WHEN 'SC' THEN 'Other'
                WHEN 'KN' THEN 'Other'
                WHEN 'EP' THEN 'Other'
                WHEN 'UN' THEN 'Other'
                WHEN 'XX' THEN 'Other'
                WHEN 'PO' THEN 'Other'
                ELSE NULL
            END
        WHERE pitch_type IS NOT NULL
    """)
    
    # Commit the changes
    conn.commit()
    
    # Close the connection
    conn.close()
    print(f"Advanced metrics updated successfully for {table_name}")

if __name__ == "__main__":
    # Update both major and minor league tables
    print("Updating major league table...")
    update_advanced_metrics('Full_DB/mlb_statcast.db', 'statcast_major')
    
    print("\nUpdating minor league table...")
    update_advanced_metrics('Full_DB/mlb_statcast.db', 'statcast_minor')
    
    print("\nBoth tables updated successfully!") 
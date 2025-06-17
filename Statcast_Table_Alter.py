import duckdb
import numpy as np

def update_advanced_metrics():
    # Connect to the database
    conn = duckdb.connect('mlb_statcast.db')
    
    print("Adding new columns if they don't exist...")
    conn.execute("ALTER TABLE statcast_data ADD COLUMN IF NOT EXISTS phi DOUBLE")
    conn.execute("ALTER TABLE statcast_data ADD COLUMN IF NOT EXISTS estimated_iso_using_speedangle DOUBLE")
    
    print("Updating advanced metrics...")
    
    # Calculate phi and estimated_ISO
    conn.execute("""
        UPDATE statcast_data
        SET 
            phi = CASE 
                WHEN hc_x IS NOT NULL AND hc_y IS NOT NULL AND (198.27 - hc_y) != 0 
                THEN ROUND(ATAN((hc_x - 125.42) / (198.27 - hc_y)), 4)
                ELSE NULL 
            END,
            estimated_iso_using_speedangle = CASE 
                WHEN estimated_slg_using_speedangle IS NOT NULL AND estimated_ba_using_speedangle IS NOT NULL 
                THEN ROUND(estimated_slg_using_speedangle - estimated_ba_using_speedangle, 4)
                ELSE NULL 
            END
    """)
    
    # Add is_zone column if it doesn't exist
    print("Adding is_zone column if it doesn't exist...")
    conn.execute("ALTER TABLE statcast_data ADD COLUMN IF NOT EXISTS is_zone INTEGER")
    
    # Update is_zone column with new strike zone boundaries
    print("Updating is_zone column with new strike zone boundaries...")
    conn.execute("""
        UPDATE statcast_data
        SET is_zone = CASE
            WHEN plate_x BETWEEN -10/12 AND 10/12 
            AND plate_z BETWEEN (sz_bot - 3/12 - (sz_bot - 1.69)/4) AND (sz_top - (sz_top - 3.5)/3)
            THEN 1
            ELSE 0
        END
        WHERE plate_x IS NOT NULL AND plate_z IS NOT NULL
    """)
    
    # Add pitch_bucket and pitch_subtype columns if they don't exist
    print("Adding pitch_bucket and pitch_subtype columns if they don't exist...")
    conn.execute("ALTER TABLE statcast_data ADD COLUMN IF NOT EXISTS pitch_bucket VARCHAR")
    conn.execute("ALTER TABLE statcast_data ADD COLUMN IF NOT EXISTS pitch_subtype VARCHAR")

    # Update pitch_bucket and pitch_subtype based on pitch_type
    print("Updating pitch_bucket and pitch_subtype columns...")
    conn.execute("""
        UPDATE statcast_data
        SET 
            pitch_bucket = CASE pitch_type
                WHEN 'FF' THEN 'Fastball'
                WHEN 'FA' THEN 'Fastball'
                WHEN 'SI' THEN 'Fastball'
                WHEN 'FT' THEN 'Fastball'
                WHEN 'FC' THEN 'Fastball'
                WHEN 'SL' THEN 'Breaking Ball'
                WHEN 'ST' THEN 'Breaking Ball'
                WHEN 'CU' THEN 'Breaking Ball'
                WHEN 'KC' THEN 'Breaking Ball'
                WHEN 'CS' THEN 'Breaking Ball'
                WHEN 'CH' THEN 'Off-speed'
                WHEN 'FS' THEN 'Off-speed'
                WHEN 'FO' THEN 'Off-speed'
                WHEN 'SC' THEN 'Other'
                WHEN 'KN' THEN 'Other'
                WHEN 'EP' THEN 'Other'
                WHEN 'UN' THEN 'Unknown'
                WHEN 'XX' THEN 'Unknown'
                WHEN 'PO' THEN 'Unknown'
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
                WHEN 'CU' THEN 'Curveball'
                WHEN 'KC' THEN 'Curveball'
                WHEN 'CS' THEN 'Curveball'
                WHEN 'CH' THEN 'Off-speed'
                WHEN 'FS' THEN 'Off-speed'
                WHEN 'FO' THEN 'Off-speed'
                WHEN 'SC' THEN 'Other'
                WHEN 'KN' THEN 'Other'
                WHEN 'EP' THEN 'Other'
                WHEN 'UN' THEN 'Unknown'
                WHEN 'XX' THEN 'Unknown'
                WHEN 'PO' THEN 'Unknown'
                ELSE NULL
            END
    """)
    
    # Commit the changes
    conn.commit()
    
    # Close the connection
    conn.close()

if __name__ == "__main__":
    update_advanced_metrics() 
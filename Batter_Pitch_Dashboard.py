"""
Batter Pitch Dashboard - Interactive visualization of pitches faced by a batter
"""

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np

@st.cache_data
def get_batters():
    """Get list of all batters from database with caching"""
    conn = duckdb.connect('Full_DB/mlb_statcast.db')
    batters = conn.execute("SELECT player_id, player_name FROM batters ORDER BY player_name").df()
    conn.close()
    return batters

@st.cache_data
def get_years():
    """Get list of available years from database with caching"""
    conn = duckdb.connect('Full_DB/mlb_statcast.db')
    years = conn.execute("SELECT DISTINCT game_year FROM statcast_data WHERE game_year IS NOT NULL ORDER BY game_year").df()
    conn.close()
    # Convert numpy ints to regular Python ints
    return [int(year) for year in years['game_year'].tolist()]

def get_pitcher_arms():
    """Get list of pitcher throwing arms"""
    return ['All', 'L', 'R']

def get_counts():
    """Get list of available ball-strike counts"""
    return ['All', '0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2']

@st.cache_data
def get_pitch_data(batter_id, year_start=None, year_end=None, pitcher_arm='All', counts=None):
    """
    Get pitch data for a specific batter with filters and caching
    """
    conn = duckdb.connect('Full_DB/mlb_statcast.db')
    
    # Build the query with filters
    query = """
        SELECT 
            s.*,
            p.player_name as pitcher_name,
            b.player_name as batter_name
        FROM statcast_data s
        JOIN pitchers p ON s.pitcher = p.player_id
        JOIN batters b ON s.batter = b.player_id
        WHERE s.batter = ?
    """
    params = [batter_id]
    
    if year_start and year_end:
        query += " AND s.game_year BETWEEN ? AND ?"
        # Convert to regular Python ints to avoid numpy.int32 issues
        params.extend([int(year_start), int(year_end)])
    
    if pitcher_arm != 'All':
        query += " AND s.p_throws = ?"
        params.append(pitcher_arm)
    
    if counts and 'All' not in counts:
        # Build count conditions
        count_conditions = []
        for count in counts:
            balls, strikes = count.split('-')
            count_conditions.append(f"(s.balls = {int(balls)} AND s.strikes = {int(strikes)})")
        
        if count_conditions:
            query += f" AND ({' OR '.join(count_conditions)})"
    
    # Add pitch count filter
    query += """
        AND s.plate_x IS NOT NULL 
        AND s.plate_z IS NOT NULL
        AND s.pitch_subtype IS NOT NULL
    """
    
    df = conn.execute(query, params).df()
    conn.close()
    
    return df

def calculate_strike_zone(df):
    """
    Calculate batter-specific strike zone boundaries
    """
    if df.empty:
        return None, None, None, None
    
    # Get unique batters (should be 1 in this case)
    batters = df['batter'].unique()
    if len(batters) == 0:
        return None, None, None, None
    
    batter_id = batters[0]
    
    # Calculate average strike zone for this batter
    avg_sz_top = df['sz_top'].mean()
    avg_sz_bot = df['sz_bot'].mean()
    
    # Calculate zone boundaries using the same logic as Statcast_Table_Alter.py
    zone_left = -10/12  # -0.833 feet
    zone_right = 10/12  # 0.833 feet
    zone_bottom = avg_sz_bot - 3/12 - (avg_sz_bot - 1.69)/4
    zone_top = avg_sz_top - (avg_sz_top - 3.5)/3
    
    return zone_left, zone_right, zone_bottom, zone_top

def create_pitch_plot(df, zone_boundaries):
    """
    Create the main pitch location plot
    """
    if df.empty:
        return go.Figure()
    
    # Color mapping for pitch subtypes
    color_map = {
        'Fastball': '#FF6B6B',
        'Sinker': '#4ECDC4', 
        'Cutter': '#45B7D1',
        'Slider': '#96CEB4',
        'Curveball': '#FFEAA7',
        'Off-speed': '#DDA0DD',
        'Other': '#A8A8A8'
    }
    
    fig = go.Figure()
    
    # Add pitches by subtype
    for subtype in df['pitch_subtype'].unique():
        subset = df[df['pitch_subtype'] == subtype]
        color = color_map.get(subtype, '#A8A8A8')
        
        fig.add_trace(go.Scatter(
            x=subset['plate_x'],
            y=subset['plate_z'],
            mode='markers',
            name=subtype,
            marker=dict(
                color=color,
                size=8,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=[f"{row['pitcher_name']}<br>{row['pitch_type']}<br>{row['release_speed']:.1f} mph" 
                  for _, row in subset.iterrows()],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    
    # Add strike zone if boundaries are available
    if zone_boundaries:
        zone_left, zone_right, zone_bottom, zone_top = zone_boundaries
        
        # Strike zone rectangle
        fig.add_shape(
            type="rect",
            x0=zone_left, y0=zone_bottom,
            x1=zone_right, y1=zone_top,
            line=dict(color="red", width=2),
            fillcolor="rgba(255,0,0,0.1)",
            name="Strike Zone"
        )
    
    # Add home plate outline
    home_plate_x = [-0.708, 0.708, 0.708, 0.354, -0.354, -0.708]
    home_plate_z = [0, 0, 1.5, 2.5, 2.5, 1.5]
    
    fig.add_trace(go.Scatter(
        x=home_plate_x,
        y=home_plate_z,
        mode='lines',
        line=dict(color='black', width=2),
        name='Home Plate',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title="Pitch Locations",
        xaxis_title="Horizontal Location (feet)",
        yaxis_title="Vertical Location (feet)",
        xaxis=dict(range=[-2, 2]),
        yaxis=dict(range=[0, 5]),
        width=800,
        height=600,
        showlegend=True
    )
    
    return fig

def create_advanced_summary(df):
    """
    Create advanced summary statistics including IZONE/OZONE rates and HA_Adjusted xISO
    """
    if df.empty:
        return pd.DataFrame()
    
    # Define zone and swing events
    def is_swing(description):
        return description in ['swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip']
    
    def is_whiff(description):
        return description in ['swinging_strike', 'swinging_strike_blocked']
    
    def is_foul(description):
        return description == 'foul'
    
    # Add derived columns
    df_copy = df.copy()
    df_copy['is_swing'] = df_copy['description'].apply(is_swing)
    df_copy['is_whiff'] = df_copy['description'].apply(is_whiff)
    df_copy['is_foul'] = df_copy['description'].apply(is_foul)
    df_copy['is_hit_in_play'] = (df_copy['description'].fillna('') == 'hit_into_play').astype(int)
    
    # Filter out "Other" pitch subtype for cleaner analysis
    df_copy = df_copy[df_copy['pitch_subtype'] != 'Other']
    
    # Group by pitch subtype and zone
    summary = df_copy.groupby(['pitch_subtype', 'is_zone']).agg({
        'release_speed': ['count', 'mean'],
        'is_swing': 'sum',
        'is_whiff': 'sum',
        'is_foul': 'sum',
        'is_hit_in_play': 'sum',
        'HA_Adj_estimated_xISO': 'mean'
    }).round(3)
    
    # Flatten column names
    summary.columns = ['Count', 'Avg_Speed', 'Swings', 'Whiffs', 'Fouls', 'Hit_in_Play', 'HA_Adj_xISO']
    
    # Calculate rates
    summary['Swing_Rate'] = (summary['Swings'] / summary['Count'] * 100).round(1)
    summary['Whiff_Rate_of_Swings'] = summary.apply(
        lambda row: (row['Whiffs'] / row['Swings'] * 100).round(1) if row['Swings'] > 0 else 0, axis=1
    )
    summary['Foul_Rate_of_Swings'] = summary.apply(
        lambda row: (row['Fouls'] / row['Swings'] * 100).round(1) if row['Swings'] > 0 else 0, axis=1
    )
    
    # Reset index to make zone a column
    summary = summary.reset_index()
    summary['Zone'] = summary['is_zone'].map({1: 'IZONE', 0: 'OZONE'})
    
    # Reorder columns
    summary = summary[['pitch_subtype', 'Zone', 'Count', 'Swings', 'Hit_in_Play', 'Avg_Speed', 'Swing_Rate', 
                      'Whiff_Rate_of_Swings', 'Foul_Rate_of_Swings', 'HA_Adj_xISO']]
    
    # Add IZONE and OZONE summary rows
    for zone_val, zone_name in [(1, 'IZONE'), (0, 'OZONE')]:
        zone_df = df_copy[df_copy['is_zone'] == zone_val]
        if not zone_df.empty:
            count = len(zone_df)
            swings = zone_df['is_swing'].sum()
            hit_in_play = zone_df['is_hit_in_play'].sum()
            avg_speed = zone_df['release_speed'].mean()
            whiffs = zone_df['is_whiff'].sum()
            fouls = zone_df['is_foul'].sum()
            ha_adj_xiso = zone_df['HA_Adj_estimated_xISO'].mean()
            swing_rate = (swings / count * 100) if count > 0 else 0
            whiff_rate = (whiffs / swings * 100) if swings > 0 else 0
            foul_rate = (fouls / swings * 100) if swings > 0 else 0

            total_row = pd.DataFrame({
                'pitch_subtype': [f'**{zone_name} (ALL)**'],
                'Zone': [zone_name],
                'Count': [count],
                'Swings': [swings],
                'Hit_in_Play': [hit_in_play],
                'Avg_Speed': [round(avg_speed, 3)],
                'Swing_Rate': [round(swing_rate, 1)],
                'Whiff_Rate_of_Swings': [round(whiff_rate, 1)],
                'Foul_Rate_of_Swings': [round(foul_rate, 1)],
                'HA_Adj_xISO': [round(ha_adj_xiso, 3)]
            })
            summary = pd.concat([summary, total_row], ignore_index=True)

    return summary

def create_pitch_summary(df):
    """
    Create basic summary statistics for pitches
    """
    if df.empty:
        return pd.DataFrame()
    
    summary = df.groupby('pitch_subtype').agg({
        'release_speed': ['count', 'mean', 'std'],
        'is_zone': 'mean',
        'events': lambda x: (x == 'hit_into_play').sum()
    }).round(2)
    
    summary.columns = ['Count', 'Avg_Speed', 'Speed_Std', 'Zone_Rate', 'Balls_in_Play']
    summary['Zone_Rate'] = (summary['Zone_Rate'] * 100).round(1)
    
    return summary

def create_speed_distribution(df):
    """
    Create speed distribution plot
    """
    if df.empty:
        return go.Figure()
    
    fig = px.histogram(
        df, 
        x='release_speed', 
        color='pitch_subtype',
        nbins=20,
        title="Pitch Speed Distribution",
        labels={'release_speed': 'Release Speed (mph)', 'count': 'Count'}
    )
    
    fig.update_layout(
        width=800,
        height=400,
        showlegend=True
    )
    
    return fig

def main():
    st.set_page_config(page_title="Batter Pitch Dashboard", layout="wide")
    
    st.title("🏟️ Batter Pitch Dashboard")
    st.markdown("Analyze pitches faced by a specific batter with interactive filters")
    
    # Sidebar controls
    st.sidebar.header("Filters")
    
    # Get data for dropdowns with caching
    batters = get_batters()
    years = get_years()
    pitcher_arms = get_pitcher_arms()
    counts = get_counts()
    
    # Batter selection with search
    st.sidebar.markdown("**Select Batter:**")
    search_term = st.sidebar.text_input("Search for batter:", placeholder="Type to search...")
    
    if search_term:
        filtered_batters = batters[batters['player_name'].str.contains(search_term, case=False, na=False)]
    else:
        filtered_batters = batters
    
    if len(filtered_batters) > 0:
        selected_batter = st.sidebar.selectbox(
            "Choose batter:",
            options=filtered_batters['player_name'].tolist(),
            index=0,
            key="batter_select"
        )
    else:
        st.sidebar.warning("No batters found matching your search.")
        return
    
    # Get batter ID
    batter_id = int(batters[batters['player_name'] == selected_batter]['player_id'].iloc[0])
    
    # Year filter
    st.sidebar.markdown("**Year Range:**")
    year_start = st.sidebar.selectbox(
        "Start Year:",
        options=[None] + years,
        index=0,
        format_func=lambda x: "All Years" if x is None else str(x)
    )
    
    year_end = st.sidebar.selectbox(
        "End Year:",
        options=[None] + years,
        index=0,
        format_func=lambda x: "All Years" if x is None else str(x)
    )
    
    # Pitcher arm filter
    selected_arm = st.sidebar.selectbox(
        "Pitcher Throwing Arm:",
        options=pitcher_arms,
        index=0
    )
    
    # Count filter
    st.sidebar.markdown("**Ball-Strike Counts:**")
    selected_counts = st.sidebar.multiselect(
        "Select Counts:",
        options=counts,
        default=['All']
    )
    
    # Handle year range logic
    if year_start is None and year_end is None:
        year_start = None
        year_end = None
    elif year_start is None:
        year_start = year_end
    elif year_end is None:
        year_end = year_start
    elif year_start > year_end:
        # Swap if start is after end
        year_start, year_end = year_end, year_start
    
    # Get filtered data with caching
    df = get_pitch_data(batter_id, year_start, year_end, selected_arm, selected_counts)
    
    if df.empty:
        st.warning("No data found for the selected filters. Try adjusting your criteria.")
        return
    
    # Calculate strike zone
    zone_boundaries = calculate_strike_zone(df)
    
    # Display summary stats
    st.header(f"📊 {selected_batter} - Pitch Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pitches", len(df))
    with col2:
        st.metric("Unique Pitchers", df['pitcher'].nunique())
    with col3:
        st.metric("Pitch Types", df['pitch_subtype'].nunique())
    with col4:
        avg_speed = df['release_speed'].mean()
        st.metric("Avg Speed", f"{avg_speed:.1f} mph")
    
    # Create plots
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Pitch Locations")
        pitch_plot = create_pitch_plot(df, zone_boundaries)
        st.plotly_chart(pitch_plot, use_container_width=True)
    
    with col2:
        st.subheader("Basic Summary")
        summary = create_pitch_summary(df)
        st.dataframe(summary, use_container_width=True)
    
    # Advanced analytics
    st.subheader("🎯 Advanced Analytics - IZONE/OZONE Performance")
    advanced_summary = create_advanced_summary(df)
    st.dataframe(advanced_summary, use_container_width=True)
    
    # Speed distribution
    st.subheader("Speed Distribution")
    speed_plot = create_speed_distribution(df)
    st.plotly_chart(speed_plot, use_container_width=True)
    
    # Raw data table
    st.subheader("Raw Data")
    display_cols = ['game_date', 'pitcher_name', 'pitch_type', 'pitch_subtype', 
                   'release_speed', 'plate_x', 'plate_z', 'is_zone', 'events', 'HA_Adj_estimated_xISO']
    st.dataframe(df[display_cols].head(100), use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name=f"{selected_batter.replace(' ', '_')}_pitches.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main() 
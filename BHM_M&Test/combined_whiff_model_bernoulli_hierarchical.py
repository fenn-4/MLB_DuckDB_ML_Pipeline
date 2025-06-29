import duckdb
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# --- Date Range Constants ---
QUALIFYING_YEAR = 2025
TRAINING_START_DATE = '2024-04-01'
TRAINING_END_DATE = '2025-06-21'

def load_and_preprocess_data(db_path):
    """
    Loads and preprocesses data from the DuckDB database for the hierarchical Bernoulli model.
    """
    con = duckdb.connect(db_path, read_only=True)
    
    query = f"""
    WITH qualified_batters AS (
        SELECT batter
        FROM statcast_major
        WHERE game_year = {QUALIFYING_YEAR}
          AND description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score', 'foul_bunt')
        GROUP BY batter
        HAVING COUNT(*) > 100
    ),
    qualified_pitchers AS (
        SELECT pitcher
        FROM statcast_major
        WHERE game_year = {QUALIFYING_YEAR}
          AND description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score', 'foul_bunt')
        GROUP BY pitcher
        HAVING COUNT(*) > 100
    ),
    combined_data AS (
        SELECT batter, pitcher, stand, pitch_subtype, is_zone, zone, p_throws, description, game_date, balls, strikes, 'major' as league
        FROM statcast_major
        WHERE batter IN (SELECT batter FROM qualified_batters)
          AND pitcher IN (SELECT pitcher FROM qualified_pitchers)
        
        UNION ALL

        SELECT batter, pitcher, stand, pitch_subtype, is_zone, zone, p_throws, description, game_date, balls, strikes, 'minor' as league
        FROM statcast_minor
        WHERE batter IN (SELECT batter FROM qualified_batters)
          AND pitcher IN (SELECT pitcher FROM qualified_pitchers)
    )
    SELECT
        batter,
        pitcher,
        stand,
        pitch_subtype,
        is_zone,
        zone,
        p_throws,
        balls,
        strikes,
        league,
        CASE WHEN description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END AS is_whiff
    FROM combined_data
    WHERE 
        game_date BETWEEN '{TRAINING_START_DATE}' AND '{TRAINING_END_DATE}'
        AND description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score', 'foul_bunt')
        AND pitch_subtype IS NOT NULL 
        AND pitch_subtype != 'Other'
        AND is_zone IS NOT NULL
        AND zone IS NOT NULL
        AND zone IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14)
        AND p_throws IS NOT NULL
        AND stand IS NOT NULL
        AND pitcher IS NOT NULL
        AND balls IS NOT NULL
        AND strikes IS NOT NULL;
    """
    
    data = con.execute(query).fetchdf()
    con.close()
    
    # Calculate batter whiff rates using MLB data only and create clusters
    mlb_data = data[data['league'] == 'major']
    batter_whiff_rates = mlb_data.groupby('batter')['is_whiff'].mean().reset_index()
    batter_whiff_rates['whiff_group_3'] = pd.qcut(batter_whiff_rates['is_whiff'], q=3, labels=['Low', 'Medium', 'High'])
    batter_whiff_rates['whiff_group_6'] = pd.qcut(batter_whiff_rates['is_whiff'], q=6, labels=['Very_Low', 'Low', 'Medium', 'Med_High', 'High', 'Extreme'])
    
    # Calculate pitcher whiff rates using MLB data only and create clusters
    pitcher_whiff_rates = mlb_data.groupby('pitcher')['is_whiff'].mean().reset_index()
    pitcher_whiff_rates['pitcher_whiff_group_3'] = pd.qcut(pitcher_whiff_rates['is_whiff'], q=3, labels=['Low', 'Medium', 'High'])
    pitcher_whiff_rates['pitcher_whiff_group_6'] = pd.qcut(pitcher_whiff_rates['is_whiff'], q=6, labels=['Very_Low', 'Low', 'Medium', 'Med_High', 'High', 'Extreme'])
    
    # Merge back to main data
    data = data.merge(batter_whiff_rates[['batter', 'whiff_group_3', 'whiff_group_6']], on='batter', how='left')
    data = data.merge(pitcher_whiff_rates[['pitcher', 'pitcher_whiff_group_3', 'pitcher_whiff_group_6']], on='pitcher', how='left')
    
    # Create count as a categorical variable
    data['count'] = data['balls'].astype(str) + '-' + data['strikes'].astype(str)
    # Map to buckets
    neutral = {'0-0', '1-1', '0-1', '1-0'}
    at_risk = {'0-2', '1-2', '2-2', '3-2'}
    ahead = {'2-0', '2-1', '3-1'}
    
    data['count_bucket'] = data['count'].apply(lambda x: 
        'neutral' if x in neutral else
        'at_risk' if x in at_risk else
        'ahead' if x in ahead else 'neutral'
    )
    
    encoders = {}
    for col in ['batter', 'pitcher', 'stand', 'pitch_subtype', 'p_throws', 'league', 'whiff_group_3', 'whiff_group_6', 'pitcher_whiff_group_3', 'pitcher_whiff_group_6']:
        le = LabelEncoder()
        data[col + '_idx'] = le.fit_transform(data[col])
        encoders[col] = le
        
    data['zone_idx'] = data['is_zone'].astype(int)
    
    # Create zone number index (1-9, 11-14 -> 0-12)
    zone_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 11: 9, 12: 10, 13: 11, 14: 12}
    data['zone_num_idx'] = data['zone'].map(zone_mapping)

    return data, encoders

def build_and_run_hierarchical_model(data, encoders):
    """
    Builds and runs a hierarchical Bernoulli model with group-to-batter and group-to-pitcher hierarchy.
    """
    coords = {
        'batter': encoders['batter'].classes_,
        'pitcher': encoders['pitcher'].classes_,
        'pitch_subtype': encoders['pitch_subtype'].classes_,
        'zone': ['out_of_zone', 'in_zone'],
        'zone_num': list(range(13)),  # 0-12 for zones 1-9, 11-14
        'p_throws': encoders['p_throws'].classes_,
        'stand': encoders['stand'].classes_,
        'league': encoders['league'].classes_,
        'whiff_group_3': encoders['whiff_group_3'].classes_,
        'whiff_group_6': encoders['whiff_group_6'].classes_,
        'pitcher_whiff_group_3': encoders['pitcher_whiff_group_3'].classes_,
        'pitcher_whiff_group_6': encoders['pitcher_whiff_group_6'].classes_,
    }
    
    with pm.Model(coords=coords) as model:
        batter_idx = pm.Data('batter_idx', data['batter_idx'].values, mutable=True)
        pitcher_idx = pm.Data('pitcher_idx', data['pitcher_idx'].values, mutable=True)
        pitch_subtype_idx = pm.Data('pitch_subtype_idx', data['pitch_subtype_idx'].values, mutable=True)
        zone_idx = pm.Data('zone_idx', data['zone_idx'].values, mutable=True)
        zone_num_idx = pm.Data('zone_num_idx', data['zone_num_idx'].values, mutable=True)
        p_throws_idx = pm.Data('p_throws_idx', data['p_throws_idx'].values, mutable=True)
        stand_idx = pm.Data('stand_idx', data['stand_idx'].values, mutable=True)
        league_idx = pm.Data('league_idx', data['league_idx'].values, mutable=True)
        whiff_group_3_idx = pm.Data('whiff_group_3_idx', data['whiff_group_3_idx'].values, mutable=True)
        whiff_group_6_idx = pm.Data('whiff_group_6_idx', data['whiff_group_6_idx'].values, mutable=True)
        pitcher_whiff_group_3_idx = pm.Data('pitcher_whiff_group_3_idx', data['pitcher_whiff_group_3_idx'].values, mutable=True)
        pitcher_whiff_group_6_idx = pm.Data('pitcher_whiff_group_6_idx', data['pitcher_whiff_group_6_idx'].values, mutable=True)
        is_whiff = pm.Data('is_whiff', data['is_whiff'].values, mutable=True)

        alpha_global = pm.Normal('alpha_global', mu=0, sigma=1.5)
        
        # --- HIERARCHICAL BATTER EFFECTS ---
        
        # Group-level 3-way effects (population-level patterns) - uniform priors
        sigma_group_zone_subtype = pm.HalfNormal('sigma_group_zone_subtype', sigma=1.0)  # Uniform prior
        theta_group_zone_subtype_offset = pm.Normal('theta_group_zone_subtype_offset', mu=0, sigma=1, dims=('whiff_group_6', 'zone', 'pitch_subtype'))
        theta_group_zone_subtype = pm.Deterministic('theta_group_zone_subtype', theta_group_zone_subtype_offset * sigma_group_zone_subtype, dims=('whiff_group_6', 'zone', 'pitch_subtype'))
        
        # Individual deviations from group (hierarchical) - uniform priors
        sigma_batter_zone_subtype_deviation = pm.HalfNormal('sigma_batter_zone_subtype_deviation', sigma=1.0)  # Uniform prior
        batter_zone_subtype_deviation_offset = pm.Normal('batter_zone_subtype_deviation_offset', mu=0, sigma=1, dims=('batter', 'zone', 'pitch_subtype'))
        batter_zone_subtype_deviation = pm.Deterministic('batter_zone_subtype_deviation', batter_zone_subtype_deviation_offset * sigma_batter_zone_subtype_deviation, dims=('batter', 'zone', 'pitch_subtype'))
        
        # Non-hierarchical batter effects (pitcher throwing hand) - uniform prior
        sigma_batter_p_throws = pm.HalfNormal('sigma_batter_p_throws', sigma=1.0)  # Uniform prior
        eta_batter_p_throws_offset = pm.Normal('eta_batter_p_throws_offset', mu=0, sigma=1, dims=('batter', 'p_throws'))
        eta_batter_p_throws = pm.Deterministic('eta_batter_p_throws', eta_batter_p_throws_offset * sigma_batter_p_throws, dims=('batter', 'p_throws'))

        # --- HIERARCHICAL LEAGUE EFFECTS ---
        
        # Whiff group × league effects (batter) - uniform priors
        sigma_whiff_group_league = pm.HalfNormal('sigma_whiff_group_league', sigma=1.0)  # Uniform prior
        theta_whiff_group_league_offset = pm.Normal('theta_whiff_group_league_offset', mu=0, sigma=1, dims=('whiff_group_6', 'league'))
        theta_whiff_group_league = pm.Deterministic('theta_whiff_group_league', theta_whiff_group_league_offset * sigma_whiff_group_league, dims=('whiff_group_6', 'league'))
        
        # Individual batter deviations from whiff group × league effects - uniform prior
        sigma_batter_league_deviation = pm.HalfNormal('sigma_batter_league_deviation', sigma=1.0)  # Uniform prior
        batter_league_deviation_offset = pm.Normal('batter_league_deviation_offset', mu=0, sigma=1, dims=('batter', 'league'))
        batter_league_deviation = pm.Deterministic('batter_league_deviation', batter_league_deviation_offset * sigma_batter_league_deviation, dims=('batter', 'league'))
        
        # --- HIERARCHICAL PITCHER EFFECTS ---
        
        # Group-level 3-way effects (population-level patterns) - uniform priors
        sigma_pitcher_group_zone_subtype = pm.HalfNormal('sigma_pitcher_group_zone_subtype', sigma=1.0)  # Uniform prior
        theta_pitcher_group_zone_subtype_offset = pm.Normal('theta_pitcher_group_zone_subtype_offset', mu=0, sigma=1, dims=('pitcher_whiff_group_6', 'zone', 'pitch_subtype'))
        theta_pitcher_group_zone_subtype = pm.Deterministic('theta_pitcher_group_zone_subtype', theta_pitcher_group_zone_subtype_offset * sigma_pitcher_group_zone_subtype, dims=('pitcher_whiff_group_6', 'zone', 'pitch_subtype'))
        
        # Individual deviations from group (hierarchical) - uniform priors
        sigma_pitcher_zone_subtype_deviation = pm.HalfNormal('sigma_pitcher_zone_subtype_deviation', sigma=1.0)  # Uniform prior
        pitcher_zone_subtype_deviation_offset = pm.Normal('pitcher_zone_subtype_deviation_offset', mu=0, sigma=1, dims=('pitcher', 'zone', 'pitch_subtype'))
        pitcher_zone_subtype_deviation = pm.Deterministic('pitcher_zone_subtype_deviation', pitcher_zone_subtype_deviation_offset * sigma_pitcher_zone_subtype_deviation, dims=('pitcher', 'zone', 'pitch_subtype'))
        
        # Non-hierarchical pitcher effects (batter stance) - uniform prior
        sigma_pitcher_stand = pm.HalfNormal('sigma_pitcher_stand', sigma=1.0)  # Uniform prior
        eta_pitcher_stand_offset = pm.Normal('eta_pitcher_stand_offset', mu=0, sigma=1, dims=('pitcher', 'stand'))
        eta_pitcher_stand = pm.Deterministic('eta_pitcher_stand', eta_pitcher_stand_offset * sigma_pitcher_stand, dims=('pitcher', 'stand'))

        # --- HIERARCHICAL LEAGUE EFFECTS ---
        
        # Whiff group × league effects (pitcher) - uniform priors
        sigma_pitcher_whiff_group_league = pm.HalfNormal('sigma_pitcher_whiff_group_league', sigma=1.0)  # Uniform prior
        theta_pitcher_whiff_group_league_offset = pm.Normal('theta_pitcher_whiff_group_league_offset', mu=0, sigma=1, dims=('pitcher_whiff_group_6', 'league'))
        theta_pitcher_whiff_group_league = pm.Deterministic('theta_pitcher_whiff_group_league', theta_pitcher_whiff_group_league_offset * sigma_pitcher_whiff_group_league, dims=('pitcher_whiff_group_6', 'league'))
        
        # Individual pitcher deviations from whiff group × league effects - uniform prior
        sigma_pitcher_league_deviation = pm.HalfNormal('sigma_pitcher_league_deviation', sigma=1.0)  # Uniform prior
        pitcher_league_deviation_offset = pm.Normal('pitcher_league_deviation_offset', mu=0, sigma=1, dims=('pitcher', 'league'))
        pitcher_league_deviation = pm.Deterministic('pitcher_league_deviation', pitcher_league_deviation_offset * sigma_pitcher_league_deviation, dims=('pitcher', 'league'))

        # --- GROUP-LEVEL EFFECTS (3-way interactions) ---
        sigma_whiff_group_zone_pitch = pm.HalfNormal('sigma_whiff_group_zone_pitch', sigma=1.0)  # Uniform prior
        theta_whiff_group_zone_pitch_offset = pm.Normal('theta_whiff_group_zone_pitch_offset', mu=0, sigma=1, dims=('zone_num', 'pitch_subtype'))
        theta_whiff_group_zone_pitch = pm.Deterministic('theta_whiff_group_zone_pitch', theta_whiff_group_zone_pitch_offset * sigma_whiff_group_zone_pitch, dims=('zone_num', 'pitch_subtype'))

        # 4-way interaction: whiff_group × pitcher_whiff_group × zone × pitch_subtype - uniform prior
        sigma_whiff_pitcher_group_zone_subtype = pm.HalfNormal('sigma_whiff_pitcher_group_zone_subtype', sigma=1.0)  # Uniform prior
        kappa_whiff_pitcher_group_zone_subtype_offset = pm.Normal('kappa_whiff_pitcher_group_zone_subtype_offset', mu=0, sigma=1, dims=('whiff_group_6', 'pitcher_whiff_group_6', 'zone', 'pitch_subtype'))
        kappa_whiff_pitcher_group_zone_subtype = pm.Deterministic('kappa_whiff_pitcher_group_zone_subtype', kappa_whiff_pitcher_group_zone_subtype_offset * sigma_whiff_pitcher_group_zone_subtype, dims=('whiff_group_6', 'pitcher_whiff_group_6', 'zone', 'pitch_subtype'))

        # --- HIERARCHICAL LOGIT ---
        # Get group indices for each batter and pitcher
        batter_to_group = data.groupby('batter')['whiff_group_6_idx'].first().values
        pitcher_to_group = data.groupby('pitcher')['pitcher_whiff_group_6_idx'].first().values
        batter_group_indices = pm.Data('batter_group_indices', batter_to_group, mutable=True)
        pitcher_group_indices = pm.Data('pitcher_group_indices', pitcher_to_group, mutable=True)
        
        logit_p = (
            alpha_global +
            # Hierarchical league effects = whiff group effect + individual deviation
            (theta_whiff_group_league[batter_group_indices[batter_idx], league_idx] + batter_league_deviation[batter_idx, league_idx]) +
            (theta_pitcher_whiff_group_league[pitcher_group_indices[pitcher_idx], league_idx] + pitcher_league_deviation[pitcher_idx, league_idx]) +
            # Hierarchical batter effects = group effect + individual deviation
            (theta_group_zone_subtype[batter_group_indices[batter_idx], zone_idx, pitch_subtype_idx] + batter_zone_subtype_deviation[batter_idx, zone_idx, pitch_subtype_idx]) +
            # Non-hierarchical batter effects
            eta_batter_p_throws[batter_idx, p_throws_idx] +
            # Hierarchical pitcher effects = group effect + individual deviation
            (theta_pitcher_group_zone_subtype[pitcher_group_indices[pitcher_idx], zone_idx, pitch_subtype_idx] + pitcher_zone_subtype_deviation[pitcher_idx, zone_idx, pitch_subtype_idx]) +
            # Non-hierarchical pitcher effects
            eta_pitcher_stand[pitcher_idx, stand_idx] +
            # Group-level effects
            theta_whiff_group_zone_pitch[zone_num_idx, pitch_subtype_idx] +
            kappa_whiff_pitcher_group_zone_subtype[whiff_group_6_idx, pitcher_whiff_group_6_idx, zone_idx, pitch_subtype_idx]
        )
        
        pm.Bernoulli('whiffs', p=pm.math.invlogit(logit_p), observed=is_whiff)

        trace = pm.sample(2000, tune=1000, chains=4, cores=4, target_accept=.8, max_treedepth=14, random_seed=42)
        
    return trace

def save_artifacts(trace, encoders, data):
    """
    Saves the model trace and other artifacts for the hierarchical Bernoulli model.
    """
    trace.to_netcdf("Model_Weights/combined_whiff_model_bernoulli_hierarchical_trace.nc")
    
    coords_to_save = {
        'batter': encoders['batter'].classes_,
        'pitcher': encoders['pitcher'].classes_,
        'pitch_subtype': encoders['pitch_subtype'].classes_,
        'zone': ['out_of_zone', 'in_zone'],
        'zone_num': list(range(13)),  # 0-12 for zones 1-9, 11-14
        'p_throws': encoders['p_throws'].classes_,
        'stand': encoders['stand'].classes_,
        'league': encoders['league'].classes_,
        'whiff_group_3': encoders['whiff_group_3'].classes_,
        'whiff_group_6': encoders['whiff_group_6'].classes_,
        'pitcher_whiff_group_3': encoders['pitcher_whiff_group_3'].classes_,
        'pitcher_whiff_group_6': encoders['pitcher_whiff_group_6'].classes_,
    }
    
    artifacts = {
        'encoders': encoders,
        'coords': coords_to_save,
        'data_columns': data.columns
    }
    with open("Model_Weights/combined_whiff_model_bernoulli_hierarchical_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
        
    print("Hierarchical Bernoulli model trace and artifacts saved successfully.")

if __name__ == "__main__":
    DB_PATH = 'Full_DB/mlb_statcast.db'
    
    df, model_encoders = load_and_preprocess_data(DB_PATH)
    
    print("Building and running hierarchical Bernoulli model...")
    model_trace = build_and_run_hierarchical_model(df, model_encoders)
    
    print("Saving model artifacts...")
    save_artifacts(model_trace, model_encoders, df)
    
    print("\nModel fitting complete.")
    summary = az.summary(model_trace, var_names=[
        'alpha_global',
        # Batter hierarchical effects
        'sigma_group_zone_subtype', 'sigma_batter_zone_subtype_deviation',
        'sigma_batter_p_throws',
        # Batter league effects
        'sigma_whiff_group_league', 'sigma_batter_league_deviation',
        # Pitcher hierarchical effects
        'sigma_pitcher_group_zone_subtype', 'sigma_pitcher_zone_subtype_deviation',
        'sigma_pitcher_stand',
        # Pitcher league effects
        'sigma_pitcher_whiff_group_league', 'sigma_pitcher_league_deviation',
        # Group-level effects
        'sigma_whiff_group_zone_pitch', 'sigma_whiff_pitcher_group_zone_subtype'
    ])
    print(summary) 
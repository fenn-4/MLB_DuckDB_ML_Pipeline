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
TRAINING_END_DATE = '2025-6-21'

def load_and_preprocess_data(db_path):
    """
    Loads and preprocesses data from the DuckDB database.
    """
    con = duckdb.connect(db_path, read_only=True)
    
    query = f"""
    WITH qualified_batters AS (
        SELECT batter
        FROM statcast_major
        WHERE game_year = {QUALIFYING_YEAR}
          AND description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score', 'foul_bunt')
        GROUP BY batter
        HAVING COUNT(*) > 50
    ),
    qualified_pitchers AS (
        SELECT pitcher
        FROM statcast_major
        WHERE game_year = {QUALIFYING_YEAR}
          AND description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score', 'foul_bunt')
        GROUP BY pitcher
        HAVING COUNT(*) > 50
    ),
    combined_data AS (
        SELECT batter, pitcher, stand, pitch_subtype, is_zone, p_throws, description, game_date 
        FROM statcast_major
        WHERE batter IN (SELECT batter FROM qualified_batters)
          AND pitcher IN (SELECT pitcher FROM qualified_pitchers)
        
        UNION ALL

        SELECT batter, pitcher, stand, pitch_subtype, is_zone, p_throws, description, game_date 
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
        p_throws,
        COUNT(*) AS n_swings,
        SUM(CASE WHEN description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS n_whiffs
    FROM combined_data
    WHERE 
        game_date BETWEEN '{TRAINING_START_DATE}' AND '{TRAINING_END_DATE}'
        AND description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score', 'foul_bunt')
        AND pitch_subtype IS NOT NULL 
        AND is_zone IS NOT NULL
        AND p_throws IS NOT NULL
        AND stand IS NOT NULL
        AND pitcher IS NOT NULL
    GROUP BY batter, pitcher, stand, pitch_subtype, is_zone, p_throws
    HAVING COUNT(*) >= 1;
    """
    
    data = con.execute(query).fetchdf()
    con.close()
    
    encoders = {}
    for col in ['batter', 'pitcher', 'stand', 'pitch_subtype', 'p_throws']:
        le = LabelEncoder()
        data[col + '_idx'] = le.fit_transform(data[col])
        encoders[col] = le
        
    data['zone_idx'] = data['is_zone'].astype(int)

    return data, encoders

def build_and_run_model(data, encoders):
    """
    Builds and runs a Hierarchical Binomial model with symmetric batter and pitcher effects.
    """
    coords = {
        'batter': encoders['batter'].classes_,
        'pitcher': encoders['pitcher'].classes_,
        'pitch_subtype': encoders['pitch_subtype'].classes_,
        'zone': ['out_of_zone', 'in_zone'],
        'p_throws': encoders['p_throws'].classes_,
        'stand': encoders['stand'].classes_
    }
    
    with pm.Model(coords=coords) as model:
        batter_idx = pm.Data('batter_idx', data['batter_idx'].values, mutable=True)
        pitcher_idx = pm.Data('pitcher_idx', data['pitcher_idx'].values, mutable=True)
        pitch_subtype_idx = pm.Data('pitch_subtype_idx', data['pitch_subtype_idx'].values, mutable=True)
        zone_idx = pm.Data('zone_idx', data['zone_idx'].values, mutable=True)
        p_throws_idx = pm.Data('p_throws_idx', data['p_throws_idx'].values, mutable=True)
        stand_idx = pm.Data('stand_idx', data['stand_idx'].values, mutable=True)
        n_swings = pm.Data('n_swings', data['n_swings'].values, mutable=True)
        n_whiffs = pm.Data('n_whiffs', data['n_whiffs'].values, mutable=True)
        
        alpha_global = pm.Normal('alpha_global', mu=0, sigma=1.5)
        
        # --- Batter Effects ---
        sigma_batter = pm.HalfNormal('sigma_batter', sigma=0.5)
        beta_batter_offset = pm.Normal('beta_batter_offset', mu=0, sigma=1, dims='batter')
        beta_batter = pm.Deterministic('beta_batter', beta_batter_offset * sigma_batter, dims='batter')
        
        sigma_batter_zone = pm.HalfNormal('sigma_batter_zone', sigma=1)
        zeta_batter_zone_offset = pm.Normal('zeta_batter_zone_offset', mu=0, sigma=1, dims=('batter', 'zone'))
        zeta_batter_zone = pm.Deterministic('zeta_batter_zone', zeta_batter_zone_offset * sigma_batter_zone, dims=('batter', 'zone'))

        sigma_batter_pitch_subtype = pm.HalfNormal('sigma_batter_pitch_subtype', sigma=1)
        gamma_batter_pitch_subtype_offset = pm.Normal('gamma_batter_pitch_subtype_offset', mu=0, sigma=1, dims=('batter', 'pitch_subtype'))
        gamma_batter_pitch_subtype = pm.Deterministic('gamma_batter_pitch_subtype', gamma_batter_pitch_subtype_offset * sigma_batter_pitch_subtype, dims=('batter', 'pitch_subtype'))
        
        sigma_batter_p_throws = pm.HalfNormal('sigma_batter_p_throws', sigma=0.5)
        eta_batter_p_throws_offset = pm.Normal('eta_batter_p_throws_offset', mu=0, sigma=1, dims=('batter', 'p_throws'))
        eta_batter_p_throws = pm.Deterministic('eta_batter_p_throws', eta_batter_p_throws_offset * sigma_batter_p_throws, dims=('batter', 'p_throws'))

        # --- Pitcher Effects (Symmetric to Batter) ---
        sigma_pitcher = pm.HalfNormal('sigma_pitcher', sigma=0.5)
        beta_pitcher_offset = pm.Normal('beta_pitcher_offset', mu=0, sigma=1, dims='pitcher')
        beta_pitcher = pm.Deterministic('beta_pitcher', beta_pitcher_offset * sigma_pitcher, dims='pitcher')
        
        sigma_pitcher_zone = pm.HalfNormal('sigma_pitcher_zone', sigma=1)
        zeta_pitcher_zone_offset = pm.Normal('zeta_pitcher_zone_offset', mu=0, sigma=1, dims=('pitcher', 'zone'))
        zeta_pitcher_zone = pm.Deterministic('zeta_pitcher_zone', zeta_pitcher_zone_offset * sigma_pitcher_zone, dims=('pitcher', 'zone'))

        sigma_pitcher_pitch_subtype = pm.HalfNormal('sigma_pitcher_pitch_subtype', sigma=1)
        gamma_pitcher_pitch_subtype_offset = pm.Normal('gamma_pitcher_pitch_subtype_offset', mu=0, sigma=1, dims=('pitcher', 'pitch_subtype'))
        gamma_pitcher_pitch_subtype = pm.Deterministic('gamma_pitcher_pitch_subtype', gamma_pitcher_pitch_subtype_offset * sigma_pitcher_pitch_subtype, dims=('pitcher', 'pitch_subtype'))
        
        sigma_pitcher_stand = pm.HalfNormal('sigma_pitcher_stand', sigma=0.5)
        eta_pitcher_stand_offset = pm.Normal('eta_pitcher_stand_offset', mu=0, sigma=1, dims=('pitcher', 'stand'))
        eta_pitcher_stand = pm.Deterministic('eta_pitcher_stand', eta_pitcher_stand_offset * sigma_pitcher_stand, dims=('pitcher', 'stand'))

        logit_p = (
            alpha_global +
            beta_batter[batter_idx] +
            zeta_batter_zone[batter_idx, zone_idx] +
            gamma_batter_pitch_subtype[batter_idx, pitch_subtype_idx] +
            eta_batter_p_throws[batter_idx, p_throws_idx] +
            beta_pitcher[pitcher_idx] +
            zeta_pitcher_zone[pitcher_idx, zone_idx] +
            gamma_pitcher_pitch_subtype[pitcher_idx, pitch_subtype_idx] +
            eta_pitcher_stand[pitcher_idx, stand_idx]
        )
        
        pm.Binomial('whiffs', n=n_swings, p=pm.math.invlogit(logit_p), observed=n_whiffs)

        trace = pm.sample(2000, tune=1000, chains=4, cores=4, target_accept=.98, max_treedepth=14, random_seed=42)
        
    return trace

def save_artifacts(trace, encoders, data):
    """
    Saves the model trace and other artifacts for the combined model.
    """
    trace.to_netcdf("Model_Weights/combined_whiff_model_trace.nc")
    
    coords_to_save = {
        'batter': encoders['batter'].classes_,
        'pitcher': encoders['pitcher'].classes_,
        'pitch_subtype': encoders['pitch_subtype'].classes_,
        'p_throws': encoders['p_throws'].classes_,
        'stand': encoders['stand'].classes_,
        'zone': ['out_of_zone', 'in_zone'] 
    }
    
    artifacts = {
        'encoders': encoders,
        'coords': coords_to_save,
        'data_columns': data.columns
    }
    with open("Model_Weights/combined_whiff_model_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
        
    print("Combined model trace and artifacts saved successfully.")

if __name__ == "__main__":
    DB_PATH = 'Full_DB/mlb_statcast.db'
    
    print("Loading and preprocessing data...")
    df, model_encoders = load_and_preprocess_data(DB_PATH)
    
    print("Building and running combined model...")
    model_trace = build_and_run_model(df, model_encoders)
    
    print("Saving model artifacts...")
    save_artifacts(model_trace, model_encoders, df)
    
    print("\nModel fitting complete.")
    summary = az.summary(model_trace, var_names=[
        'alpha_global', 
        'sigma_batter', 'sigma_batter_zone', 'sigma_batter_pitch_subtype', 'sigma_batter_p_throws',
        'sigma_pitcher', 'sigma_pitcher_zone', 'sigma_pitcher_pitch_subtype', 'sigma_pitcher_stand'
    ])
    print(summary) 
import duckdb
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# --- Date Range Constants ---
QUALIFYING_YEAR = 2025
TRAINING_START_DATE = '2024-01-01'
TRAINING_END_DATE = '2025-12-31'

def load_and_preprocess_data(db_path):
    """
    Loads and preprocesses data from the DuckDB database from the pitcher's perspective.
    
    - Identifies pitchers with > 50 MLB swings against them in QUALIFYING_YEAR.
    - Gathers all data for those pitchers from statcast_major and statcast_minor.
    - Aggregates the data for the Binomial model.
    - Encodes categorical variables for the model.
    - Returns preprocessed data and encoders.
    """
    con = duckdb.connect(db_path, read_only=True)
    
    query = f"""
    WITH qualified_pitchers AS (
        SELECT pitcher
        FROM statcast_major
        WHERE game_year = {QUALIFYING_YEAR}
          AND description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score', 'foul_bunt')
        GROUP BY pitcher
        HAVING COUNT(*) > 50
    ),
    combined_data AS (
        SELECT pitcher, pitch_subtype, is_zone, stand, description, game_date FROM statcast_major
        WHERE pitcher IN (SELECT pitcher FROM qualified_pitchers)
        UNION ALL
        SELECT pitcher, pitch_subtype, is_zone, stand, description, game_date FROM statcast_minor
        WHERE pitcher IN (SELECT pitcher FROM qualified_pitchers)
    )
    SELECT
        pitcher, pitch_subtype, is_zone, stand,
        COUNT(*) AS n_swings,
        SUM(CASE WHEN description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS n_whiffs
    FROM combined_data
    WHERE 
        game_date BETWEEN '{TRAINING_START_DATE}' AND '{TRAINING_END_DATE}'
        AND description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score', 'foul_bunt')
        AND pitch_subtype IS NOT NULL AND is_zone IS NOT NULL AND stand IS NOT NULL
    GROUP BY pitcher, pitch_subtype, is_zone, stand
    HAVING COUNT(*) >= 5; -- A balanced threshold for data inclusion and performance
    """
    
    data = con.execute(query).fetchdf()
    con.close()
    
    # Encode categorical features
    encoders = {}
    for col in ['pitcher', 'pitch_subtype', 'stand']:
        le = LabelEncoder()
        data[col + '_idx'] = le.fit_transform(data[col])
        encoders[col] = le
        
    data['zone_idx'] = data['is_zone'].astype(int)

    return data, encoders

def build_and_run_model(data, encoders):
    """
    Builds and runs the Hierarchical Binomial whiff model for pitchers.
    """
    coords = {
        'pitcher': encoders['pitcher'].classes_,
        'pitch_subtype': encoders['pitch_subtype'].classes_,
        'zone': ['out_of_zone', 'in_zone'],
        'stand': encoders['stand'].classes_
    }
    
    with pm.Model(coords=coords) as model:
        # --- Data Placeholders ---
        pitcher_idx = pm.Data('pitcher_idx', data['pitcher_idx'].values, mutable=True)
        pitch_subtype_idx = pm.Data('pitch_subtype_idx', data['pitch_subtype_idx'].values, mutable=True)
        zone_idx = pm.Data('zone_idx', data['zone_idx'].values, mutable=True)
        stand_idx = pm.Data('stand_idx', data['stand_idx'].values, mutable=True)

        n_swings = pm.Data('n_swings', data['n_swings'].values, mutable=True)
        n_whiffs = pm.Data('n_whiffs', data['n_whiffs'].values, mutable=True)
        
        # --- Priors ---
        alpha_global = pm.Normal('alpha_global', mu=0, sigma=1.5)
        
        # --- Hierarchical Effects (Non-Centered) ---
        # Main pitcher effect
        sigma_pitcher = pm.HalfNormal('sigma_pitcher', sigma=1)
        beta_pitcher_offset = pm.Normal('beta_pitcher_offset', mu=0, sigma=1, dims='pitcher')
        beta_pitcher = pm.Deterministic('beta_pitcher', beta_pitcher_offset * sigma_pitcher)

        # Pitcher vs batter handedness (stand)
        sigma_pitcher_stand = pm.HalfNormal('sigma_pitcher_stand', sigma=1)
        zeta_pitcher_stand_offset = pm.Normal('zeta_pitcher_stand_offset', mu=0, sigma=1, dims=('pitcher', 'stand'))
        zeta_pitcher_stand = pm.Deterministic('zeta_pitcher_stand', zeta_pitcher_stand_offset * sigma_pitcher_stand)

        # Pitcher by pitch type
        sigma_pitcher_pitch_subtype = pm.HalfNormal('sigma_pitcher_pitch_subtype', sigma=1)
        gamma_offset = pm.Normal('gamma_offset', mu=0, sigma=1, dims=('pitcher', 'pitch_subtype'))
        gamma_pitcher_pitch_subtype = pm.Deterministic('gamma_pitcher_pitch_subtype', gamma_offset * sigma_pitcher_pitch_subtype)

        # Pitcher by zone
        sigma_pitcher_zone = pm.HalfNormal('sigma_pitcher_zone', sigma=1)
        eta_offset = pm.Normal('eta_offset', mu=0, sigma=1, dims=('pitcher', 'zone'))
        eta_pitcher_zone = pm.Deterministic('eta_pitcher_zone', eta_offset * sigma_pitcher_zone)

        # --- Linear Model ---
        logit_p = (
            alpha_global +
            beta_pitcher[pitcher_idx] +
            zeta_pitcher_stand[pitcher_idx, stand_idx] +
            gamma_pitcher_pitch_subtype[pitcher_idx, pitch_subtype_idx] +
            eta_pitcher_zone[pitcher_idx, zone_idx]
        )
        
        # --- Likelihood ---
        pm.Binomial('whiffs', n=n_swings, p=pm.math.invlogit(logit_p), observed=n_whiffs)
        
        # --- Sampling ---
        trace = pm.sample(2000, tune=1000, chains=4, cores=4, target_accept=0.98, max_treedepth=12)
        
    return trace

def save_artifacts(trace, encoders, data):
    """
    Saves the model trace and other artifacts.
    """
    trace.to_netcdf("Model_Weights/pitcher_whiff_model_trace.nc")
    
    artifacts = {
        'encoders': encoders,
        'coords': trace.posterior.coords,
        'data_columns': data.columns
    }
    with open("Model_Weights/pitcher_whiff_model_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
        
    print("Model trace and artifacts saved successfully.")

if __name__ == "__main__":
    DB_PATH = 'Full_DB/mlb_statcast.db'
    
    print("Loading and preprocessing data...")
    df, model_encoders = load_and_preprocess_data(DB_PATH)
    
    print("Building and running model...")
    model_trace = build_and_run_model(df, model_encoders)
    
    print("Saving model artifacts...")
    save_artifacts(model_trace, model_encoders, df)
    
    print("\nModel fitting complete.")
    print(az.summary(model_trace, var_names=['alpha_global', 'sigma_pitcher', 'sigma_pitcher_stand', 'sigma_pitcher_pitch_subtype', 'sigma_pitcher_zone'])) 
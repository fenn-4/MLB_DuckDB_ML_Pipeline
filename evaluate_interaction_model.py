import duckdb
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import pickle
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --- Evaluation Date Range Constants ---
TEST_START_DATE = '2025-06-22'
TEST_END_DATE = '2025-06-24'

# --- File Paths ---
DB_PATH = 'Full_DB/mlb_statcast.db'
TRACE_PATH = "Model_Weights/routinized_whiff_interaction_model_trace.nc"
ARTIFACTS_PATH = "Model_Weights/routinized_whiff_interaction_model_artifacts.pkl"

def load_test_data(db_path, encoders):
    """
    Loads and preprocesses the test data for the interaction model.
    """
    con = duckdb.connect(db_path, read_only=True)
    
    query = f"""
    WITH combined_data AS (
        SELECT batter, pitch_subtype, is_zone, p_throws, description, game_date
        FROM statcast_major
        UNION ALL
        SELECT batter, pitch_subtype, is_zone, p_throws, description, game_date
        FROM statcast_minor
    )
    SELECT
        batter, pitch_subtype, is_zone, p_throws,
        COUNT(*) AS n_swings,
        SUM(CASE WHEN description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END) AS n_whiffs
    FROM combined_data
    WHERE 
        game_date BETWEEN '{TEST_START_DATE}' AND '{TEST_END_DATE}'
        AND description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score', 'foul_bunt')
        AND pitch_subtype IS NOT NULL AND is_zone IS NOT NULL AND p_throws IS NOT NULL
    GROUP BY batter, pitch_subtype, is_zone, p_throws
    """
    
    data = con.execute(query).fetchdf()
    con.close()
    
    for col, le in encoders.items():
        if col in data.columns:
            known_labels = set(le.classes_)
            data = data[data[col].isin(known_labels)]
            if not data.empty:
                data[col + '_idx'] = le.transform(data[col])
            
    data['zone_idx'] = data['is_zone'].astype(int)

    required_idx_cols = [col + '_idx' for col in encoders.keys()]
    data.dropna(subset=required_idx_cols, inplace=True)
    for col in required_idx_cols:
        data[col] = data[col].astype(int)

    return data

def build_model_for_prediction(coords):
    """
    Rebuilds the interaction model structure for prediction.
    """
    with pm.Model(coords=coords) as model:
        batter_idx = pm.Data('batter_idx', np.array([], dtype='int32'), mutable=True)
        pitch_subtype_idx = pm.Data('pitch_subtype_idx', np.array([], dtype='int32'), mutable=True)
        zone_idx = pm.Data('zone_idx', np.array([], dtype='int32'), mutable=True)
        p_throws_idx = pm.Data('p_throws_idx', np.array([], dtype='int32'), mutable=True)

        alpha_global = pm.Normal('alpha_global', mu=0, sigma=1.5)
        
        sigma_batter = pm.HalfNormal('sigma_batter', sigma=1)
        beta_batter_offset = pm.Normal('beta_batter_offset', mu=0, sigma=1, dims='batter')
        beta_batter = pm.Deterministic('beta_batter', beta_batter_offset * sigma_batter, dims='batter')
        
        sigma_batter_zone = pm.HalfNormal('sigma_batter_zone', sigma=1)
        zeta_batter_zone_offset = pm.Normal('zeta_batter_zone_offset', mu=0, sigma=1, dims=('batter', 'zone'))
        zeta_batter_zone = pm.Deterministic('zeta_batter_zone', zeta_batter_zone_offset * sigma_batter_zone, dims=('batter', 'zone'))

        sigma_batter_pitch_subtype = pm.HalfNormal('sigma_batter_pitch_subtype', sigma=1)
        gamma_batter_pitch_subtype_offset = pm.Normal('gamma_batter_pitch_subtype_offset', mu=0, sigma=1, dims=('batter', 'pitch_subtype'))
        gamma_batter_pitch_subtype = pm.Deterministic('gamma_batter_pitch_subtype', gamma_batter_pitch_subtype_offset * sigma_batter_pitch_subtype, dims=('batter', 'pitch_subtype'))
        
        sigma_batter_p_throws = pm.HalfNormal('sigma_batter_p_throws', sigma=1)
        eta_batter_p_throws_offset = pm.Normal('eta_batter_p_throws_offset', mu=0, sigma=1, dims=('batter', 'p_throws'))
        eta_batter_p_throws = pm.Deterministic('eta_batter_p_throws', eta_batter_p_throws_offset * sigma_batter_p_throws, dims=('batter', 'p_throws'))

        sigma_batter_zone_subtype = pm.HalfNormal('sigma_batter_zone_subtype', sigma=1)
        omega_offset = pm.Normal('omega_offset', mu=0, sigma=1, dims=('batter', 'zone', 'pitch_subtype'))
        omega_batter_zone_subtype = pm.Deterministic('omega_batter_zone_subtype', omega_offset * sigma_batter_zone_subtype, dims=('batter', 'zone', 'pitch_subtype'))

        logit_p = (
            alpha_global +
            beta_batter[batter_idx] +
            zeta_batter_zone[batter_idx, zone_idx] +
            gamma_batter_pitch_subtype[batter_idx, pitch_subtype_idx] +
            eta_batter_p_throws[batter_idx, p_throws_idx] +
            omega_batter_zone_subtype[batter_idx, zone_idx, pitch_subtype_idx]
        )
        
        pm.Deterministic('p', pm.math.invlogit(logit_p))

    return model

def main():
    print("Loading trained interaction model and artifacts...")
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    trace = az.from_netcdf(TRACE_PATH)
    
    encoders = artifacts['encoders']
    coords = artifacts['coords']

    print("Loading and preprocessing test data...")
    test_data = load_test_data(DB_PATH, encoders)
    
    if test_data.empty:
        print("No test data found for the specified date range with players from the training set. Exiting.")
        return

    print(f"Found {len(test_data)} aggregated rows in the test set.")

    prediction_model = build_model_for_prediction(coords)
    with prediction_model:
        pm.set_data({
            'batter_idx': test_data['batter_idx'].values,
            'pitch_subtype_idx': test_data['pitch_subtype_idx'].values,
            'zone_idx': test_data['zone_idx'].values,
            'p_throws_idx': test_data['p_throws_idx'].values,
        })
        
        print("Generating posterior predictions...")
        posterior_pred = pm.sample_posterior_predictive(trace, var_names=['p'])
    
    p_hat = posterior_pred.posterior_predictive['p'].mean(dim=('chain', 'draw')).values
    test_data['predicted_whiff_prob'] = p_hat
    
    y_true = []
    y_pred = []
    for _, row in test_data.iterrows():
        n_whiffs_int = int(row['n_whiffs'])
        n_swings_int = int(row['n_swings'])
        y_true.extend([1] * n_whiffs_int)
        y_true.extend([0] * (n_swings_int - n_whiffs_int))
        y_pred.extend([row['predicted_whiff_prob']] * n_swings_int)
        
    print("\n--- Interaction Model Evaluation on Test Set ---")
    if len(np.unique(y_true)) > 1:
        brier = brier_score_loss(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        print(f"Brier Score: {brier:.4f}")
        print(f"AUC: {auc:.4f}")

        base_rate = np.mean(y_true)
        brier_random = brier_score_loss(y_true, [base_rate] * len(y_true))
        print(f"Brier Score (Random): {brier_random:.4f}")
    else:
        print("Test set contains only one class. Cannot calculate AUC or Brier score.")

if __name__ == "__main__":
    main() 
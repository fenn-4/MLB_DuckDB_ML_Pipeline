import duckdb
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import pickle
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --- Evaluation Date Range Constants ---
# Ensure this range does not overlap with the training data
TEST_START_DATE = '2025-06-22'
TEST_END_DATE = '2025-06-24'

# --- File Paths ---
DB_PATH = 'Full_DB/mlb_statcast.db'
TRACE_PATH = "Model_Weights/routinized_whiff_model_trace.nc"
ARTIFACTS_PATH = "Model_Weights/routinized_whiff_model_artifacts.pkl"

def load_test_data(db_path, encoders):
    """
    Loads and preprocesses the test data for the combined model.
    """
    con = duckdb.connect(db_path, read_only=True)
    
    # We don't need to pre-qualify players here; we just need data in the test range
    # for players that the model was trained on.
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
    
    # Filter data to only include players/categories seen during training
    for col, le in encoders.items():
        if col in data.columns:
            known_labels = set(le.classes_)
            data = data[data[col].isin(known_labels)]
            if not data.empty:
                data[col + '_idx'] = le.transform(data[col])
            
    data['zone_idx'] = data['is_zone'].astype(int)

    # Drop rows where an index could not be assigned
    required_idx_cols = [col + '_idx' for col in encoders.keys()]
    data.dropna(subset=required_idx_cols, inplace=True)
    for col in required_idx_cols:
        data[col] = data[col].astype(int)

    return data

def build_model_for_prediction(coords):
    """
    Rebuilds the routinized model structure for prediction.
    """
    with pm.Model(coords=coords) as model:
        # --- Data Placeholders ---
        batter_idx = pm.Data('batter_idx', np.array([], dtype='int32'), mutable=True)
        pitch_subtype_idx = pm.Data('pitch_subtype_idx', np.array([], dtype='int32'), mutable=True)
        zone_idx = pm.Data('zone_idx', np.array([], dtype='int32'), mutable=True)
        p_throws_idx = pm.Data('p_throws_idx', np.array([], dtype='int32'), mutable=True)

        # --- Priors & Effects (Structure must match training) ---
        alpha_global = pm.Normal('alpha_global', mu=0, sigma=1.5)

        sigma_batter = pm.HalfNormal('sigma_batter', sigma=1)
        beta_batter = pm.Normal('beta_batter', mu=0, sigma=sigma_batter, dims='batter')

        sigma_batter_zone = pm.HalfNormal('sigma_batter_zone', sigma=1)
        zeta_batter_zone = pm.Normal('zeta_batter_zone', mu=0, sigma=sigma_batter_zone, dims=('batter', 'zone'))

        sigma_batter_pitch_subtype = pm.HalfNormal('sigma_batter_pitch_subtype', sigma=1)
        gamma_batter_pitch_subtype = pm.Normal('gamma_batter_pitch_subtype', mu=0, sigma=sigma_batter_pitch_subtype, dims=('batter', 'pitch_subtype'))

        sigma_batter_p_throws = pm.HalfNormal('sigma_batter_p_throws', sigma=1)
        eta_batter_p_throws = pm.Normal('eta_batter_p_throws', mu=0, sigma=sigma_batter_p_throws, dims=('batter', 'p_throws'))

        # --- Linear Model ---
        logit_p = (
            alpha_global +
            beta_batter[batter_idx] +
            zeta_batter_zone[batter_idx, zone_idx] +
            gamma_batter_pitch_subtype[batter_idx, pitch_subtype_idx] +
            eta_batter_p_throws[batter_idx, p_throws_idx]
        )
        
        # A deterministic variable to easily sample the predicted probability
        pm.Deterministic('p', pm.math.invlogit(logit_p))

    return model

def main():
    print("Loading trained model and artifacts...")
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    trace = az.from_netcdf(TRACE_PATH)
    
    encoders = artifacts['encoders']
    # The saved coords can be a dictionary of dictionaries, let's fix it
    if isinstance(artifacts['coords'], dict) and all(isinstance(v, dict) for v in artifacts['coords'].values()):
         coords = {k: v['data'] for k, v in artifacts['coords'].items()}
    else:
         coords = artifacts['coords']

    print("Loading and preprocessing test data...")
    test_data = load_test_data(DB_PATH, encoders)
    
    if test_data.empty:
        print("No test data found for the specified date range with players from the training set. Exiting.")
        return

    print(f"Found {len(test_data)} aggregated rows in the test set.")

    # Rebuild the model structure and set the test data
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
    
    # Calculate the mean predicted probability for each observation
    p_hat = posterior_pred.posterior_predictive['p'].mean(dim=('chain', 'draw')).values
    test_data['predicted_whiff_prob'] = p_hat
    
    # "Un-aggregate" the data for scoring
    y_true = []
    y_pred = []
    for _, row in test_data.iterrows():
        n_whiffs_int = int(row['n_whiffs'])
        n_swings_int = int(row['n_swings'])
        y_true.extend([1] * n_whiffs_int)
        y_true.extend([0] * (n_swings_int - n_whiffs_int))
        y_pred.extend([row['predicted_whiff_prob']] * n_swings_int)
        
    print("\n--- Routinized Model Evaluation on Test Set ---")
    if len(np.unique(y_true)) > 1:
        brier = brier_score_loss(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        
        print(f"Brier Score: {brier:.4f}")
        print(f"AUC: {auc:.4f}")

        # --- Calculate Brier score for a random (climatological) forecast ---
        base_rate = np.mean(y_true)
        brier_random = brier_score_loss(y_true, [base_rate] * len(y_true))
        print(f"Brier Score (Random): {brier_random:.4f}")

        # --- Calculate AUC for a "guess-zero" forecast ---
        auc_zeros = roc_auc_score(y_true, [0] * len(y_true))
        print(f"AUC (Guessing Zeros): {auc_zeros:.4f}")

        # --- Plot ROC Curve ---
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curve.png')
        print("ROC curve plot saved to roc_curve.png")
        plt.close()

        # --- Plot Brier Score ---
        plt.figure(figsize=(6, 4))
        scores = {'Model': brier, 'Random': brier_random}
        plt.bar(scores.keys(), scores.values(), color=['skyblue', 'lightgray'])
        plt.ylabel('Score')
        plt.title('Brier Score Comparison')
        plt.ylim(0, max(0.25, brier * 1.5, brier_random * 1.5))
        plt.savefig('brier_score_comparison.png')
        print("Brier score comparison plot saved to brier_score_comparison.png")
        plt.close()

        # --- Contextual Performance Breakdown ---
        print("\n--- Contextual Performance Breakdown ---")
        
        # Use the un-aggregated original test_data for easier slicing
        y_true_full = np.array(y_true)
        y_pred_full = np.array(y_pred)
        
        # Create a DataFrame that mirrors the un-aggregated y_true and y_pred
        expanded_rows = []
        for _, row in test_data.iterrows():
            for _ in range(int(row['n_swings'])):
                expanded_rows.append(row)
        context_df = pd.DataFrame(expanded_rows)
        context_df['y_true'] = y_true_full
        context_df['y_pred'] = y_pred_full

        context_cols = ['pitch_subtype', 'is_zone', 'p_throws']
        
        for col in context_cols:
            print(f"\n--- Performance by {col} ---")
            
            # Group by the context column and calculate metrics
            for context, group in context_df.groupby(col):
                y_true_context = group['y_true'].values
                y_pred_context = group['y_pred'].values
                
                if len(y_true_context) < 2 or len(np.unique(y_true_context)) < 2:
                    print(f"  - {context}: Not enough data or only one class present.")
                    continue

                brier_context = brier_score_loss(y_true_context, y_pred_context)
                auc_context = roc_auc_score(y_true_context, y_pred_context)
                n_swings_context = len(y_true_context)
                
                context_label = context
                if col == 'is_zone':
                    context_label = 'In Zone' if context == 1 else 'Out of Zone'

                print(f"  - {context_label}: (Swings: {n_swings_context})")
                print(f"    - Brier: {brier_context:.4f}, AUC: {auc_context:.4f}")
    else:
        print("Test set contains only one class. Cannot calculate AUC or Brier score.")

if __name__ == "__main__":
    main() 
import duckdb
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, classification_report, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings('ignore')

# --- Date Range Constants ---
QUALIFYING_YEAR = 2025
TRAINING_START_DATE = '2024-05-01'
TRAINING_END_DATE = '2025-06-21'
TEST_START_DATE = '2025-06-22'
TEST_END_DATE = '2025-09-28'

def load_model_artifacts(artifacts_path):
    """
    Load the model artifacts from the pickle file.
    """
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

def load_test_data(db_path, encoders, test_start_date, test_end_date):
    """
    Load test data for Bernoulli model validation.
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
        SELECT batter, pitcher, stand, pitch_subtype, is_zone, p_throws, description, game_date, balls, strikes 
        FROM statcast_major
        WHERE batter IN (SELECT batter FROM qualified_batters)
          AND pitcher IN (SELECT pitcher FROM qualified_pitchers)
        
        UNION ALL

        SELECT batter, pitcher, stand, pitch_subtype, is_zone, p_throws, description, game_date, balls, strikes 
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
        balls,
        strikes,
        CASE WHEN description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END AS is_whiff
    FROM combined_data
    WHERE 
        game_date BETWEEN '{test_start_date}' AND '{test_end_date}'
        AND description IN ('swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play', 'foul_tip', 'hit_into_play_no_out', 'hit_into_play_score', 'foul_bunt')
        AND pitch_subtype IS NOT NULL 
        AND pitch_subtype != 'Other'
        AND is_zone IS NOT NULL
        AND p_throws IS NOT NULL
        AND stand IS NOT NULL
        AND pitcher IS NOT NULL
        AND balls IS NOT NULL
        AND strikes IS NOT NULL;
    """
    
    data = con.execute(query).fetchdf()
    con.close()
    
    print(f"Loaded {len(data)} test observations")
    print(f"Whiff rate in test data: {data['is_whiff'].mean():.3f}")
    
    # Create count buckets
    data['count'] = data['balls'].astype(str) + '-' + data['strikes'].astype(str)
    neutral = {'0-0', '1-1', '0-1', '1-0'}
    at_risk = {'0-2', '1-2', '2-2'}
    ahead = {'2-0', '2-1', '3-1', '3-2'}
    def bucket_count(c):
        if c in neutral:
            return 'neutral'
        elif c in at_risk:
            return 'at_risk'
        elif c in ahead:
            return 'ahead'
        else:
            return 'other'
    data['count_bucket'] = data['count'].apply(bucket_count)
    data = data[data['count_bucket'] != 'other']
    
    print(f"After filtering count buckets: {len(data)} observations")
    
    # Encode categorical variables using the same encoders from training
    for col in ['batter', 'pitcher', 'stand', 'pitch_subtype', 'p_throws', 'count_bucket']:
        if col in encoders:
            # Handle unseen categories by mapping them to -1
            data[col + '_idx'] = data[col].map(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)
        else:
            print(f"Warning: Encoder for {col} not found")
            # If encoder doesn't exist, create a new one
            le = LabelEncoder()
            data[col + '_idx'] = le.fit_transform(data[col])
            encoders[col] = le
    
    data['zone_idx'] = data['is_zone'].astype(int)
    
    # Remove rows with unseen categories
    for col in ['batter', 'pitcher', 'stand', 'pitch_subtype', 'p_throws', 'count_bucket']:
        data = data[data[col + '_idx'] != -1]
    
    print(f"After removing unseen categories: {len(data)} observations")
    
    return data

def predict_whiff_probabilities_2way(trace, data, coords):
    """
    Generate predictions using the 2-way interaction model trace.
    """
    # Extract posterior samples
    alpha_global = trace.posterior['alpha_global'].values
    
    # Extract batter 2-way interactions
    zeta_batter_zone = trace.posterior['zeta_batter_zone'].values
    gamma_batter_pitch_subtype = trace.posterior['gamma_batter_pitch_subtype'].values
    eta_batter_p_throws = trace.posterior['eta_batter_p_throws'].values
    lambda_batter_count = trace.posterior['lambda_batter_count'].values
    
    # Extract pitcher 2-way interactions
    zeta_pitcher_zone = trace.posterior['zeta_pitcher_zone'].values
    gamma_pitcher_pitch_subtype = trace.posterior['gamma_pitcher_pitch_subtype'].values
    eta_pitcher_stand = trace.posterior['eta_pitcher_stand'].values
    lambda_pitcher_count = trace.posterior['lambda_pitcher_count'].values
    
    # Calculate logits for each sample
    n_samples = alpha_global.shape[0] * alpha_global.shape[1]
    predictions = np.zeros((n_samples, len(data)))
    
    sample_idx = 0
    for chain in range(alpha_global.shape[0]):
        for draw in range(alpha_global.shape[1]):
            logits = (
                alpha_global[chain, draw] +
                # Batter 2-way interactions + count
                zeta_batter_zone[chain, draw, data['batter_idx'].values, data['zone_idx'].values] +
                gamma_batter_pitch_subtype[chain, draw, data['batter_idx'].values, data['pitch_subtype_idx'].values] +
                eta_batter_p_throws[chain, draw, data['batter_idx'].values, data['p_throws_idx'].values] +
                lambda_batter_count[chain, draw, data['batter_idx'].values, data['count_bucket_idx'].values] +
                # Pitcher 2-way interactions + count
                zeta_pitcher_zone[chain, draw, data['pitcher_idx'].values, data['zone_idx'].values] +
                gamma_pitcher_pitch_subtype[chain, draw, data['pitcher_idx'].values, data['pitch_subtype_idx'].values] +
                eta_pitcher_stand[chain, draw, data['pitcher_idx'].values, data['stand_idx'].values] +
                lambda_pitcher_count[chain, draw, data['pitcher_idx'].values, data['count_bucket_idx'].values]
            )
            
            predictions[sample_idx] = 1 / (1 + np.exp(-logits))
            sample_idx += 1
    
    # Return mean predictions across all samples
    return np.mean(predictions, axis=0)

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.
    """
    # Convert predictions to binary for classification metrics
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    
    # Classification report
    class_report = classification_report(y_true, y_pred_binary, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    return {
        'auc': auc,
        'brier_score': brier,
        'log_loss': logloss,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'true_values': y_true
    }

def plot_diagnostics(metrics, save_path='Model_Weights/bernoulli_model_2way_diagnostics.png'):
    """
    Create comprehensive diagnostic plots for the 2-way interaction model.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Bernoulli Model with 2-Way Interactions - Diagnostics', fontsize=16, fontweight='bold')
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(metrics['true_values'], metrics['predictions'])
    axes[0, 0].plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.3f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Prediction Distribution
    axes[0, 1].hist(metrics['predictions'][metrics['true_values'] == 0], 
                   alpha=0.7, label='No Whiff', bins=30, density=True)
    axes[0, 1].hist(metrics['predictions'][metrics['true_values'] == 1], 
                   alpha=0.7, label='Whiff', bins=30, density=True)
    axes[0, 1].set_xlabel('Predicted Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Prediction Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Calibration Plot
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        metrics['true_values'], metrics['predictions'], n_bins=10
    )
    axes[0, 2].plot(mean_predicted_value, fraction_of_positives, 'o-', label='Model')
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    axes[0, 2].set_xlabel('Mean Predicted Probability')
    axes[0, 2].set_ylabel('Fraction of Positives')
    axes[0, 2].set_title('Calibration Plot')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # 5. Metrics Summary
    metrics_text = f"""
    AUC: {metrics['auc']:.3f}
    Brier Score: {metrics['brier_score']:.3f}
    Log Loss: {metrics['log_loss']:.3f}
    
    Precision: {metrics['classification_report']['1']['precision']:.3f}
    Recall: {metrics['classification_report']['1']['recall']:.3f}
    F1-Score: {metrics['classification_report']['1']['f1-score']:.3f}
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                   fontsize=12, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].set_title('Model Metrics')
    axes[1, 1].axis('off')
    
    # 6. Residual Plot
    residuals = metrics['true_values'] - metrics['predictions']
    axes[1, 2].scatter(metrics['predictions'], residuals, alpha=0.5)
    axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Predicted Probability')
    axes[1, 2].set_ylabel('Residuals')
    axes[1, 2].set_title('Residual Plot')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Diagnostic plots saved to {save_path}")

def print_detailed_report(metrics):
    """
    Print a detailed evaluation report for the 2-way interaction model.
    """
    print("=" * 70)
    print("BERNOULLI MODEL WITH 2-WAY INTERACTIONS - EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\n📊 MODEL PERFORMANCE METRICS:")
    print(f"   AUC Score: {metrics['auc']:.4f}")
    print(f"   Brier Score: {metrics['brier_score']:.4f}")
    print(f"   Log Loss: {metrics['log_loss']:.4f}")
    
    print(f"\n📈 CLASSIFICATION METRICS:")
    report = metrics['classification_report']
    print(f"   Precision (Whiff): {report['1']['precision']:.4f}")
    print(f"   Recall (Whiff): {report['1']['recall']:.4f}")
    print(f"   F1-Score (Whiff): {report['1']['f1-score']:.4f}")
    print(f"   Precision (No Whiff): {report['0']['precision']:.4f}")
    print(f"   Recall (No Whiff): {report['0']['recall']:.4f}")
    print(f"   F1-Score (No Whiff): {report['0']['f1-score']:.4f}")
    
    print(f"\n📋 CONFUSION MATRIX:")
    cm = metrics['confusion_matrix']
    print(f"   True Negatives: {cm[0, 0]}")
    print(f"   False Positives: {cm[0, 1]}")
    print(f"   False Negatives: {cm[1, 0]}")
    print(f"   True Positives: {cm[1, 1]}")
    
    # Calculate additional metrics
    total = cm.sum()
    accuracy = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])  # True Positive Rate
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # True Negative Rate
    
    print(f"\n🔍 ADDITIONAL METRICS:")
    print(f"   Overall Accuracy: {accuracy:.4f}")
    print(f"   Sensitivity (TPR): {sensitivity:.4f}")
    print(f"   Specificity (TNR): {specificity:.4f}")
    
    print("=" * 70)

def main():
    """
    Main evaluation function for the 2-way interaction model.
    """
    print("Loading model artifacts...")
    artifacts_path = "Model_Weights/combined_whiff_model_bernoulli_2way_artifacts.pkl"
    artifacts = load_model_artifacts(artifacts_path)
    
    print("Loading test data...")
    db_path = 'Full_DB/mlb_statcast.db'
    test_data = load_test_data(db_path, artifacts['encoders'], TEST_START_DATE, TEST_END_DATE)
    
    print(f"Test dataset size: {len(test_data)} observations")
    print(f"Whiff rate in test set: {test_data['is_whiff'].mean():.3f}")
    
    print("Loading model trace...")
    trace_path = "Model_Weights/combined_whiff_model_bernoulli_2way_trace.nc"
    trace = az.from_netcdf(trace_path)
    
    print("Generating predictions...")
    predictions = predict_whiff_probabilities_2way(trace, test_data, artifacts['coords'])
    
    print("Calculating metrics...")
    metrics = calculate_metrics(test_data['is_whiff'].values, predictions)
    
    print("Generating diagnostic plots...")
    plot_diagnostics(metrics)
    
    print("Printing detailed report...")
    print_detailed_report(metrics)

    # --- Diagnostics by pitch_subtype ---
    print("\nDiagnostics by pitch_subtype:")
    pitchtype_results = []
    for pitch_type, group in test_data.groupby('pitch_subtype'):
        idx = group.index
        y_true = group['is_whiff'].values
        y_pred = predictions[idx]
        if len(np.unique(y_true)) < 2:
            # Skip if only one class present
            continue
        try:
            auc = roc_auc_score(y_true, y_pred)
        except Exception:
            auc = np.nan
        try:
            brier = brier_score_loss(y_true, y_pred)
        except Exception:
            brier = np.nan
        try:
            logloss = log_loss(y_true, y_pred)
        except Exception:
            logloss = np.nan
        y_pred_binary = (y_pred > 0.5).astype(int)
        class_report = classification_report(y_true, y_pred_binary, output_dict=True)
        pitchtype_results.append({
            'pitch_subtype': pitch_type,
            'n': len(y_true),
            'auc': auc,
            'brier_score': brier,
            'log_loss': logloss,
            'precision_whiff': class_report['1']['precision'] if '1' in class_report else np.nan,
            'recall_whiff': class_report['1']['recall'] if '1' in class_report else np.nan,
            'f1_whiff': class_report['1']['f1-score'] if '1' in class_report else np.nan,
            'precision_no_whiff': class_report['0']['precision'] if '0' in class_report else np.nan,
            'recall_no_whiff': class_report['0']['recall'] if '0' in class_report else np.nan,
            'f1_no_whiff': class_report['0']['f1-score'] if '0' in class_report else np.nan,
        })
    pitchtype_df = pd.DataFrame(pitchtype_results)
    pitchtype_df = pitchtype_df.sort_values('auc', ascending=False)
    pitchtype_df.to_csv('Model_Weights/bernoulli_model_2way_pitchtype_diagnostics.csv', index=False)
    print(pitchtype_df[['pitch_subtype','n','auc','brier_score','log_loss','precision_whiff','recall_whiff','f1_whiff']])
    print("\nPer-pitch-type diagnostics saved to Model_Weights/bernoulli_model_2way_pitchtype_diagnostics.csv")

    # Optionally: plot ROC curves for each pitch type (top 6 by count)
    top_types = pitchtype_df.sort_values('n', ascending=False).head(6)['pitch_subtype']
    plt.figure(figsize=(10,7))
    for pitch_type in top_types:
        group = test_data[test_data['pitch_subtype'] == pitch_type]
        idx = group.index
        y_true = group['is_whiff'].values
        y_pred = predictions[idx]
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label=f'{pitch_type} (AUC={pitchtype_df[pitchtype_df.pitch_subtype==pitch_type]["auc"].values[0]:.2f})')
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Pitch Subtype - 2-Way Interaction Model (Top 6)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Model_Weights/bernoulli_model_2way_pitchtype_roc.png', dpi=300)
    plt.show()
    print("ROC curves by pitch type saved to Model_Weights/bernoulli_model_2way_pitchtype_roc.png")

    # Save results
    results = {
        'metrics': metrics,
        'test_data_size': len(test_data),
        'test_whiff_rate': test_data['is_whiff'].mean(),
        'predictions': predictions,
        'true_values': test_data['is_whiff'].values,
        'pitchtype_diagnostics': pitchtype_results
    }
    
    with open('Model_Weights/bernoulli_model_2way_evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✅ Evaluation complete! Results saved to Model_Weights/bernoulli_model_2way_evaluation_results.pkl")

if __name__ == "__main__":
    main() 
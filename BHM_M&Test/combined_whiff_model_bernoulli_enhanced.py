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
    Loads and preprocesses data from the DuckDB database for the enhanced Bernoulli model with zone parameters and player x zone interactions.
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
    
    # Calculate batter whiff rates and create clusters
    batter_whiff_rates = data.groupby('batter')['is_whiff'].mean().reset_index()
    # Use 3-bucket grouping
    batter_whiff_rates['batter_whiff_group'] = pd.qcut(batter_whiff_rates['is_whiff'], q=3, 
                                                      labels=['Low', 'Medium', 'High'])
    # Merge back to main data
    data = data.merge(batter_whiff_rates[['batter', 'batter_whiff_group']], on='batter', how='left')
    
    # Create count as a categorical variable (e.g., 'balls-strikes')
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
    for col in ['batter', 'pitcher', 'stand', 'pitch_subtype', 'p_throws', 'league', 'batter_whiff_group', 'count_bucket']:
        le = LabelEncoder()
        data[col + '_idx'] = le.fit_transform(data[col])
        encoders[col] = le
        
    data['zone_idx'] = data['is_zone'].astype(int)
    
    # Create zone number index (1-9, 11-14 -> 0-12)
    zone_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 11: 9, 12: 10, 13: 11, 14: 12}
    data['zone_num_idx'] = data['zone'].map(zone_mapping)

    return data, encoders

def build_and_run_model(data, encoders):
    """
    Builds and runs an enhanced Hierarchical Bernoulli model with zone parameters and player x zone interactions.
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
        'batter_whiff_group': encoders['batter_whiff_group'].classes_,
        'count_bucket': encoders['count_bucket'].classes_,
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
        batter_whiff_group_idx = pm.Data('batter_whiff_group_idx', data['batter_whiff_group_idx'].values, mutable=True)
        count_bucket_idx = pm.Data('count_bucket_idx', data['count_bucket_idx'].values, mutable=True)
        is_whiff = pm.Data('is_whiff', data['is_whiff'].values, mutable=True)

        alpha_global = pm.Normal('alpha_global', mu=0, sigma=1.5)
        
        # --- Batter Effects ---
        # Batter x Zone interactions (in/out of zone) - base effects
        sigma_batter_zone = pm.HalfNormal('sigma_batter_zone', sigma=0.5)
        zeta_batter_zone_offset = pm.Normal('zeta_batter_zone_offset', mu=0, sigma=1, dims=('batter', 'zone'))
        zeta_batter_zone = pm.Deterministic('zeta_batter_zone', zeta_batter_zone_offset * sigma_batter_zone, dims=('batter', 'zone'))

        # Batter x pitch subtype interactions - base effects
        sigma_batter_pitch_subtype = pm.HalfNormal('sigma_batter_pitch_subtype', sigma=0.5)
        gamma_batter_pitch_subtype_offset = pm.Normal('gamma_batter_pitch_subtype_offset', mu=0, sigma=1, dims=('batter', 'pitch_subtype'))
        gamma_batter_pitch_subtype = pm.Deterministic('gamma_batter_pitch_subtype', gamma_batter_pitch_subtype_offset * sigma_batter_pitch_subtype, dims=('batter', 'pitch_subtype'))
        
        # Batter x pitcher throwing hand interactions
        sigma_batter_p_throws = pm.HalfNormal('sigma_batter_p_throws', sigma=0.5)
        eta_batter_p_throws_offset = pm.Normal('eta_batter_p_throws_offset', mu=0, sigma=1, dims=('batter', 'p_throws'))
        eta_batter_p_throws = pm.Deterministic('eta_batter_p_throws', eta_batter_p_throws_offset * sigma_batter_p_throws, dims=('batter', 'p_throws'))

        # Batter league adjustment (how batters perform differently in majors vs minors)
        # League-specific batter effects (captures difficulty difference)
        sigma_batter_league = pm.HalfNormal('sigma_batter_league', sigma=0.3)
        beta_batter_league_offset = pm.Normal('beta_batter_league_offset', mu=0, sigma=1, dims=('batter', 'league'))
        beta_batter_league = pm.Deterministic('beta_batter_league', beta_batter_league_offset * sigma_batter_league, dims=('batter', 'league'))

        # --- Pitcher Effects ---
        # Pitcher x Zone interactions (in/out of zone) - base effects
        sigma_pitcher_zone = pm.HalfNormal('sigma_pitcher_zone', sigma=0.5)
        zeta_pitcher_zone_offset = pm.Normal('zeta_pitcher_zone_offset', mu=0, sigma=1, dims=('pitcher', 'zone'))
        zeta_pitcher_zone = pm.Deterministic('zeta_pitcher_zone', zeta_pitcher_zone_offset * sigma_pitcher_zone, dims=('pitcher', 'zone'))

        # Pitcher x pitch subtype interactions - base effects
        sigma_pitcher_pitch_subtype = pm.HalfNormal('sigma_pitcher_pitch_subtype', sigma=0.5)
        gamma_pitcher_pitch_subtype_offset = pm.Normal('gamma_pitcher_pitch_subtype_offset', mu=0, sigma=1, dims=('pitcher', 'pitch_subtype'))
        gamma_pitcher_pitch_subtype = pm.Deterministic('gamma_pitcher_pitch_subtype', gamma_pitcher_pitch_subtype_offset * sigma_pitcher_pitch_subtype, dims=('pitcher', 'pitch_subtype'))
        
        # Pitcher x batter stance interactions
        sigma_pitcher_stand = pm.HalfNormal('sigma_pitcher_stand', sigma=.5)
        eta_pitcher_stand_offset = pm.Normal('eta_pitcher_stand_offset', mu=0, sigma=1, dims=('pitcher', 'stand'))
        eta_pitcher_stand = pm.Deterministic('eta_pitcher_stand', eta_pitcher_stand_offset * sigma_pitcher_stand, dims=('pitcher', 'stand'))

        # League-specific pitcher effects (captures difficulty difference)
        sigma_pitcher_league = pm.HalfNormal('sigma_pitcher_league', sigma=0.3)
        beta_pitcher_league_offset = pm.Normal('beta_pitcher_league_offset', mu=0, sigma=1, dims=('pitcher', 'league'))
        beta_pitcher_league = pm.Deterministic('beta_pitcher_league', beta_pitcher_league_offset * sigma_pitcher_league, dims=('pitcher', 'league'))

        # --- Whiff Group Effects (3-way interactions) ---
        # Whiff group effect: batter_whiff_group × zone_num × pitch_type
        sigma_batter_whiff_group_zone_pitch = pm.HalfNormal('sigma_batter_whiff_group_zone_pitch', sigma=1)
        theta_batter_whiff_group_zone_pitch_offset = pm.Normal('theta_batter_whiff_group_zone_pitch_offset', mu=0, sigma=1, dims=('batter_whiff_group', 'zone_num', 'pitch_subtype'))
        theta_batter_whiff_group_zone_pitch = pm.Deterministic('theta_batter_whiff_group_zone_pitch', theta_batter_whiff_group_zone_pitch_offset * sigma_batter_whiff_group_zone_pitch, dims=('batter_whiff_group', 'zone_num', 'pitch_subtype'))

        # Whiff group × count interaction
        sigma_batter_whiff_group_count = pm.HalfNormal('sigma_batter_whiff_group_count', sigma=0.5)
        phi_batter_whiff_group_count_offset = pm.Normal('phi_batter_whiff_group_count_offset', mu=0, sigma=1, dims=('batter_whiff_group', 'count_bucket'))
        phi_batter_whiff_group_count = pm.Deterministic('phi_batter_whiff_group_count', phi_batter_whiff_group_count_offset * sigma_batter_whiff_group_count, dims=('batter_whiff_group', 'count_bucket'))

        # Enhanced logit with zone effects and player x zone interactions
        logit_p = (
            alpha_global +
            # League-specific batter effects (captures difficulty difference)
            beta_batter_league[batter_idx, league_idx] +
            # Batter effects (2-way base + pitcher throwing hand)
            zeta_batter_zone[batter_idx, zone_idx] +
            gamma_batter_pitch_subtype[batter_idx, pitch_subtype_idx] +
            eta_batter_p_throws[batter_idx, p_throws_idx] +
            # Pitcher effects (2-way base + batter stance)
            zeta_pitcher_zone[pitcher_idx, zone_idx] +
            gamma_pitcher_pitch_subtype[pitcher_idx, pitch_subtype_idx] +
            eta_pitcher_stand[pitcher_idx, stand_idx] +
            # League-specific pitcher effects (captures difficulty difference)
            beta_pitcher_league[pitcher_idx, league_idx] +
            # Whiff group effects (3-way interactions + count)
            theta_batter_whiff_group_zone_pitch[batter_whiff_group_idx, zone_num_idx, pitch_subtype_idx] +
            phi_batter_whiff_group_count[batter_whiff_group_idx, count_bucket_idx]
        )
        
        pm.Bernoulli('whiffs', p=pm.math.invlogit(logit_p), observed=is_whiff)

        trace = pm.sample(2000, tune=1000, chains=4, cores=4, target_accept=.8, max_treedepth=14, random_seed=42)
        
    return trace

def save_artifacts(trace, encoders, data):
    """
    Saves the model trace and other artifacts for the enhanced Bernoulli model with zone parameters and player x zone interactions.
    """
    trace.to_netcdf("Model_Weights/combined_whiff_model_bernoulli_enhanced_trace.nc")
    
    coords_to_save = {
        'batter': encoders['batter'].classes_,
        'pitcher': encoders['pitcher'].classes_,
        'pitch_subtype': encoders['pitch_subtype'].classes_,
        'zone': ['out_of_zone', 'in_zone'],
        'zone_num': list(range(13)),  # 0-12 for zones 1-9, 11-14
        'p_throws': encoders['p_throws'].classes_,
        'stand': encoders['stand'].classes_,
        'league': encoders['league'].classes_,
        'batter_whiff_group': encoders['batter_whiff_group'].classes_,
        'count_bucket': encoders['count_bucket'].classes_,
    }
    
    artifacts = {
        'encoders': encoders,
        'coords': coords_to_save,
        'data_columns': data.columns
    }
    with open("Model_Weights/combined_whiff_model_bernoulli_enhanced_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
        
    print("Enhanced Bernoulli model with zone parameters and player x zone interactions trace and artifacts saved successfully.")

if __name__ == "__main__":
    DB_PATH = 'Full_DB/mlb_statcast.db'
    
    df, model_encoders = load_and_preprocess_data(DB_PATH)
    
    print("Building and running enhanced Bernoulli model with zone parameters and player x zone interactions...")
    model_trace = build_and_run_model(df, model_encoders)
    
    print("Saving model artifacts...")
    save_artifacts(model_trace, model_encoders, df)
    
    print("\nModel fitting complete.")
    summary = az.summary(model_trace, var_names=[
        'alpha_global',
        'sigma_batter_zone', 'sigma_batter_pitch_subtype', 'sigma_batter_p_throws', 'sigma_batter_league',
        'sigma_pitcher_zone', 'sigma_pitcher_pitch_subtype', 'sigma_pitcher_stand', 'sigma_pitcher_league',
        'sigma_batter_whiff_group_zone_pitch', 'sigma_batter_whiff_group_count'
    ])
    print(summary) 
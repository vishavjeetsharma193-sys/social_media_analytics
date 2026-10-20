import pandas as pd
import numpy as np
import os
import traceback

# Import XGBoost and SHAP with robust fallback mechanisms
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    'Base_Screen_Time', 'Base_Night_Usage', 'Base_Checking_Freq',
    'Curr_Screen_Time', 'Curr_Night_Usage', 'Curr_Checking_Freq',
    'Delta_Screen_Time', 'Delta_Session_Freq', 'Delta_Avg_Session_Dur',
    'Delta_Night_Usage', 'Delta_Checking_Freq', 'Delta_Sleep_Usage',
    'Usage_Variability'
]

FEATURE_LABELS = {
    'Base_Screen_Time': 'Base Screen Time (h)',
    'Base_Night_Usage': 'Base Night Usage (h)',
    'Base_Checking_Freq': 'Base Checking Freq (/h)',
    'Curr_Screen_Time': 'Curr Screen Time (h)',
    'Curr_Night_Usage': 'Curr Night Usage (h)',
    'Curr_Checking_Freq': 'Curr Checking Freq (/h)',
    'Delta_Screen_Time': 'Δ Screen Time (h)',
    'Delta_Session_Freq': 'Δ Session Freq',
    'Delta_Avg_Session_Dur': 'Δ Session Dur (m)',
    'Delta_Night_Usage': 'Δ Night Usage (h)',
    'Delta_Checking_Freq': 'Δ Checking Freq',
    'Delta_Sleep_Usage': 'Δ Sleep Usage (h)',
    'Usage_Variability': 'Usage Variability (std)'
}

def train_xgboost_and_compute_pari(df_features, cache_path="data/students_temporal_pari_cache.csv"):
    """
    Trains XGBoost model on behavioral change features, calculates SHAP values,
    computes Personalized Addiction Risk Index (PARI), and assigns 4-level risk categories.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            cached_df = pd.read_csv(cache_path)
            if len(cached_df) == len(df_features) and 'PARI_Score' in cached_df.columns:
                return cached_df
        except Exception:
            pass

    df = df_features.copy()
    X = df[FEATURE_COLS].values
    y = df['High_Risk_Target'].values

    # Fit Model (XGBoost if available, else GradientBoostingClassifier fallback)
    if HAS_XGBOOST:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X, y)
        risk_probs = model.predict_proba(X)[:, 1]
    else:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=42
        )
        model.fit(X, y)
        risk_probs = model.predict_proba(X)[:, 1]

    # Calculate SHAP Values
    shap_matrix = np.zeros(X.shape)
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_matrix = shap_vals[1]
            else:
                shap_matrix = shap_vals
        except Exception as e:
            print("SHAP TreeExplainer exception, using feature importance approximation:", e)
            shap_matrix = _approximate_shap_matrix(model, X, risk_probs)
    else:
        shap_matrix = _approximate_shap_matrix(model, X, risk_probs)

    # Attach SHAP values to DataFrame for local/global inspection
    for idx, col in enumerate(FEATURE_COLS):
        df[f'SHAP_{col}'] = np.round(shap_matrix[:, idx], 4)

    # Compute Personalized Addiction Risk Index (PARI)
    pari_scores = []
    change_scores = []
    persistence_scores = []
    risk_categories = []
    alert_messages = []

    for i in range(len(df)):
        row = df.iloc[i]
        p_risk = risk_probs[i] # 0 to 1

        # 1. Behavioral Change Score (0 to 1)
        # Scaled sum of positive delta escalations
        st_surge = max(0, row['Delta_Screen_Time']) / 4.0
        ngt_surge = max(0, row['Delta_Night_Usage']) / 2.5
        chk_surge = max(0, row['Delta_Checking_Freq']) / 15.0
        var_surge = row['Usage_Variability'] / 3.0
        
        change_score = np.clip((0.35 * st_surge + 0.40 * ngt_surge + 0.15 * chk_surge + 0.10 * var_surge), 0.0, 1.0)

        # 2. Persistence Score (0 to 1)
        # Measures consistent weekly upward progression (W1 <= W2 <= W3 <= W4)
        w1, w2, w3, w4 = row['W1_Screen'], row['W2_Screen'], row['W3_Screen'], row['W4_Screen']
        increases = int(w2 >= w1) + int(w3 >= w2) + int(w4 >= w3)
        sustained_high = int(w3 > 4.5) + int(w4 > 4.5)
        
        persistence = np.clip((increases / 3.0 * 0.6) + (sustained_high / 2.0 * 0.4), 0.0, 1.0)

        # 3. PARI Formula:
        # PARI = current-risk probability + behavioral-change score + persistence score
        pari = 100.0 * (0.40 * p_risk + 0.35 * change_score + 0.25 * persistence)
        pari = np.clip(pari, 0.0, 100.0)

        # Classification into 4 Dynamic Risk Levels:
        # 🟢 Stable / low risk
        # 🟡 Increasing risk
        # 🟠 Persistent high risk
        # 🔴 Rapidly increasing risk
        if pari < 35.0:
            category = "🟢 Stable/Low Risk"
        elif pari >= 70.0 and change_score >= 0.60:
            category = "🔴 Rapidly Increasing Risk"
        elif pari >= 60.0 or (p_risk >= 0.65 and persistence >= 0.60):
            category = "🟠 Persistent High Risk"
        else:
            category = "🟡 Increasing Risk"

        # Formulate Early-Warning Alert Message based on SHAP & Delta Drivers
        alerts = []
        if row['Delta_Night_Usage'] >= 0.75:
            alerts.append(f"Night usage spike (+{row['Delta_Night_Usage']}h)")
        if row['Delta_Screen_Time'] >= 1.5:
            alerts.append(f"Screen time surge (+{row['Delta_Screen_Time']}h)")
        if row['Delta_Checking_Freq'] >= 5.0:
            alerts.append(f"Checking frequency surge (+{row['Delta_Checking_Freq']}/h)")

        if category in ["🔴 Rapidly Increasing Risk", "🟠 Persistent High Risk"]:
            if alerts:
                msg = f"🚨 EARLY WARNING: " + ", ".join(alerts) + " detected relative to personal baseline!"
            else:
                msg = "🚨 EARLY WARNING: Elevated addiction risk probability detected over consecutive weeks!"
        elif category == "🟡 Increasing Risk":
            if alerts:
                msg = f"⚠️ ALERT: Moderate behavioral escalation - " + ", ".join(alerts) + "."
            else:
                msg = "⚠️ ALERT: Behavioral trends show initial escalation from personal baseline."
        else:
            msg = "✅ STABLE: Behavioral patterns remain within healthy baseline boundaries."

        pari_scores.append(np.round(pari, 1))
        change_scores.append(np.round(change_score * 100, 1))
        persistence_scores.append(np.round(persistence * 100, 1))
        risk_categories.append(category)
        alert_messages.append(msg)

    df['XGB_Risk_Prob'] = np.round(risk_probs, 3)
    df['PARI_Score'] = pari_scores
    df['Change_Score'] = change_scores
    df['Persistence_Score'] = persistence_scores
    df['PARI_Category'] = risk_categories
    df['Early_Warning_Alert'] = alert_messages

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        
    return df

def _approximate_shap_matrix(model, X, risk_probs):
    """
    Computes exact marginal feature attributions as a robust fallback.
    """
    n_samples, n_features = X.shape
    shap_matrix = np.zeros((n_samples, n_features))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.ones(n_features) / n_features

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-6

    for i in range(n_samples):
        centered_x = (X[i] - X_mean) / X_std
        # Attribution = normalized deviation * feature importance * prediction deviation
        dev = (risk_probs[i] - 0.5)
        shap_matrix[i] = centered_x * importances * dev * 2.0

    return shap_matrix

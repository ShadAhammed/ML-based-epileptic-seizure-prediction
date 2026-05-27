"""
=============================================================================
PASTE THIS INTO A NEW CELL AT THE END OF Epilepsy.ipynb
=============================================================================

Run it AFTER:
  - Cell that scales X with MinMaxScaler  (global variable: scaler)
  - Cell 26 (SMOTE + Smodel training)     (global variable: Smodel)
  - And the GetFeature call that defines f  (global variable: f)

This writes:  models/seizure_model.joblib
The dashboard will then show:  Model loaded successfully
=============================================================================
"""

# --- Re-fit RFECV once to capture the selector object (not returned by GetFeature) ---
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rfe = RFECV(estimator=XGBClassifier(), min_features_to_select=16, cv=3)
rfe.fit(X_train, y_train)
print(f"Selected {rfe.n_features_} features:", [c for c, s in zip(X_train.columns, rfe.support_) if s])

# --- Save in application format ---
import sys
from pathlib import Path

# Add src to path if running inside notebook from repo root
repo = Path.cwd()
if (repo / "src").exists():
    sys.path.insert(0, str(repo / "src"))

from epilepsy_detection.notebook_export import save_trained_model

out = save_trained_model(
    scaler=scaler,           # MinMaxScaler from notebook (fit on X)
    rfe=rfe,                 # RFECV fitted above
    model=Smodel,            # XGBoost trained with SMOTE (cell 26)
    feature_columns=list(X.columns),
    selected_features=f,     # feature names from GetFeature
    strategy="smote",
    output_path="models/seizure_model.joblib",
)

print(f"Model saved to: {out}")
print("Restart the dashboard - the sidebar should show a green checkmark.")

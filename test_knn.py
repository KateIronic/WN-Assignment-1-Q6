"""
KNN Localization - Testing Script
Loads a pre-trained KNN model and generates location predictions for the test set.
"""

import pandas as pd
import numpy as np
import pickle

# ── Load saved model ────────────────────────────────────────────────────────────
with open("knn_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model      = bundle["model"]
dist_cols  = bundle["dist_cols"]
target_cols = bundle["target_cols"]
print(f"Loaded KNN model  (k = {bundle['k']})")

# ── Load test data ──────────────────────────────────────────────────────────────
test_df = pd.read_csv("/home/iiitd/Desktop/WN-Assignment-1-Q6/data/datatest.csv")
X_test  = test_df[dist_cols].values
print(f"Test samples: {len(X_test)}")

# ── Predict ─────────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── Save results ────────────────────────────────────────────────────────────────
results = pd.DataFrame(y_pred, columns=target_cols)
results.to_csv("/home/iiitd/Desktop/WN-Assignment-1-Q6/data/test_predictions.csv", index=False)
print("Predictions saved to  /home/iiitd/Desktop/WN-Assignment-1-Q6/data/test_predictions.csv")
print(results.head())

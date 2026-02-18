"""
BONUS – Localization Accuracy vs. Training Set Size
Uses 20% of the training CSV as a held-out validation set.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# ── Load data ───────────────────────────────────────────────────────────────────
train_df = pd.read_csv("/home/iiitd/Desktop/WN-Assignment-1-Q6/data/datatrain.csv")
DIST_COLS   = [f"Dist_A{i}" for i in range(8)]
TARGET_COLS = ["Target_X", "Target_Y", "Target_Z"]

X = train_df[DIST_COLS].values
y = train_df[TARGET_COLS].values

# Load best k from saved model
with open("knn_model.pkl", "rb") as f:
    best_k = pickle.load(f)["k"]

# 80/20 split: use only the 80% portion for training, 20% for validation
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

fractions = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
sizes, mean_errors = [], []

for frac in fractions:
    n = max(best_k, int(frac * len(X_tr)))   # need at least k samples
    idx = np.random.default_rng(0).choice(len(X_tr), size=n, replace=False)
    knn = KNeighborsRegressor(n_neighbors=best_k, weights="distance", metric="euclidean")
    knn.fit(X_tr[idx], y_tr[idx])
    y_pred_val = knn.predict(X_val)
    errors = np.sqrt(np.sum((y_pred_val - y_val) ** 2, axis=1))
    sizes.append(n)
    mean_errors.append(errors.mean())
    print(f"  Train size={n:4d}  Mean 3-D error = {errors.mean():.4f} m")

# ── Bar plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar([str(s) for s in sizes], mean_errors,
              color=plt.cm.Blues(np.linspace(0.4, 0.85, len(sizes))),
              edgecolor="navy", linewidth=0.8)

for bar, err in zip(bars, mean_errors):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{err:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xlabel("Training Set Size (# samples)", fontsize=12)
ax.set_ylabel("Mean 3-D Localization Error (m)", fontsize=12)
ax.set_title(f"KNN Localization Accuracy vs. Training Set Size  (k={best_k})", fontsize=13)
ax.set_ylim(0, max(mean_errors) * 1.20)
plt.tight_layout()
plt.savefig("accuracy_vs_trainsize.png", dpi=150)
print("Saved accuracy_vs_trainsize.png")
plt.show()

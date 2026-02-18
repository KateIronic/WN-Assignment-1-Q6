"""
KNN Localization - Training Script
Trains a KNN model on the provided dataset and saves it for later use.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import pickle
import matplotlib.pyplot as plt


train_df = pd.read_csv("/home/iiitd/Desktop/WN-Assignment-1-Q6/data/datatrain.csv")

DIST_COLS = [f"Dist_A{i}" for i in range(8)]
TARGET_COLS = ["Target_X", "Target_Y", "Target_Z"]

X_train = train_df[DIST_COLS].values
y_train = train_df[TARGET_COLS].values

print(f"Training samples : {len(X_train)}")
print(f"Feature columns  : {DIST_COLS}")
print(f"Target columns   : {TARGET_COLS}")


k_values = range(1, 21)
cv_errors = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance", metric="euclidean")

    scores = cross_val_score(knn, X_train, y_train,
                             cv=5, scoring="neg_mean_squared_error")
    rmse = np.sqrt(-scores.mean())
    cv_errors.append(rmse)
    print(f"  k={k:2d}  CV RMSE = {rmse:.4f} m")

best_k = k_values[int(np.argmin(cv_errors))]
print(f"\nBest k = {best_k}  (CV RMSE = {min(cv_errors):.4f} m)")


model = KNeighborsRegressor(n_neighbors=best_k, weights="distance", metric="euclidean")
model.fit(X_train, y_train)


with open("knn_model.pkl", "wb") as f:
    pickle.dump({"model": model, "k": best_k, "dist_cols": DIST_COLS,
                 "target_cols": TARGET_COLS}, f)
print("Model saved to knn_model.pkl")


y_pred_train = model.predict(X_train)
train_errors = np.sqrt(np.sum((y_pred_train - y_train) ** 2, axis=1))
print(f"\nTraining-set localization error  (mean ± std): "
      f"{train_errors.mean():.4f} ± {train_errors.std():.4f} m")


plt.figure(figsize=(7, 4))
plt.plot(list(k_values), cv_errors, marker="o", linewidth=2)
plt.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("5-Fold CV RMSE (m)")
plt.title("KNN Hyperparameter Selection")
plt.legend()
plt.tight_layout()
plt.savefig("cv_rmse_vs_k.png", dpi=150)
print("Saved cv_rmse_vs_k.png")

"""Robustness of the CKD XGBoost model to synthetic input noise.

Usage (from the repo root, next to the .pkl and .csv):
    python noise_robustness.py
Produces: noise_robustness.png
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = "final_xg_model_for_ckd_status.pkl"
DATA_PATH = "kidney_disease_dataset.csv"
CONTINUOUS = ["Age", "Creatinine_Level", "BUN", "GFR", "Urine_Output"]
SCALES = [0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
N_SEEDS = 20

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
y = df["CKD_Status"]
X = df.drop(columns=["CKD_Status"])[list(model.feature_names_in_)]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
std = X.std()


def noisy(kind, scale, rng):
    Z = X_test.copy().astype(float)
    shape = (len(Z), len(CONTINUOUS))
    eps = rng.normal(0, scale, shape)
    if kind == "additive":          # sigma = scale * feature std
        Z[CONTINUOUS] = Z[CONTINUOUS] + eps * std[CONTINUOUS].values
    else:                           # multiplicative: x * (1 + eps)
        Z[CONTINUOUS] = Z[CONTINUOUS] * (1 + eps)
    return Z


results = {}
for kind in ["additive", "multiplicative"]:
    means, stds = [], []
    for s in SCALES:
        accs = [accuracy_score(y_test, model.predict(noisy(kind, s, np.random.default_rng(seed))))
                for seed in range(N_SEEDS)]
        means.append(np.mean(accs) * 100)
        stds.append(np.std(accs) * 100)
    results[kind] = (np.array(means), np.array(stds))

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
styles = {
    "additive": ("Additive Gaussian (σ = scale × feature std)", "#1f77b4", "o"),
    "multiplicative": ("Multiplicative (x · (1 + ε))", "#d62728", "s"),
}
for kind, (label, color, marker) in styles.items():
    m, s = results[kind]
    ax.plot(SCALES, m, marker=marker, color=color, label=label, linewidth=2)
    ax.fill_between(SCALES, m - s, m + s, color=color, alpha=0.15)

ax.set_xlabel("Noise scale")
ax.set_ylabel("Accuracy (%)")
ax.set_title("CKD model accuracy vs. input noise (held-out test set)")
ax.set_ylim(78, 101)
ax.grid(alpha=0.3)
ax.legend(loc="lower left")
fig.text(0.01, 0.01, f"Mean over {N_SEEDS} seeds; shaded band = ±1 std. Noise applied to continuous features.",
         fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("noise_robustness.png")

print(pd.DataFrame({"scale": SCALES,
                    "additive_%": results["additive"][0].round(1),
                    "multiplicative_%": results["multiplicative"][0].round(1)}))

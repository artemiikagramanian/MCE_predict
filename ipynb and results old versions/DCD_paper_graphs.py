#!/usr/bin/env python
# coding: utf-8

# In[29]:


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import RFECV

import xgboost as xgb


# ============================================================
# 0. PATHS
# ============================================================

file_path = r"/Users/artemykagramanian/Desktop/ампермагнит/Узбекистан/magnetocaloric_perovskites_full_descriptors.xlsx"
out_dir = r"/Users/artemykagramanian/Desktop/ампермагнит/Узбекистан/DCD_paper"
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# 1. PUBLICATION STYLE FOR 8.5 cm WIDTH
# ============================================================

CM_TO_INCH = 1 / 2.54
COLUMN_WIDTH = 8.5 * CM_TO_INCH

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,

    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,

    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "blue": "#1f4e79",
    "red": "#b22222",
    "green": "#2e7d32",
    "gray": "#666666",
    "black": "#000000",
}


def save_figure(fig, name):
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(out_dir, f"{name}.svg"))
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=600)


def format_axes(ax, grid=True):
    ax.tick_params(direction="in", top=True, right=True)

    if grid:
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


# ============================================================
# 2. PLOT FUNCTIONS
# ============================================================

def plot_delta_s_histogram(delta_s):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.72))

    ax.hist(
        delta_s,
        bins=22,
        color=COLORS["blue"],
        edgecolor=COLORS["black"],
        linewidth=0.45,
        alpha=0.88
    )

    ax.axvline(delta_s.mean(), color=COLORS["red"], linestyle="--",
               linewidth=1.1, label="Mean")
    ax.axvline(delta_s.median(), color=COLORS["black"], linestyle="-",
               linewidth=1.1, label="Median")

    ax.set_xlabel(r"$\Delta S_M$ (J kg$^{-1}$ K$^{-1}$)")
    ax.set_ylabel("Frequency")
    ax.set_title(r"Distribution of $\Delta S_M$")

    ax.legend(frameon=True, edgecolor="black", framealpha=1.0)
    format_axes(ax)

    fig.tight_layout(pad=0.4)
    save_figure(fig, "Fig1_delta_S_histogram")
    plt.show()


def plot_transformation_histogram(y_original):
    y_log = np.log1p(y_original)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.72))

    ax.hist(
        y_original,
        bins=22,
        color=COLORS["blue"],
        edgecolor=COLORS["black"],
        linewidth=0.45,
        alpha=0.88
    )

    ax.set_xlabel(r"$\Delta S_M$ (J kg$^{-1}$ K$^{-1}$)")
    ax.set_ylabel("Frequency")
    ax.set_title(r"Original $\Delta S_M$ distribution")

    format_axes(ax)
    fig.tight_layout(pad=0.4)
    save_figure(fig, "Fig2_original_distribution")
    plt.show()

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.72))

    ax.hist(
        y_log,
        bins=22,
        color=COLORS["green"],
        edgecolor=COLORS["black"],
        linewidth=0.45,
        alpha=0.88
    )

    ax.set_xlabel(r"$\log(1+\Delta S_M)$")
    ax.set_ylabel("Frequency")
    ax.set_title(r"Log-transformed distribution")

    format_axes(ax)
    fig.tight_layout(pad=0.4)
    save_figure(fig, "Fig3_log_distribution")
    plt.show()


def plot_rfecv_curve(rfecv, optimal_k):
    if rfecv is None:
        print("RFECV plot skipped: RFECV was not performed.")
        return

    mean_scores = rfecv.cv_results_["mean_test_score"]
    n_features = np.arange(1, len(mean_scores) + 1)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.72))

    ax.plot(
        n_features,
        mean_scores,
        marker="o",
        markersize=3.2,
        linewidth=1.2,
        color=COLORS["blue"],
        label=r"Mean CV $R^2$"
    )

    ax.axvline(
        optimal_k,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.1,
        label=f"Optimal: {optimal_k}"
    )

    ax.set_xlabel("Number of selected features")
    ax.set_ylabel(r"Cross-validation $R^2$")
    ax.set_title("RFECV feature selection")

    ax.legend(frameon=True, edgecolor="black", framealpha=1.0)
    format_axes(ax)

    fig.tight_layout(pad=0.4)
    save_figure(fig, "Fig4_rfecv_curve")
    plt.show()


def plot_feature_importance(imp_df, top_n=12):
    imp = imp_df.sort_values("rf_importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 1.35))

    bars = ax.barh(
        imp["feature"],
        imp["rf_importance"],
        color=COLORS["blue"],
        edgecolor=COLORS["black"],
        linewidth=0.45,
        height=0.68
    )

    xmax = imp["rf_importance"].max()
    ax.set_xlim(0, xmax * 1.28)

    for bar, value in zip(bars, imp["rf_importance"]):
        ax.text(
            value + xmax * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=7.5
        )

    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Descriptor")
    ax.set_title("Random Forest feature importance")

    format_axes(ax)
    fig.tight_layout(pad=0.35)
    save_figure(fig, "Fig5_feature_importance_rf")
    plt.show()


def plot_model_comparison(results_df):
    res = results_df.sort_values("R2_transformed", ascending=True)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.85))

    bars = ax.barh(
        res["Model"],
        res["R2_transformed"],
        color=COLORS["blue"],
        edgecolor=COLORS["black"],
        linewidth=0.45,
        height=0.68
    )

    xmax = max(res["R2_transformed"].max(), 0.1)
    ax.set_xlim(0, xmax * 1.25)

    for bar, value in zip(bars, res["R2_transformed"]):
        ax.text(
            value + xmax * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=7.5
        )

    ax.set_xlabel(r"Test $R^2$")
    ax.set_ylabel("Model")
    ax.set_title("Model performance")

    format_axes(ax)
    fig.tight_layout(pad=0.35)
    save_figure(fig, "Fig6_model_comparison")
    plt.show()


def plot_parity(y_true, y_pred, name="rf"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = 0.07 * (max_val - min_val)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH))

    ax.scatter(
        y_true,
        y_pred,
        s=22,
        color=COLORS["green"],
        edgecolor=COLORS["black"],
        linewidth=0.35,
        alpha=0.78
    )

    ax.plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        linestyle="--",
        linewidth=1.1,
        color=COLORS["red"],
        label="Ideal"
    )

    ax.text(
        0.06,
        0.94,
        rf"$R^2$ = {r2:.3f}" + "\n" +
        rf"MAE = {mae:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92
        )
    )

    ax.set_xlabel(r"Experimental $\Delta S_M$" + "\n" + r"(J kg$^{-1}$ K$^{-1}$)")
    ax.set_ylabel(r"Predicted $\Delta S_M$" + "\n" + r"(J kg$^{-1}$ K$^{-1}$)")
    ax.set_title(r"Experimental vs predicted $\Delta S_M$")

    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    ax.set_aspect("equal", adjustable="box")

    ax.legend(frameon=True, edgecolor="black", framealpha=1.0, loc="lower right")
    format_axes(ax)

    fig.tight_layout(pad=0.35)
    save_figure(fig, f"Fig7_parity_plot_{name}")
    plt.show()


def plot_residuals(y_true, y_pred, name="rf"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.72))

    ax.axhline(0, color=COLORS["black"], linewidth=0.9)

    ax.scatter(
        y_true,
        residuals,
        s=22,
        color=COLORS["blue"],
        edgecolor=COLORS["black"],
        linewidth=0.35,
        alpha=0.78
    )

    ax.set_xlabel(r"Experimental $\Delta S_M$" + "\n" + r"(J kg$^{-1}$ K$^{-1}$)")
    ax.set_ylabel("Residual")
    ax.set_title("Residual analysis")

    format_axes(ax)
    fig.tight_layout(pad=0.35)
    save_figure(fig, f"Fig8_residuals_{name}")
    plt.show()


def plot_learning_curve(train_sizes_abs, train_mean, test_mean, train_std, test_std):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.72))

    ax.plot(
        train_sizes_abs,
        train_mean,
        marker="o",
        markersize=3.2,
        linewidth=1.2,
        color=COLORS["blue"],
        label="Training"
    )

    ax.plot(
        train_sizes_abs,
        test_mean,
        marker="s",
        markersize=3.2,
        linewidth=1.2,
        color=COLORS["red"],
        label="CV"
    )

    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        color=COLORS["blue"],
        alpha=0.15,
        linewidth=0
    )

    ax.fill_between(
        train_sizes_abs,
        test_mean - test_std,
        test_mean + test_std,
        color=COLORS["red"],
        alpha=0.15,
        linewidth=0
    )

    ax.set_xlabel("Training set size")
    ax.set_ylabel(r"$R^2$")
    ax.set_title("Learning curve")

    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, edgecolor="black", framealpha=1.0)
    format_axes(ax)

    fig.tight_layout(pad=0.35)
    save_figure(fig, "Fig9_learning_curve_rf")
    plt.show()


# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

print(f"Initial dataset size: {df.shape}")
print("Columns:", list(df.columns))

target = "delta_S_M (J/kg K)"

if target not in df.columns:
    raise KeyError(f"Target column '{target}' not found.")

df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target]).copy()

if "H_ex (T)" in df.columns:
    df["H_ex (T)"] = pd.to_numeric(df["H_ex (T)"], errors="coerce")

print(f"After target cleaning: {len(df)} rows")

drop_cols = [
    "Chem formula",
    "Clean_formula",
    "Ref.",
    "T_c (K)",
    "delta_T_ad (K)",
    "RCP(S) (J/kg)",
    target
]

X = df.drop(columns=drop_cols, errors="ignore")
X = X.select_dtypes(include=[np.number])
X = X.fillna(X.median())

y_original = df[target].values
delta_s = pd.Series(y_original)
delta_s = delta_s[delta_s > 0]

print(f"Initial number of features: {X.shape[1]}")
print("Feature columns:", list(X.columns))

if X.shape[1] == 0:
    raise ValueError("No numeric features found.")

if X.shape[1] > 1:
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column for column in upper.columns
        if any(upper[column] > 0.95)
    ]

    X = X.drop(columns=to_drop)

print(f"Number of features after correlation filtering: {X.shape[1]}")


# ============================================================
# 4. TARGET TRANSFORMATION
# ============================================================

skewness = pd.Series(y_original).skew()

if abs(skewness) > 1:
    print(f"Target distribution is skewed (skew = {skewness:.2f}). Applying log1p transformation.")
    y = np.log1p(y_original)
    target_transformed = True
else:
    print(f"Target distribution is not strongly skewed (skew = {skewness:.2f}).")
    y = y_original.copy()
    target_transformed = False


# ============================================================
# 5. SCALING AND TRAIN/TEST SPLIT
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.1,
    random_state=40
)


# ============================================================
# 6. RFECV FEATURE SELECTION
# ============================================================

if X_train.shape[1] >= 2:
    base_xgb = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
        objective="reg:squarederror"
    )

    rfecv = RFECV(
        estimator=base_xgb,
        step=1,
        cv=5,
        scoring="r2",
        min_features_to_select=5 if X_train.shape[1] >= 5 else 1,
        verbose=1,
        n_jobs=-1
    )

    rfecv.fit(X_train, y_train)

    optimal_k = rfecv.n_features_
    selected_mask = rfecv.support_
    selected_features = X.columns[selected_mask]

    print(f"\nOptimal number of features by RFECV: {optimal_k}")

else:
    print("\nOnly one feature is available. RFECV is skipped.")
    rfecv = None
    optimal_k = 1
    selected_mask = np.array([True])
    selected_features = X.columns[selected_mask]

print("Selected features:")
for f in selected_features:
    print("  -", f)

X_train_sel = X_train[:, selected_mask]
X_test_sel = X_test[:, selected_mask]


# ============================================================
# 7. MODELS
# ============================================================

ridge = RidgeCV(alphas=np.logspace(-2, 2, 30), cv=5)
ridge.fit(X_train_sel, y_train)

lasso = LassoCV(alphas=np.logspace(-2, 2, 30), cv=5, random_state=42)
lasso.fit(X_train_sel, y_train)

elastic = ElasticNetCV(
    alphas=np.logspace(-2, 2, 30),
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
    cv=5,
    random_state=42
)
elastic.fit(X_train_sel, y_train)


rf_param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 7, 10, None],
    "min_samples_leaf": [2, 4, 6, 8],
    "max_features": [0.3, 0.5, 0.7, 1.0]
}

rf_random = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=30,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_random.fit(X_train_sel, y_train)
best_rf = rf_random.best_estimator_


xgb_param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 1, 5, 10],
    "reg_lambda": [1, 5, 10, 20]
}

xgb_random = RandomizedSearchCV(
    xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        objective="reg:squarederror"
    ),
    param_distributions=xgb_param_dist,
    n_iter=30,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=1
)
xgb_random.fit(X_train_sel, y_train)
best_xgb = xgb_random.best_estimator_


voting = VotingRegressor([
    ("rf", best_rf),
    ("xgb", best_xgb)
])
voting.fit(X_train_sel, y_train)


models = {
    "Ridge": ridge,
    "Lasso": lasso,
    "ElasticNet": elastic,
    "Random Forest": best_rf,
    "XGBoost": best_xgb,
    "Voting": voting
}

predictions = {
    name: model.predict(X_test_sel)
    for name, model in models.items()
}


# ============================================================
# 8. METRICS
# ============================================================

results = []

for name, y_pred in predictions.items():
    results.append({
        "Model": name,
        "R2_transformed": r2_score(y_test, y_pred),
        "MAE_transformed": mean_absolute_error(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values("R2_transformed", ascending=False)

print("\nMetrics on test set, transformed scale:")
print(results_df.to_string(index=False))

if target_transformed:
    y_test_orig = np.expm1(y_test)
    predictions_orig = {
        name: np.expm1(pred)
        for name, pred in predictions.items()
    }
else:
    y_test_orig = y_test.copy()
    predictions_orig = predictions.copy()

results_orig = []

for name, y_pred_orig in predictions_orig.items():
    results_orig.append({
        "Model": name,
        "R2_original": r2_score(y_test_orig, y_pred_orig),
        "MAE_original": mean_absolute_error(y_test_orig, y_pred_orig)
    })

results_orig_df = pd.DataFrame(results_orig).sort_values("R2_original", ascending=False)

print("\nMetrics on test set, original units:")
print(results_orig_df.to_string(index=False))


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

if hasattr(best_rf, "feature_importances_"):
    imp_df = pd.DataFrame({
        "feature": selected_features,
        "rf_importance": best_rf.feature_importances_
    }).sort_values("rf_importance", ascending=False)
else:
    imp_df = pd.DataFrame({
        "feature": selected_features,
        "rf_importance": np.abs(ridge.coef_)
    }).sort_values("rf_importance", ascending=False)


# ============================================================
# 10. LEARNING CURVE
# ============================================================

train_sizes_abs, train_scores, test_scores = learning_curve(
    best_rf,
    X_train_sel,
    y_train,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.2, 1.0, 6),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)


# ============================================================
# 11. PLOTS
# ============================================================

plot_delta_s_histogram(delta_s)
plot_transformation_histogram(y_original)
plot_rfecv_curve(rfecv, optimal_k)
plot_feature_importance(imp_df)
plot_model_comparison(results_df)

plot_parity(y_test_orig, predictions_orig["Random Forest"], name="random_forest")
plot_residuals(y_test_orig, predictions_orig["Random Forest"], name="random_forest")

plot_parity(y_test_orig, predictions_orig["XGBoost"], name="xgboost")
plot_residuals(y_test_orig, predictions_orig["XGBoost"], name="xgboost")

plot_parity(y_test_orig, predictions_orig["Voting"], name="voting")

plot_learning_curve(
    train_sizes_abs,
    train_mean,
    test_mean,
    train_std,
    test_std
)


# ============================================================
# 12. SAVE TABLES
# ============================================================

results_df.to_excel(os.path.join(out_dir, "model_metrics_transformed.xlsx"), index=False)
results_orig_df.to_excel(os.path.join(out_dir, "model_metrics_original_units.xlsx"), index=False)
imp_df.to_excel(os.path.join(out_dir, "feature_importance_random_forest.xlsx"), index=False)

print(f"\nAll figures and tables saved to folder: {out_dir}")


# In[ ]:
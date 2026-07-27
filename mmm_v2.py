"""
MMM (Marketing Mix Model) — Ridge Regression Baseline
with Intercept, Saturation Points, and Scenario Analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
FILE_PATH = "data/mmm_dataset.csv"        # <-- change path if needed
OUTPUT_DIR = "output_ridge"
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

SPEND_COLS = [
    "competitor_spend", "instagram_spend", "google_ads_spend",
    "tv_spend", "youtube_spend", "newspaper_spend",
    "influencer_spend", "ott_spend"
]
MEDIA_COLS = [c for c in SPEND_COLS if c != "competitor_spend"]
CONTROL_SPEND_COLS = ["competitor_spend"]

CATEGORICAL_COLS = ["holiday", "sales_promotion"]
TARGET_COL = "sales"
DATE_COL = "date"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# STEP 1: LOAD + CLEAN (with date imputation)
# ------------------------------------------------------------------
def impute_dates_sequentially(df, date_col="date", default_start="2020-01-05"):
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    valid_mask = parsed.notna()
    if not valid_mask.any():
        print(f"[INFO] No valid dates found. Generating weekly dates from {default_start}.")
        df[date_col] = pd.date_range(start=default_start, periods=len(df), freq="7D")
        return df
    first_valid_idx = valid_mask.idxmax()
    first_valid_date = parsed.iloc[first_valid_idx]
    start_date = first_valid_date - pd.Timedelta(days=7 * first_valid_idx)
    new_dates = pd.date_range(start=start_date, periods=len(df), freq="7D")
    df[date_col] = new_dates
    n_imputed = (~valid_mask).sum()
    if n_imputed > 0:
        print(f"[INFO] Imputed {n_imputed} missing/invalid dates using weekly sequence anchored at {first_valid_date}.")
    return df


def load_and_clean(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find '{path}'. Update FILE_PATH.")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Dates
    df = impute_dates_sequentially(df, date_col=DATE_COL)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Media spend: NaN -> 0
    for col in MEDIA_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Competitor spend: median imputation
    for col in CONTROL_SPEND_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Sales: drop missing (never impute target)
    before = len(df)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    if before - len(df):
        print(f"[INFO] Dropped {before - len(df)} row(s) with missing '{TARGET_COL}'.")

    # Categorical: fill missing with 'unknown'
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", "unknown").fillna("unknown")
    if "holiday" in df.columns:
        df["holiday"] = pd.to_numeric(df["holiday"], errors="coerce").fillna(0).astype(int)

    return df


# ------------------------------------------------------------------
# STEP 2: ADSTOCK + SATURATION TRANSFORMS
# ------------------------------------------------------------------
def adstock_transform(x, decay_rate=0.5):
    adstocked = np.zeros_like(x, dtype=float)
    adstocked[0] = x[0]
    for t in range(1, len(x)):
        adstocked[t] = x[t] + decay_rate * adstocked[t - 1]
    return adstocked


def saturation_transform(x, alpha=1.0):
    return np.log1p(alpha * x)


def apply_media_transforms(df, decay_rates=None, alphas=None):
    if decay_rates is None:
        decay_rates = {
            "tv_spend": 0.6,
            "ott_spend": 0.6,
            "newspaper_spend": 0.5,
            "youtube_spend": 0.4,
            "instagram_spend": 0.3,
            "influencer_spend": 0.3,
            "google_ads_spend": 0.2,
        }
    if alphas is None:
        alphas = {col: 1.0 / (df[col].std() + 1e-9) for col in MEDIA_COLS}

    transformed = df.copy()
    for col in MEDIA_COLS:
        if col not in df.columns:
            continue
        raw = df[col].values.astype(float)
        ads = adstock_transform(raw, decay_rate=decay_rates.get(col, 0.4))
        sat = saturation_transform(ads, alpha=alphas.get(col, 1.0))
        transformed[f"{col}_transformed"] = sat
    return transformed, decay_rates, alphas


# ------------------------------------------------------------------
# STEP 3: FEATURE ENGINEERING
# ------------------------------------------------------------------
def build_feature_matrix(df):
    feature_cols = [f"{c}_transformed" for c in MEDIA_COLS if f"{c}_transformed" in df.columns]
    feature_cols += CONTROL_SPEND_COLS

    promo_dummies = pd.get_dummies(df["sales_promotion"], prefix="promo", drop_first=True)
    X = pd.concat([df[feature_cols], df[["holiday"]], promo_dummies], axis=1)

    X["time_trend"] = np.arange(len(df))
    week_of_year = df[DATE_COL].dt.isocalendar().week.astype(int)
    X["week_sin"] = np.sin(2 * np.pi * week_of_year / 52)
    X["week_cos"] = np.cos(2 * np.pi * week_of_year / 52)

    y = df[TARGET_COL].values
    return X, y


# ------------------------------------------------------------------
# STEP 4: MODEL FITTING (Ridge with time-series CV)
# ------------------------------------------------------------------
def fit_ridge_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {"alpha": np.logspace(-2, 3, 30)}

    grid = GridSearchCV(
        Ridge(), param_grid, cv=tscv,
        scoring="r2", n_jobs=-1
    )
    grid.fit(X_scaled, y)

    best_model = grid.best_estimator_
    print(f"\n[INFO] Best Ridge alpha: {grid.best_params_['alpha']:.4f}")

    y_pred = best_model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)

    print(f"[INFO] In-sample R^2:   {r2:.4f}")
    print(f"[INFO] In-sample MAE:   {mae:,.0f}")
    print(f"[INFO] In-sample MAPE:  {mape:.2%}")
    print(f"[INFO] Intercept:        {best_model.intercept_:,.0f}")

    return best_model, scaler, y_pred, {"r2": r2, "mae": mae, "mape": mape}


# ------------------------------------------------------------------
# STEP 5: COEFFICIENTS, SATURATION POINTS, SCENARIO ANALYSIS
# ------------------------------------------------------------------
def report_coefficients(model, X):
    coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
    print("\n" + "=" * 70)
    print("STANDARDIZED RIDGE COEFFICIENTS")
    print("=" * 70)
    print(coefs)
    coefs.to_csv(os.path.join(OUTPUT_DIR, "ridge_coefficients.csv"))

    plt.figure(figsize=(9, 6))
    colors = ["seagreen" if v > 0 else "firebrick" for v in coefs.values]
    plt.barh(coefs.index, coefs.values, color=colors)
    plt.title("Ridge Regression Coefficients (Standardized)")
    plt.xlabel("Coefficient value")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ridge_coefficients.png"), dpi=150)
    plt.close()
    return coefs


def compute_saturation_points(alphas):
    """
    Saturation point defined as spend level where marginal response
    drops to 10% of its value at spend=0.
    For saturation transform log1p(alpha * x), derivative = alpha/(1+alpha*x).
    Set derivative = 0.1 * derivative(0) => 1/(1+alpha*x) = 0.1 => x = 9/alpha.
    """
    points = {}
    for col, alpha in alphas.items():
        points[col] = 9.0 / alpha
    return points


def scenario_doubling(df_original, df_transformed, X, scaler, model, decay_rates, alphas):
    """
    For each media channel, double its spend, recompute the transformed column,
    rebuild X, scale, predict, and compute lift vs. original.
    Returns a DataFrame with results.
    """
    results = []
    original_pred = model.predict(scaler.transform(X))
    original_total_sales = original_pred.sum()  # total sales over period

    for col in MEDIA_COLS:
        # Copy original dataframe
        df_mod = df_original.copy()
        # Double the spend for this channel
        df_mod[col] = df_mod[col] * 2.0

        # Re-apply transforms only for this channel (keeping others from df_transformed)
        # We'll build a fresh transformed dataframe from df_mod but only replace this channel's transformed column
        # Easiest: run full transform on df_mod and then extract the new column
        df_mod_trans, _, _ = apply_media_transforms(df_mod, decay_rates, alphas)
        new_trans_col = f"{col}_transformed"

        # Build X_mod by replacing the column in original X
        X_mod = X.copy()
        X_mod[new_trans_col] = df_mod_trans[new_trans_col].values

        # Scale and predict
        X_mod_scaled = scaler.transform(X_mod)
        pred_mod = model.predict(X_mod_scaled)
        total_sales_mod = pred_mod.sum()
        lift = total_sales_mod - original_total_sales

        # Extra spend: sum of (2*spend - spend) = sum(original spend)
        extra_spend = df_original[col].sum()
        roi = lift / extra_spend if extra_spend > 0 else np.nan

        results.append({
            "channel": col,
            "total_extra_spend": extra_spend,
            "total_sales_lift": lift,
            "lift_percent": (lift / original_total_sales) * 100,
            "ROI": roi
        })

    return pd.DataFrame(results)


def plot_actual_vs_predicted(df, y, y_pred):
    plt.figure(figsize=(12, 5))
    plt.plot(df[DATE_COL], y, label="Actual sales", marker="o", markersize=3)
    plt.plot(df[DATE_COL], y_pred, label="Predicted sales", marker="x", markersize=3, alpha=0.8)
    plt.title("Actual vs. Predicted Sales (Ridge Baseline)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    ensure_output_dir()

    print("Loading and cleaning data...")
    df = load_and_clean(FILE_PATH)
    print(f"Cleaned dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nApplying adstock + saturation transforms...")
    df_transformed, decay_rates, alphas = apply_media_transforms(df)
    print(f"[INFO] Adstock decay rates: {decay_rates}")
    print(f"[INFO] Saturation alphas:   {alphas}")

    print("\nBuilding feature matrix...")
    X, y = build_feature_matrix(df_transformed)
    print(f"Feature matrix shape: {X.shape}")

    print("\nFitting Ridge regression with time-series CV...")
    model, scaler, y_pred, metrics = fit_ridge_model(X, y)

    # --- Coefficients ---
    coefs = report_coefficients(model, X)

    # --- Saturation points ---
    sat_points = compute_saturation_points(alphas)
    sat_df = pd.DataFrame(list(sat_points.items()), columns=["channel", "saturation_spend"])
    sat_df.to_csv(os.path.join(OUTPUT_DIR, "saturation_points.csv"), index=False)
    print("\n" + "=" * 70)
    print("SATURATION POINTS (spend where marginal response = 10% of initial)")
    print("=" * 70)
    print(sat_df.to_string(index=False))

    # --- Scenario: doubling each channel's spend ---
    scenario_df = scenario_doubling(df, df_transformed, X, scaler, model, decay_rates, alphas)
    scenario_df.to_csv(os.path.join(OUTPUT_DIR, "doubling_scenario.csv"), index=False)
    print("\n" + "=" * 70)
    print("SCENARIO: DOUBLE EACH CHANNEL'S SPEND (ceteris paribus)")
    print("=" * 70)
    print(scenario_df.to_string(index=False, float_format="%.0f"))

    # --- Plot actual vs predicted ---
    plot_actual_vs_predicted(df_transformed, y, y_pred)

    # Save cleaned + transformed dataset
    df_transformed.to_csv(os.path.join(OUTPUT_DIR, "cleaned_transformed_data.csv"), index=False)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model fit: R² = {metrics['r2']:.3f}, MAPE = {metrics['mape']:.2%}")
    print(f"Intercept: {model.intercept_:,.0f}")
    print("\nTop 5 features (standardized coefficients):")
    print(coefs.head(5))
    print(f"\nAll outputs saved to: ./{OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

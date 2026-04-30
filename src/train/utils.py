import numpy as np
import pandas as pd


# =========================
# FEATURE ENGINEERING
# =========================
def engineer_features(
    df: pd.DataFrame,
    reference_df: pd.DataFrame = None,
    return_stats: bool = False
) -> pd.DataFrame:
    df = df.copy()

    # =========================
    # TIME FEATURES
    # =========================
    df['time_diff'] = df['Time'].diff().fillna(0)
    df['time_diff'] = df['time_diff'].clip(lower=0, upper=86400)
    df['time_diff_log'] = np.log1p(df['time_diff'])

    # =========================
    # AMOUNT FEATURES (NO LEAKAGE)
    # =========================
    if reference_df is None:
        reference_df = df

    mean_amt = reference_df['Amount'].mean()
    median_amt = reference_df['Amount'].median()
    threshold_95 = reference_df['Amount'].quantile(0.95)

    df['log_amount'] = np.log1p(df['Amount'])
    df['amt_to_mean_ratio'] = df['Amount'] / mean_amt
    df['amt_to_median_ratio'] = df['Amount'] / median_amt
    df['is_high_amount'] = (df['Amount'] > threshold_95).astype(int)

    if return_stats:
        return df, {
            "mean_amt": mean_amt,
            "median_amt": median_amt,
            "threshold_95": threshold_95
        }

    return df


# =========================
# TRAIN STATS
# =========================
def get_train_stats(df: pd.DataFrame) -> dict:
    return {
        "mean_amt": df['Amount'].mean(),
        "median_amt": df['Amount'].median(),
        "threshold_95": df['Amount'].quantile(0.95)
    }


# =========================
# FEATURE LIST
# =========================
def get_feature_columns() -> list:
    return [
        # PCA features
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',

        # amount features
        'log_amount',
        'amt_to_mean_ratio',
        'amt_to_median_ratio',
        'is_high_amount',

        # time features
        'time_diff',
        'time_diff_log'
    ]
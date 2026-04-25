import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_features = df.copy()
    
    df_features['hour_of_day'] = (df_features['Time'] / 3600) % 24
    df_features['hour_of_day'] = df_features['hour_of_day'].astype(int)
    
    df_features['is_night_transaction'] = df_features['hour_of_day'].apply(lambda x: 1 if 2 <= x <= 6 else 0)
    
    mean_amt = df_features['Amount'].mean()
    df_features['amt_to_mean_ratio'] = df_features['Amount'] / mean_amt
    
    threshold_95 = df_features['Amount'].quantile(0.95)
    df_features['is_high_amount'] = (df_features['Amount'] > threshold_95).astype(int)
    
    df_features['log_amount'] = np.log1p(df_features['Amount'])
    
    return df_features


def get_feature_columns() -> list:
    return [
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
        'Amount', 'Time',
        'hour_of_day', 'is_night_transaction', 'amt_to_mean_ratio', 'is_high_amount', 'log_amount'
    ]
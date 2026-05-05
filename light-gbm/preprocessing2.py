#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from preprocessing1 import df, df_clean_outliers, df_outliers_replace

# ─── 1. Feature Engineering ─────────────────────────────────────────────────

df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
col_index = df.columns.get_loc('Loan_Status')
df.insert(col_index, 'Total_Income', df.pop('Total_Income'))

df_clean_outliers['Total_Income'] = df_clean_outliers['ApplicantIncome'] + df_clean_outliers['CoapplicantIncome']
col_index = df_clean_outliers.columns.get_loc('Loan_Status')
df_clean_outliers.insert(col_index, 'Total_Income', df_clean_outliers.pop('Total_Income'))

df_outliers_replace['Total_Income'] = df_outliers_replace['ApplicantIncome'] + df_outliers_replace['CoapplicantIncome']
col_index = df_outliers_replace.columns.get_loc('Loan_Status')
df_outliers_replace.insert(col_index, 'Total_Income', df_outliers_replace.pop('Total_Income'))

# ─── 2. Encoding ────────────────────────────────────────────────────────────

binary_map = {
    'Gender':        {'Male': 1, 'Female': 0},
    'Married':       {'Yes': 1, 'No': 0},
    'Education':     {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Loan_Status':   {'Y': 1, 'N': 0},
    'Property_Area': {'Urban': 1, 'Rural': 2, 'Semiurban': 3},
}

df_encoded = df.copy()
df_encoded_clean_outliers = df_clean_outliers.copy()
df_encoded_outliers_replace = df_outliers_replace.copy()

for col, mapping in binary_map.items():
    df_encoded[col] = df_encoded[col].map(mapping)
    df_encoded_clean_outliers[col] = df_encoded_clean_outliers[col].map(mapping)
    df_encoded_outliers_replace[col] = df_encoded_outliers_replace[col].map(mapping)

# kolom numerik (exclude Loan_Status) — dipakai semua transformasi
num_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns
num_cols = num_cols.drop('Loan_Status')

# ─── 3. Z-Score Transformation ──────────────────────────────────────────────

df_zscore = df_encoded.copy()
df_zscore[num_cols] = StandardScaler().fit_transform(df_zscore[num_cols])

df_zscore_drop = df_encoded_clean_outliers.copy()
df_zscore_drop[num_cols] = StandardScaler().fit_transform(df_zscore_drop[num_cols])

df_zscore_replace = df_encoded_outliers_replace.copy()
df_zscore_replace[num_cols] = StandardScaler().fit_transform(df_zscore_replace[num_cols])

# ─── 4. Log Transformation ──────────────────────────────────────────────────

df_log = df_encoded.copy()
for col in num_cols:
    df_log[col] = np.log1p(df_log[col])

df_log_drop = df_encoded_clean_outliers.copy()
for col in num_cols:
    df_log_drop[col] = np.log1p(df_log_drop[col])

df_log_replace = df_encoded_outliers_replace.copy()
for col in num_cols:
    df_log_replace[col] = np.log1p(df_log_replace[col])

# ─── 5. MinMax Scaling ──────────────────────────────────────────────────────

df_minmax = df_encoded.copy()
df_minmax[num_cols] = MinMaxScaler().fit_transform(df_minmax[num_cols])

df_minmax_drop = df_encoded_clean_outliers.copy()
df_minmax_drop[num_cols] = MinMaxScaler().fit_transform(df_minmax_drop[num_cols])

df_minmax_replace = df_encoded_outliers_replace.copy()
df_minmax_replace[num_cols] = MinMaxScaler().fit_transform(df_minmax_replace[num_cols])

# ─── 6. Decimal Scaling ─────────────────────────────────────────────────────

def decimal_scaling(dataframe, cols):
    df_scaled = dataframe.copy()
    for col in cols:
        max_val = df_scaled[col].abs().max()
        j = len(str(int(max_val)))
        df_scaled[col] = df_scaled[col] / (10 ** j)
    return df_scaled

df_decimal         = decimal_scaling(df_encoded, num_cols)
df_decimal_drop    = decimal_scaling(df_encoded_clean_outliers, num_cols)
df_decimal_replace = decimal_scaling(df_encoded_outliers_replace, num_cols)


# ─── Hanya jalan kalau dirun langsung, TIDAK waktu diimport ─────────────────

if __name__ == "__main__":
    print("=== Z-Score ===")
    print(df_zscore.head())
    print(df_zscore_drop.head())
    print(df_zscore_replace.head())

    print("\n=== Log Transform ===")
    print(df_log.head())
    print(df_log_drop.head())
    print(df_log_replace.head())

    print("\n=== MinMax ===")
    print(df_minmax.head())
    print(df_minmax_drop.head())
    print(df_minmax_replace.head())

    print("\n=== Decimal Scaling ===")
    print(df_decimal.head())
    print(df_decimal_drop.head())
    print(df_decimal_replace.head())

    print("\n=== Info df_zscore_drop ===")
    df_zscore_drop.info()

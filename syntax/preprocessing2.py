from preprocessing1 import df, df_clean_outliers, df_outliers_replace
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# --- Base: df ---
df_ti = df.copy()
df_ti['Total_Income'] = df_ti['ApplicantIncome'] + df_ti['CoapplicantIncome']
if 'Loan_Status' in df_ti.columns:
    col_index = df_ti.columns.get_loc('Loan_Status')
    df_ti.insert(col_index, 'Total_Income', df_ti.pop('Total_Income'))

df_lair = df.copy()
df_lair['Loan_Amount_Income_Ratio'] = (df_lair['LoanAmount'] / (df_lair['ApplicantIncome'] + df_lair['CoapplicantIncome'])) + 1
if 'Loan_Status' in df_lair.columns:
    col_index = df_lair.columns.get_loc('Loan_Status')
    df_lair.insert(col_index, 'Loan_Amount_Income_Ratio', df_lair.pop('Loan_Amount_Income_Ratio'))

df_both = df.copy()
df_both['Total_Income'] = df_both['ApplicantIncome'] + df_both['CoapplicantIncome']
df_both['Loan_Amount_Income_Ratio'] = (df_both['LoanAmount'] / df_both['Total_Income']) + 1
if 'Loan_Status' in df_both.columns:
    col_index = df_both.columns.get_loc('Loan_Status')
    df_both.insert(col_index, 'Total_Income', df_both.pop('Total_Income'))
    col_index = df_both.columns.get_loc('Loan_Status')
    df_both.insert(col_index, 'Loan_Amount_Income_Ratio', df_both.pop('Loan_Amount_Income_Ratio'))

# --- Base: df_clean_outliers ---
df_ti_drop = df_clean_outliers.copy()
df_ti_drop['Total_Income'] = df_ti_drop['ApplicantIncome'] + df_ti_drop['CoapplicantIncome']
if 'Loan_Status' in df_ti_drop.columns:
    col_index = df_ti_drop.columns.get_loc('Loan_Status')
    df_ti_drop.insert(col_index, 'Total_Income', df_ti_drop.pop('Total_Income'))

df_lair_drop = df_clean_outliers.copy()
df_lair_drop['Loan_Amount_Income_Ratio'] = (df_lair_drop['LoanAmount'] / (df_lair_drop['ApplicantIncome'] + df_lair_drop['CoapplicantIncome'])) + 1
if 'Loan_Status' in df_lair_drop.columns:
    col_index = df_lair_drop.columns.get_loc('Loan_Status')
    df_lair_drop.insert(col_index, 'Loan_Amount_Income_Ratio', df_lair_drop.pop('Loan_Amount_Income_Ratio'))

df_both_drop = df_clean_outliers.copy()
df_both_drop['Total_Income'] = df_both_drop['ApplicantIncome'] + df_both_drop['CoapplicantIncome']
df_both_drop['Loan_Amount_Income_Ratio'] = (df_both_drop['LoanAmount'] / df_both_drop['Total_Income']) + 1
if 'Loan_Status' in df_both_drop.columns:
    col_index = df_both_drop.columns.get_loc('Loan_Status')
    df_both_drop.insert(col_index, 'Total_Income', df_both_drop.pop('Total_Income'))
    col_index = df_both_drop.columns.get_loc('Loan_Status')
    df_both_drop.insert(col_index, 'Loan_Amount_Income_Ratio', df_both_drop.pop('Loan_Amount_Income_Ratio'))

# --- Base: df_outliers_replace ---
df_ti_replace = df_outliers_replace.copy()
df_ti_replace['Total_Income'] = df_ti_replace['ApplicantIncome'] + df_ti_replace['CoapplicantIncome']
if 'Loan_Status' in df_ti_replace.columns:
    col_index = df_ti_replace.columns.get_loc('Loan_Status')
    df_ti_replace.insert(col_index, 'Total_Income', df_ti_replace.pop('Total_Income'))

df_lair_replace = df_outliers_replace.copy()
df_lair_replace['Loan_Amount_Income_Ratio'] = (df_lair_replace['LoanAmount'] / (df_lair_replace['ApplicantIncome'] + df_lair_replace['CoapplicantIncome'])) + 1
if 'Loan_Status' in df_lair_replace.columns:
    col_index = df_lair_replace.columns.get_loc('Loan_Status')
    df_lair_replace.insert(col_index, 'Loan_Amount_Income_Ratio', df_lair_replace.pop('Loan_Amount_Income_Ratio'))

df_both_replace = df_outliers_replace.copy()
df_both_replace['Total_Income'] = df_both_replace['ApplicantIncome'] + df_both_replace['CoapplicantIncome']
df_both_replace['Loan_Amount_Income_Ratio'] = (df_both_replace['LoanAmount'] / df_both_replace['Total_Income']) + 1
if 'Loan_Status' in df_both_replace.columns:
    col_index = df_both_replace.columns.get_loc('Loan_Status')
    df_both_replace.insert(col_index, 'Total_Income', df_both_replace.pop('Total_Income'))
    col_index = df_both_replace.columns.get_loc('Loan_Status')
    df_both_replace.insert(col_index, 'Loan_Amount_Income_Ratio', df_both_replace.pop('Loan_Amount_Income_Ratio'))



def binary_encode(df_in):
    df_out = df_in.copy()
    if 'Gender' in df_out.columns: df_out['Gender'] = df_out['Gender'].map({'Male': 1, 'Female': 0})
    if 'Married' in df_out.columns: df_out['Married'] = df_out['Married'].map({'Yes': 1, 'No': 0})
    if 'Education' in df_out.columns: df_out['Education'] = df_out['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    if 'Self_Employed' in df_out.columns: df_out['Self_Employed'] = df_out['Self_Employed'].map({'Yes': 1, 'No': 0})
    if 'Loan_Status' in df_out.columns: df_out['Loan_Status'] = df_out['Loan_Status'].map({'Y': 1, 'N': 0})
    if 'Property_Area' in df_out.columns: df_out['Property_Area'] = df_out['Property_Area'].map({'Urban': 1, 'Rural': 2, 'Semiurban' : 3})
    return df_out

df_encoded_ti = binary_encode(df_ti)
df_encoded_lair = binary_encode(df_lair)
df_encoded_both = binary_encode(df_both)
df_encoded_ti_drop = binary_encode(df_ti_drop)
df_encoded_lair_drop = binary_encode(df_lair_drop)
df_encoded_both_drop = binary_encode(df_both_drop)
df_encoded_ti_replace = binary_encode(df_ti_replace)
df_encoded_lair_replace = binary_encode(df_lair_replace)
df_encoded_both_replace = binary_encode(df_both_replace)


from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_data(df_encoded):
    num_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns.drop('Loan_Status', errors='ignore')
    
    # Z-Score
    df_z = df_encoded.copy()
    scaler_z = StandardScaler()
    if len(num_cols) > 0:
        df_z[num_cols] = scaler_z.fit_transform(df_z[num_cols])
        
    # Log
    df_log = df_encoded.copy()
    for c in num_cols:
        df_log[c] = np.log1p(np.maximum(df_log[c], 0))

    # MinMax
    df_mm = df_encoded.copy()
    scaler_mm = MinMaxScaler()
    if len(num_cols) > 0:
        df_mm[num_cols] = scaler_mm.fit_transform(df_mm[num_cols])
        
    # Decimal
    df_dec = df_encoded.copy()
    for c in num_cols:
        max_val = df_dec[c].abs().max()
        if max_val > 0:
            j = len(str(int(max_val)))
            df_dec[c] = df_dec[c] / (10 ** j)
            
    return df_z, df_log, df_mm, df_dec

df_ti_zscore, df_ti_log, df_ti_minmax, df_ti_decimal = scale_data(df_encoded_ti)
df_lair_zscore, df_lair_log, df_lair_minmax, df_lair_decimal = scale_data(df_encoded_lair)
df_both_zscore, df_both_log, df_both_minmax, df_both_decimal = scale_data(df_encoded_both)
df_ti_zscore_drop, df_ti_log_drop, df_ti_minmax_drop, df_ti_decimal_drop = scale_data(df_encoded_ti_drop)
df_lair_zscore_drop, df_lair_log_drop, df_lair_minmax_drop, df_lair_decimal_drop = scale_data(df_encoded_lair_drop)
df_both_zscore_drop, df_both_log_drop, df_both_minmax_drop, df_both_decimal_drop = scale_data(df_encoded_both_drop)
df_ti_zscore_replace, df_ti_log_replace, df_ti_minmax_replace, df_ti_decimal_replace = scale_data(df_encoded_ti_replace)
df_lair_zscore_replace, df_lair_log_replace, df_lair_minmax_replace, df_lair_decimal_replace = scale_data(df_encoded_lair_replace)
df_both_zscore_replace, df_both_log_replace, df_both_minmax_replace, df_both_decimal_replace = scale_data(df_encoded_both_replace)



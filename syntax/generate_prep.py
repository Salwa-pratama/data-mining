import json

base_dfs = ['df', 'df_clean_outliers', 'df_outliers_replace']
suffixes = ['', '_drop', '_replace']

cells = []

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from preprocessing1 import df, df_clean_outliers, df_outliers_replace\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n"
    ]
})

feature_code = []
for base, suf in zip(base_dfs, suffixes):
    feature_code.append(f"# --- Base: {base} ---")
    
    # TI only
    feature_code.append(f"df_ti{suf} = {base}.copy()")
    feature_code.append(f"df_ti{suf}['Total_Income'] = df_ti{suf}['ApplicantIncome'] + df_ti{suf}['CoapplicantIncome']")
    feature_code.append(f"if 'Loan_Status' in df_ti{suf}.columns:")
    feature_code.append(f"    col_index = df_ti{suf}.columns.get_loc('Loan_Status')")
    feature_code.append(f"    df_ti{suf}.insert(col_index, 'Total_Income', df_ti{suf}.pop('Total_Income'))\n")
    
    # LAIR only
    feature_code.append(f"df_lair{suf} = {base}.copy()")
    feature_code.append(f"df_lair{suf}['Loan_Amount_Income_Ratio'] = (df_lair{suf}['LoanAmount'] / (df_lair{suf}['ApplicantIncome'] + df_lair{suf}['CoapplicantIncome'])) + 1")
    feature_code.append(f"if 'Loan_Status' in df_lair{suf}.columns:")
    feature_code.append(f"    col_index = df_lair{suf}.columns.get_loc('Loan_Status')")
    feature_code.append(f"    df_lair{suf}.insert(col_index, 'Loan_Amount_Income_Ratio', df_lair{suf}.pop('Loan_Amount_Income_Ratio'))\n")

    # BOTH
    feature_code.append(f"df_both{suf} = {base}.copy()")
    feature_code.append(f"df_both{suf}['Total_Income'] = df_both{suf}['ApplicantIncome'] + df_both{suf}['CoapplicantIncome']")
    feature_code.append(f"df_both{suf}['Loan_Amount_Income_Ratio'] = (df_both{suf}['LoanAmount'] / df_both{suf}['Total_Income']) + 1")
    feature_code.append(f"if 'Loan_Status' in df_both{suf}.columns:")
    feature_code.append(f"    col_index = df_both{suf}.columns.get_loc('Loan_Status')")
    feature_code.append(f"    df_both{suf}.insert(col_index, 'Total_Income', df_both{suf}.pop('Total_Income'))")
    feature_code.append(f"    col_index = df_both{suf}.columns.get_loc('Loan_Status')")
    feature_code.append(f"    df_both{suf}.insert(col_index, 'Loan_Amount_Income_Ratio', df_both{suf}.pop('Loan_Amount_Income_Ratio'))\n")

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in feature_code]
})

encode_code = []
encode_code.append("def binary_encode(df_in):\n    df_out = df_in.copy()\n    if 'Gender' in df_out.columns: df_out['Gender'] = df_out['Gender'].map({'Male': 1, 'Female': 0})\n    if 'Married' in df_out.columns: df_out['Married'] = df_out['Married'].map({'Yes': 1, 'No': 0})\n    if 'Education' in df_out.columns: df_out['Education'] = df_out['Education'].map({'Graduate': 1, 'Not Graduate': 0})\n    if 'Self_Employed' in df_out.columns: df_out['Self_Employed'] = df_out['Self_Employed'].map({'Yes': 1, 'No': 0})\n    if 'Loan_Status' in df_out.columns: df_out['Loan_Status'] = df_out['Loan_Status'].map({'Y': 1, 'N': 0})\n    if 'Property_Area' in df_out.columns: df_out['Property_Area'] = df_out['Property_Area'].map({'Urban': 1, 'Rural': 2, 'Semiurban' : 3})\n    return df_out\n")

variants = ['ti', 'lair', 'both']
encoded_vars = []

for suf in suffixes:
    for var in variants:
        encode_code.append(f"df_encoded_{var}{suf} = binary_encode(df_{var}{suf})")
        encoded_vars.append(f"df_encoded_{var}{suf}")

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in encode_code]
})

scaling_code = []
scaling_code.append("from sklearn.preprocessing import StandardScaler, MinMaxScaler\n")

scaling_code.append("def scale_data(df_encoded):\n    num_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns.drop('Loan_Status', errors='ignore')\n    \n    # Z-Score\n    df_z = df_encoded.copy()\n    scaler_z = StandardScaler()\n    if len(num_cols) > 0:\n        df_z[num_cols] = scaler_z.fit_transform(df_z[num_cols])\n        \n    # Log\n    df_log = df_encoded.copy()\n    for c in num_cols:\n        df_log[c] = np.log1p(np.maximum(df_log[c], 0))\n\n    # MinMax\n    df_mm = df_encoded.copy()\n    scaler_mm = MinMaxScaler()\n    if len(num_cols) > 0:\n        df_mm[num_cols] = scaler_mm.fit_transform(df_mm[num_cols])\n        \n    # Decimal\n    df_dec = df_encoded.copy()\n    for c in num_cols:\n        max_val = df_dec[c].abs().max()\n        if max_val > 0:\n            j = len(str(int(max_val)))\n            df_dec[c] = df_dec[c] / (10 ** j)\n            \n    return df_z, df_log, df_mm, df_dec\n")

exported_vars = []

for suf in suffixes:
    for var in variants:
        enc_name = f"df_encoded_{var}{suf}"
        z_name = f"df_{var}_zscore{suf}"
        log_name = f"df_{var}_log{suf}"
        mm_name = f"df_{var}_minmax{suf}"
        dec_name = f"df_{var}_decimal{suf}"
        
        scaling_code.append(f"{z_name}, {log_name}, {mm_name}, {dec_name} = scale_data({enc_name})")
        
        exported_vars.append(z_name)
        exported_vars.append(log_name)
        exported_vars.append(mm_name)
        exported_vars.append(dec_name)

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in scaling_code]
})

nb_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("preprocessing2.ipynb", "w") as f:
    json.dump(nb_dict, f, indent=1)

with open("exported_vars.txt", "w") as f:
    f.write(", ".join(exported_vars))


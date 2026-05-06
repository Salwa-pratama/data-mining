import json

exported_vars = [
    "df_ti_zscore", "df_ti_log", "df_ti_minmax", "df_ti_decimal", 
    "df_lair_zscore", "df_lair_log", "df_lair_minmax", "df_lair_decimal", 
    "df_both_zscore", "df_both_log", "df_both_minmax", "df_both_decimal", 
    "df_ti_zscore_drop", "df_ti_log_drop", "df_ti_minmax_drop", "df_ti_decimal_drop", 
    "df_lair_zscore_drop", "df_lair_log_drop", "df_lair_minmax_drop", "df_lair_decimal_drop", 
    "df_both_zscore_drop", "df_both_log_drop", "df_both_minmax_drop", "df_both_decimal_drop", 
    "df_ti_zscore_replace", "df_ti_log_replace", "df_ti_minmax_replace", "df_ti_decimal_replace", 
    "df_lair_zscore_replace", "df_lair_log_replace", "df_lair_minmax_replace", "df_lair_decimal_replace", 
    "df_both_zscore_replace", "df_both_log_replace", "df_both_minmax_replace", "df_both_decimal_replace"
]

import_code = "from preprocessing2 import (\n    " + ",\n    ".join([
    ", ".join(exported_vars[i:i+4]) for i in range(0, len(exported_vars), 4)
]) + "\n)\nprint('Semua 36 dataset berhasil diimport!')\n"

dataset_dict_code = "datasets = {\n"
for v in exported_vars:
    # Generate pretty name
    name_parts = []
    if "_ti" in v: name_parts.append("TI")
    elif "_lair" in v: name_parts.append("LAIR")
    elif "_both" in v: name_parts.append("BOTH")
    
    if "zscore" in v: name_parts.append("ZScore")
    elif "log" in v: name_parts.append("Log")
    elif "minmax" in v: name_parts.append("MinMax")
    elif "decimal" in v: name_parts.append("Decimal")
    
    if "drop" in v: name_parts.append("(Drop)")
    elif "replace" in v: name_parts.append("(Replace)")
    else: name_parts.append("(Original)")
    
    pretty_name = " ".join(name_parts)
    dataset_dict_code += f"    '{pretty_name}': {v},\n"
dataset_dict_code += "}\n\n"
dataset_dict_code += "results = {}\nfor name, df_data in datasets.items():\n    print(f'Training SVR: {name} ...')\n    results[name] = run_svr(df_data, name)\n\nprint('\\nSemua model selesai ditraining!')\n"

def fix_predict1(filename):
    with open(filename, 'r') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source_text = "".join(cell['source'])
            
            if "from preprocessing2 import" in source_text:
                cell['source'] = [import_code]
            elif "datasets =" in source_text and "results =" in source_text:
                cell['source'] = [dataset_dict_code]
            elif "new_data_raw =" in source_text:
                # Add Loan_Amount_Income_Ratio to new_data_raw
                if "'Total_Income'" in source_text and "'Loan_Amount_Income_Ratio'" not in source_text:
                    new_src = source_text.replace(
                        "    'Total_Income'      : 2000,   # ApplicantIncome + CoapplicantIncome\n",
                        "    'Total_Income'      : 2000,\n    'Loan_Amount_Income_Ratio': (150/2000) + 1,\n"
                    )
                    cell['source'] = [new_src]
                # Also ensure if pred_df is empty, print something graceful
                if "pred_df = pd.DataFrame(new_predictions)" in source_text:
                    # Not really needed if we fix the input features
                    pass

    with open(filename, 'w') as f:
        json.dump(nb, f, indent=1)

fix_predict1('predict1.ipynb')

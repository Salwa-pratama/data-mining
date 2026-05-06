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
dataset_dict_code += "results = {}\nfor name, data in datasets.items():\n    print(f'Training: {name} ...')\n    results[name] = run_svr(data, name)\n\nprint('\\nSemua model selesai ditraining!')\n"

def process_nb(filename, is_lgbm=False):
    with open(filename, 'r') as f:
        nb = json.load(f)
        
    new_cells = []
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            source_text = "".join(source_lines)
            
            # Remove shap imports and init
            source_text = source_text.replace("import shap\n", "")
            source_text = source_text.replace("shap.initjs()\n", "")
            
            # Remove shap plots logic completely
            if "shap.Explainer" in source_text or "shap.summary_plot" in source_text or "shap.force_plot" in source_text:
                continue # Skip SHAP cell entirely
            
            if "from preprocessing2 import" in source_text:
                cell['source'] = [import_code]
            elif "datasets =" in source_text and "results =" in source_text:
                if is_lgbm:
                    lgbm_loop = dataset_dict_code.replace("run_svr(", "run_lgbm(")
                    cell['source'] = [lgbm_loop]
                else:
                    cell['source'] = [dataset_dict_code]
            else:
                cell['source'] = [source_text]
                
            new_cells.append(cell)
        else:
            new_cells.append(cell)
            
    nb['cells'] = new_cells
    
    with open(filename, 'w') as f:
        json.dump(nb, f, indent=1)

process_nb('predict.ipynb', False)
process_nb('predict_lightgbm.ipynb', True)


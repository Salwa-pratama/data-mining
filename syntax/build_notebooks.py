import json

# 1. Update preprocessing2.ipynb
with open('preprocessing2.ipynb', 'r', encoding='utf-8') as f:
    nb_prep = json.load(f)

for cell in nb_prep.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']" in source and "Loan_Amount_Income_Ratio" not in source:
            new_source = source + "\n# Tambah feature Loan_Amount_Income_Ratio\n"
            new_source += "df['Loan_Amount_Income_Ratio'] = (df['LoanAmount'] / df['Total_Income']) + 1\n"
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]
            
        elif "df_clean_outliers['Total_Income'] = df_clean_outliers['ApplicantIncome'] + df_clean_outliers['CoapplicantIncome']" in source and "Loan_Amount_Income_Ratio" not in source:
            new_source = source + "\n# Tambah feature Loan_Amount_Income_Ratio\n"
            new_source += "df_clean_outliers['Loan_Amount_Income_Ratio'] = (df_clean_outliers['LoanAmount'] / df_clean_outliers['Total_Income']) + 1\n"
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]

        elif "df_outliers_replace['Total_Income'] = df_outliers_replace['ApplicantIncome'] + df_outliers_replace['CoapplicantIncome']" in source and "Loan_Amount_Income_Ratio" not in source:
            new_source = source + "\n# Tambah feature Loan_Amount_Income_Ratio\n"
            new_source += "df_outliers_replace['Loan_Amount_Income_Ratio'] = (df_outliers_replace['LoanAmount'] / df_outliers_replace['Total_Income']) + 1\n"
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]

with open('preprocessing2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_prep, f, indent=1)

# 2. Export to preprocessing2.py
py_lines = [
    "import pandas as pd",
    "import numpy as np",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler",
    "from preprocessing1 import df, df_clean_outliers, df_outliers_replace\n"
]

for cell in nb_prep.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Filter out display calls or info prints
        lines = source.split('\n')
        clean_lines = []
        for line in lines:
            if line.strip().startswith('df') and line.strip().endswith('.head()'):
                continue
            if line.strip().startswith('df') and line.strip().endswith('.info()'):
                continue
            if line.strip().startswith('print('):
                continue
            clean_lines.append(line)
        py_lines.extend(clean_lines)
        py_lines.append('\n')

with open('preprocessing2.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(py_lines))


# 3. Update predict.ipynb
with open('predict.ipynb', 'r', encoding='utf-8') as f:
    nb_pred = json.load(f)

new_cells = []
for cell in nb_pred.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Drop SHAP installation
        if "pip" in source and "shap" in source:
            continue
        # Modify import
        if "import shap" in source:
            source = source.replace("import shap\n", "")
            cell['source'] = [line + '\n' for line in source.split('\n') if line]
        
        # Drop SHAP blocks
        if "shap_results =" in source or "shap.summary_plot" in source or "shap.plots.waterfall" in source or "shap.force_plot" in source or "importance_records =" in source or "Heatmap Feature Importance SHAP" in source:
            continue
        
        # Modify new data to include Loan_Amount_Income_Ratio
        if "new_data_raw = {" in source:
            if "Loan_Amount_Income_Ratio" not in source:
                source = source.replace("'Total_Income'      : 6500,", "'Total_Income'      : 6500,\n    'Loan_Amount_Income_Ratio': (150/6500) + 1,")
                cell['source'] = [line + '\n' for line in source.split('\n') if line]

        # Clean up output printing of SHAP
        if "Fitur paling berpengaruh" in source:
            lines = source.split('\n')
            lines = [l for l in lines if "Fitur paling berpengaruh" not in l and "top_features" not in l and "for i, (feat, val)" not in l and "print(f'   {i:2}" not in l]
            cell['source'] = [line + '\n' for line in lines if line]
            
    new_cells.append(cell)

nb_pred['cells'] = new_cells

with open('predict.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_pred, f, indent=1)


# 4. Create predict_lightgbm.ipynb
nb_lgbm = json.loads(json.dumps(nb_pred)) # Deep copy

for cell in nb_lgbm.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        if "from sklearn.svm import SVR, SVC" in source:
            source = source.replace("from sklearn.svm import SVR, SVC", "from lightgbm import LGBMClassifier")
            cell['source'] = [line + '\n' for line in source.split('\n') if line]
            
        if "def run_svr(" in source:
            source = source.replace("def run_svr(", "def run_lgbm(")
            source = source.replace("svr = SVC(kernel='rbf', C=1.0)", "lgbm = LGBMClassifier(random_state=42)")
            source = source.replace("svr.fit(", "lgbm.fit(")
            source = source.replace("svr.predict(", "lgbm.predict(")
            source = source.replace("'model'     : svr,", "'model'     : lgbm,")
            cell['source'] = [line + '\n' for line in source.split('\n') if line]
            
        if "results[name] = run_svr(" in source:
            source = source.replace("run_svr(", "run_lgbm(")
            cell['source'] = [line + '\n' for line in source.split('\n') if line]
            
        if "Perbandingan Metrik SVR per Dataset" in source:
            source = source.replace("SVR", "LightGBM")
            cell['source'] = [line + '\n' for line in source.split('\n') if line]

with open('predict_lightgbm.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_lgbm, f, indent=1)

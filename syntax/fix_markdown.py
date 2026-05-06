import json

def fix_md(filename, model_name):
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'markdown':
            source = "".join(cell['source'])
            source = source.replace("SVR + XAI (SHAP)", f"{model_name} (Without XAI)")
            source = source.replace("3. **XAI (Explainable AI)** menggunakan SHAP untuk mengetahui feature mana yang paling berpengaruh\n", "")
            if model_name == "LightGBM":
                source = source.replace("SVR", "LightGBM")
            cell['source'] = [line + '\n' for line in source.split('\n') if line]
            
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_md('predict.ipynb', 'SVR')
fix_md('predict_lightgbm.ipynb', 'LightGBM')

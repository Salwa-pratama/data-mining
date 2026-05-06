import json

def fix_colors(filename):
    with open(filename, 'r') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source_text = "".join(cell['source'])
            
            # fix plot colors SVR
            if "colors = [" in source_text and "'#4e79a7'" in source_text:
                new_src = source_text.replace(
                    "colors = [\n    '#4e79a7', '#4e79a7', '#4e79a7',  # ZScore - biru\n    '#f28e2b', '#f28e2b', '#f28e2b',  # Log - oranye\n    '#59a14f', '#59a14f', '#59a14f',  # MinMax - hijau\n    '#e15759', '#e15759', '#e15759',  # Decimal - merah\n]",
                    "colors = []\nfor name in datasets.keys():\n    if 'ZScore' in name: colors.append('#4e79a7')\n    elif 'Log' in name: colors.append('#f28e2b')\n    elif 'MinMax' in name: colors.append('#59a14f')\n    else: colors.append('#e15759')"
                )
                cell['source'] = [new_src]
                
    with open(filename, 'w') as f:
        json.dump(nb, f, indent=1)

fix_colors('predict1.ipynb')

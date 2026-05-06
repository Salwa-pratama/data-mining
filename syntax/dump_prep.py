import json

with open('preprocessing2.ipynb', 'r') as f:
    nb = json.load(f)

code_lines = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        code_lines.extend(cell['source'])
        code_lines.append("\n\n")

with open('preprocessing2.py', 'w') as f:
    f.writelines(code_lines)

import json

with open('predict1.ipynb', 'r') as f:
    nb = json.load(f)

lines = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        lines.append(f"# CELL {i}\n")
        lines.extend(cell['source'])
        lines.append("\n\n")

with open('predict1_source.txt', 'w') as f:
    f.writelines(lines)

import json
import sys

def dump_source(ipynb_file, out_txt):
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open(out_txt, 'w', encoding='utf-8') as f_out:
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                f_out.write(f"# CELL {i}\n")
                f_out.write("".join(cell.get('source', [])))
                f_out.write("\n\n")

if __name__ == "__main__":
    dump_source('predict.ipynb', 'predict_source.txt')
    dump_source('predict1.ipynb', 'predict1_source.txt')
    dump_source('preprocessing2.ipynb', 'preprocessing2_source.txt')

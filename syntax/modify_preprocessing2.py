import json

with open("preprocessing2.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "df['Total_Income'] =" in source:
            new_source = source + "\n# Add Loan Amount Income Ratio\ndf['Loan_Amount_Income_Ratio'] = (df['LoanAmount'] / df['Total_Income']) + 1\n"
            new_source += "df_clean_outliers['Loan_Amount_Income_Ratio'] = (df_clean_outliers['LoanAmount'] / df_clean_outliers['Total_Income']) + 1\n"
            new_source += "df_outliers_replace['Loan_Amount_Income_Ratio'] = (df_outliers_replace['LoanAmount'] / df_outliers_replace['Total_Income']) + 1\n"
            
            # Update columns arrangement if needed, but LoanAmount and Total_Income are there.
            cell["source"] = new_source.splitlines(keepends=True)
            break

with open("preprocessing2.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

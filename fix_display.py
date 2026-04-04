import json

with open('cartoon_analysis_complete_new.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i in range(len(cell['source'])):
            cell['source'][i] = cell['source'][i].replace('display(anova_table_cog)', 'print(anova_table_cog)')
            cell['source'][i] = cell['source'][i].replace('display(anova_table_beh)', 'print(anova_table_beh)')
            cell['source'][i] = cell['source'][i].replace('display(mixed_results.summary().tables[1])', 'print(mixed_results.summary().tables[1])')

with open('cartoon_analysis_complete_new.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

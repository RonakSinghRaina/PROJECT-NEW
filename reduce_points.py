import json

with open('RONAK SINGH RAINA IMS23319 - assign3-2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 3 has the rejection sampling code - reduce 100000 to 10000
src = ''.join(nb['cells'][3]['source'])
src = src.replace(
    "X_u, Y_u, passed_uniform = batch_sample_uniform(target, 100000)",
    "X_u, Y_u, passed_uniform = batch_sample_uniform(target, 10000)"
)
src = src.replace(
    "X_g, Y_g, passed_norm = batch_sample_gaussian(target, 100000)",
    "X_g, Y_g, passed_norm = batch_sample_gaussian(target, 10000)"
)
nb['cells'][3]['source'] = [src]

with open('RONAK SINGH RAINA IMS23319 - assign3-2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done! Reduced samples from 100,000 to 10,000.")

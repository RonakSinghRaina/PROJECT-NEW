import json

with open('RONAK SINGH RAINA IMS23319 - assign3-2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix Cell 10: legend() is called BEFORE the labeled axvline
src10 = ''.join(nb['cells'][10]['source'])

# The problem: ax_j.legend() comes before ax_j.axvline(..., label="MLE estimate")
# Fix: move legend after the axvline calls
old_order = '''ax_j.set_xlabel("T (K)")
ax_j.set_ylabel("A")
ax_j.legend(fontsize=8)
ax_j.axvline(T_mle, color="black", linewidth=1, linestyle="--", label="MLE estimate")
ax_j.axhline(A_mle, color="black", linewidth=1, linestyle="--")'''

new_order = '''ax_j.set_xlabel("T (K)")
ax_j.set_ylabel("A")
ax_j.axvline(T_mle, color="black", linewidth=1, linestyle="--", label="MLE estimate")
ax_j.axhline(A_mle, color="black", linewidth=1, linestyle="--")
ax_j.legend(fontsize=8)'''

src10 = src10.replace(old_order, new_order)
nb['cells'][10]['source'] = [src10]

with open('RONAK SINGH RAINA IMS23319 - assign3-2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed: legend order in Cell 10.")

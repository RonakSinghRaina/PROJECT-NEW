import json

with open('RONAK SINGH RAINA IMS23319 - assign3-2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# --- CELL 5 (index 5): Raw Solar Spectrum data plot (2 subplots, no legends) ---
src5 = ''.join(nb['cells'][5]['source'])
# Add label to the errorbar call and add legends to both axes
old5 = '''    ax.errorbar(
        wavelength,
        flux,
        yerr=flux_err,
        fmt=".",
        markersize=2,
        color="#00008B",
        ecolor="black",
        elinewidth=0.35,
        alpha=0.55,
    )'''
new5 = '''    ax.errorbar(
        wavelength,
        flux,
        yerr=flux_err,
        fmt=".",
        markersize=2,
        color="#00008B",
        ecolor="black",
        elinewidth=0.35,
        alpha=0.55,
        label='Solar Spectrum Data',
    )'''
src5 = src5.replace(old5, new5)

old5b = '''axes[0].set_xlim(0, 5)
axes[0].set_xlabel("wavelength (microns)")
axes[0].set_ylabel("E")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("wavelength (microns)")
axes[1].set_ylabel("E")'''
new5b = '''axes[0].set_xlim(0, 5)
axes[0].set_xlabel("wavelength (microns)")
axes[0].set_ylabel("E")
axes[0].set_title("Solar Spectrum (Linear Scale)")
axes[0].legend(fontsize=9)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("wavelength (microns)")
axes[1].set_ylabel("E")
axes[1].set_title("Solar Spectrum (Log-Log Scale)")
axes[1].legend(fontsize=9)'''
src5 = src5.replace(old5b, new5b)
nb['cells'][5]['source'] = [src5]

# --- CELL 7 (index 7): Residuals plot (2 subplots, no legends) ---
src7 = ''.join(nb['cells'][7]['source'])

old7a = 'axes[0].plot(lam_smooth, flux_smooth, color="black", linewidth=2)'
new7a = 'axes[0].plot(lam_smooth, flux_smooth, color="black", linewidth=2, label="MLE Blackbody Fit")'
src7 = src7.replace(old7a, new7a)

old7b = '''axes[0].errorbar(
    wavelength,
    flux,
    yerr=flux_err,
    fmt=".",
    markersize=2,
    color="#00008B",
    ecolor="black",
    elinewidth=0.35,
    alpha=0.55,
)'''
new7b = '''axes[0].errorbar(
    wavelength,
    flux,
    yerr=flux_err,
    fmt=".",
    markersize=2,
    color="#00008B",
    ecolor="black",
    elinewidth=0.35,
    alpha=0.55,
    label="Solar Data",
)'''
src7 = src7.replace(old7b, new7b)

old7c = '''axes[0].set_ylabel("E")
axes[0].set_xlim(0, 5)'''
new7c = '''axes[0].set_ylabel("E")
axes[0].set_xlim(0, 5)
axes[0].set_title("(c) Blackbody Fit with Residuals")
axes[0].legend(fontsize=9)'''
src7 = src7.replace(old7c, new7c)

old7d = 'axes[1].scatter(wx, ny, s=4, alpha=0.65, color="#00008B", edgecolors="black", linewidths=0.2)'
new7d = 'axes[1].scatter(wx, ny, s=4, alpha=0.65, color="#00008B", edgecolors="black", linewidths=0.2, label="Normalized Residuals")'
src7 = src7.replace(old7d, new7d)

old7e = '''axes[1].set_xlabel("wavelength (microns)")
axes[1].set_ylabel("(data - model) / error")
axes[1].set_xlim(0, 5)'''
new7e = '''axes[1].set_xlabel("wavelength (microns)")
axes[1].set_ylabel("(data - model) / error")
axes[1].set_xlim(0, 5)
axes[1].legend(fontsize=8)'''
src7 = src7.replace(old7e, new7e)

nb['cells'][7]['source'] = [src7]

# --- CELL 9 (index 9): BB vs BB+PL comparison plot, top panel missing legends ---
src9 = ''.join(nb['cells'][9]['source'])

old9a = 'axes[0].plot(lam_smooth2, flux_bb_line, color="black", linewidth=2)'
new9a = 'axes[0].plot(lam_smooth2, flux_bb_line, color="black", linewidth=2, label="Blackbody")'
src9 = src9.replace(old9a, new9a)

old9b = 'axes[0].plot(lam_smooth2, flux_bbpl_line, color="#00008B", linewidth=2, linestyle="--")'
new9b = 'axes[0].plot(lam_smooth2, flux_bbpl_line, color="#00008B", linewidth=2, linestyle="--", label="Blackbody + Power Law")'
src9 = src9.replace(old9b, new9b)

old9c = 'axes[0].set_ylabel("E")'
new9c = '''axes[0].set_ylabel("E")
axes[0].set_title("(e) Model Comparison: BB vs BB+PL")
axes[0].legend(fontsize=9)'''
src9 = src9.replace(old9c, new9c)

# Also add label to the errorbar in the top panel
old9d = '''axes[0].errorbar(
    wavelength[mask5],
    flux[mask5],
    yerr=flux_err[mask5],
    fmt=".",
    markersize=2,
    color="#00008B",
    ecolor="black",
    elinewidth=0.35,
    alpha=0.55,
)'''
new9d = '''axes[0].errorbar(
    wavelength[mask5],
    flux[mask5],
    yerr=flux_err[mask5],
    fmt=".",
    markersize=2,
    color="#00008B",
    ecolor="black",
    elinewidth=0.35,
    alpha=0.55,
    label="Solar Data",
)'''
src9 = src9.replace(old9d, new9d)

nb['cells'][9]['source'] = [src9]

# --- CELL 10 (index 10): Corner/posterior plot - add legends to marginals ---
src10 = ''.join(nb['cells'][10]['source'])

old10a = 'ax_t.plot(p_T, T_grid, color="#00008B", linewidth=1.5)'
new10a = 'ax_t.plot(p_T, T_grid, color="#00008B", linewidth=1.5, label="P(T|data)")'
src10 = src10.replace(old10a, new10a)

old10b = '''ax_t.set_xlabel("P(T|data)")
ax_t.set_ylabel("T (K)")'''
new10b = '''ax_t.set_xlabel("P(T|data)")
ax_t.set_ylabel("T (K)")
ax_t.legend(fontsize=8)'''
src10 = src10.replace(old10b, new10b)

old10c = 'ax_a.plot(A_grid, p_A, color="#00008B", linewidth=1.5)'
new10c = 'ax_a.plot(A_grid, p_A, color="#00008B", linewidth=1.5, label="P(A|data)")'
src10 = src10.replace(old10c, new10c)

old10d = '''ax_a.set_ylabel("P(A|data)")
ax_a.set_xlabel("A")'''
new10d = '''ax_a.set_ylabel("P(A|data)")
ax_a.set_xlabel("A")
ax_a.legend(fontsize=8)'''
src10 = src10.replace(old10d, new10d)

# Add legends to joint plot dashed lines
old10e = 'ax_j.axvline(T_mle, color="black", linewidth=1, linestyle="--")'
new10e = 'ax_j.axvline(T_mle, color="black", linewidth=1, linestyle="--", label="MLE estimate")'
src10 = src10.replace(old10e, new10e)

old10f = '''ax_j.set_xlabel("T (K)")
ax_j.set_ylabel("A")'''
new10f = '''ax_j.set_xlabel("T (K)")
ax_j.set_ylabel("A")
ax_j.legend(fontsize=8)'''
src10 = src10.replace(old10f, new10f)

nb['cells'][10]['source'] = [src10]

with open('RONAK SINGH RAINA IMS23319 - assign3-2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done! Legends added to all graphs.")

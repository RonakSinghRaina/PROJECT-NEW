import json

with open('RONAK SINGH RAINA IMS23319 - assign3-2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix Cell 16 (part g): Remove duplicated MCMC code
# The code block was duplicated - the entire MCMC block appears twice
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if '(g) Bayesian Inference for blackbody temperature using MCMC' in src:
            # Replace with clean, single copy
            cell['source'] = [
                "# =============================================================================\n",
                "# (g) Bayesian Inference for blackbody temperature using MCMC (emcee)\n",
                "# =============================================================================\n",
                "import emcee\n",
                "\n",
                "# Define the log-prior: flat (uniform) prior over physically reasonable ranges\n",
                "def log_prior(theta):\n",
                "    T, A = theta\n",
                "    if 1000 < T < 20000 and 0.001 < A < 100.0:\n",
                "        return 0.0  # flat prior = log(1) = 0\n",
                "    return -np.inf  # outside bounds = impossible\n",
                "\n",
                "# Define the log-likelihood\n",
                "def log_likelihood(theta):\n",
                "    T, A = theta\n",
                "    total = 0.0\n",
                "    for i in range(n_data):\n",
                "        model_val = blackbody(wavelength[i], T, A)\n",
                "        if model_val <= 0 or np.isnan(model_val):\n",
                "            return -np.inf\n",
                "        resid = (flux[i] - model_val) / flux_err[i]\n",
                "        total += resid ** 2 + np.log(2 * np.pi * flux_err[i] ** 2)\n",
                "    return -0.5 * total\n",
                "\n",
                "# Define the log-posterior = log-prior + log-likelihood\n",
                "def log_posterior(theta):\n",
                "    lp = log_prior(theta)\n",
                "    if not np.isfinite(lp):\n",
                "        return -np.inf\n",
                "    ll = log_likelihood(theta)\n",
                "    if not np.isfinite(ll):\n",
                "        return -np.inf\n",
                "    return lp + ll\n",
                "\n",
                "# Set up the MCMC sampler\n",
                "n_walkers = 32\n",
                "n_dim = 2  # T and A\n",
                "n_steps = 5000\n",
                "n_burn = 1000  # burn-in period to discard\n",
                "\n",
                "# Initialize walkers as a small ball around the MLE solution\n",
                "np.random.seed(42)\n",
                "starting_positions = np.zeros((n_walkers, n_dim))\n",
                "for w in range(n_walkers):\n",
                "    starting_positions[w, 0] = T_mle + np.random.normal(0, 50)   # T\n",
                "    starting_positions[w, 1] = A_mle + np.random.normal(0, A_mle * 0.1)  # A\n",
                "\n",
                "# Run the sampler\n",
                "print('Running MCMC with', n_walkers, 'walkers for', n_steps, 'steps...')\n",
                "sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior)\n",
                "sampler.run_mcmc(starting_positions, n_steps, progress=True)\n",
                "print('MCMC complete!')\n",
                "\n",
                "# Extract the chain after burn-in\n",
                "samples = sampler.get_chain(discard=n_burn, flat=True)\n",
                "print('Number of samples after burn-in:', len(samples))\n",
                "\n",
                "# Compute Bayesian estimates (median and 68% credible interval)\n",
                "T_samples = samples[:, 0]\n",
                "A_samples = samples[:, 1]\n",
                "\n",
                "T_median = np.median(T_samples)\n",
                "T_lo = np.percentile(T_samples, 16)\n",
                "T_hi = np.percentile(T_samples, 84)\n",
                "\n",
                "A_median = np.median(A_samples)\n",
                "A_lo = np.percentile(A_samples, 16)\n",
                "A_hi = np.percentile(A_samples, 84)\n",
                "\n",
                "print()\n",
                "print('=== (g) Bayesian Estimates (MCMC) ===')\n",
                "print('T =', round(T_median, 1), '+', round(T_hi - T_median, 1), '/', round(T_median - T_lo, 1), 'K')\n",
                "print('A =', round(A_median, 6), '+', round(A_hi - A_median, 6), '/', round(A_median - A_lo, 6))\n"
            ]
            break

# Fix Cell for part (h): Install corner inline as fallback
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if '(h) Corner plot for posterior distribution' in src:
            cell['source'] = [
                "# =============================================================================\n",
                "# (h) Corner plot for posterior distribution\n",
                "# =============================================================================\n",
                "try:\n",
                "    import corner\n",
                "except ModuleNotFoundError:\n",
                "    import subprocess, sys\n",
                "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'corner', '-q'])\n",
                "    import corner\n",
                "\n",
                "fig = corner.corner(\n",
                "    samples,\n",
                "    labels=['T (K)', 'A'],\n",
                "    truths=[T_mle, A_mle],  # MLE values shown as red lines\n",
                "    quantiles=[0.16, 0.5, 0.84],\n",
                "    show_titles=True,\n",
                "    title_kwargs={'fontsize': 12},\n",
                "    color='#00008B',\n",
                "    truth_color='red',\n",
                ")\n",
                "fig.suptitle('(h) Corner Plot: Posterior Distribution (T, A)', fontsize=14, fontweight='bold', y=1.02)\n",
                "plt.show()\n",
                "\n",
                "print('The corner plot shows:')\n",
                "print('- Diagonal panels: 1D marginalized posteriors for T and A')\n",
                "print('- Off-diagonal panel: 2D joint posterior showing correlation between T and A')\n",
                "print('- Red lines: MLE estimates for comparison')\n",
                "print('- Dashed lines: 16th, 50th (median), and 84th percentiles')\n"
            ]
            break

with open('RONAK SINGH RAINA IMS23319 - assign3-2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("All fixes applied successfully.")

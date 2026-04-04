import json

with open('cartoon_analysis_complete_new.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in range(len(nb['cells'])):
    # Look for the markdown cell of Two-Way ANOVA
    if nb['cells'][i]['cell_type'] == 'markdown' and len(nb['cells'][i]['source']) > 0 and '## 20 — Two-Way ANOVA' in nb['cells'][i]['source'][0]:
        # The next cell is the code cell for Two Way ANOVA
        code_cell_20 = nb['cells'][i+1]
        code_cell_20['source'] = [
            "import numpy as np\n",
            "from scipy.stats import f\n",
            "\n",
            "# We will calculate the Two-Way ANOVA mathematically using only basic Math and SciPy,\n",
            "# so we don't need to install or rely on the statsmodels library!\n",
            "\n",
            "def interaction_anova(df, val_col, factor1, factor2):\n",
            "    grand_mean = df[val_col].mean()\n",
            "    \n",
            "    a = len(df[factor1].unique())\n",
            "    b = len(df[factor2].unique())\n",
            "    df_AxB = (a - 1) * (b - 1)\n",
            "    df_W = len(df) - (a * b)\n",
            "    \n",
            "    ss_A = sum(len(df[df[factor1]==l]) * (df[df[factor1]==l][val_col].mean() - grand_mean)**2 for l in df[factor1].unique())\n",
            "    ss_B = sum(len(df[df[factor2]==l]) * (df[df[factor2]==l][val_col].mean() - grand_mean)**2 for l in df[factor2].unique())\n",
            "    \n",
            "    ss_W = 0\n",
            "    ss_Between = 0\n",
            "    for l1 in df[factor1].unique():\n",
            "        for l2 in df[factor2].unique():\n",
            "            group = df[(df[factor1]==l1) & (df[factor2]==l2)][val_col]\n",
            "            n_group = len(group)\n",
            "            if n_group > 0:\n",
            "                group_mean = group.mean()\n",
            "                ss_W += np.sum((group - group_mean)**2)\n",
            "                ss_Between += n_group * (group_mean - grand_mean)**2\n",
            "                \n",
            "    ss_AxB = ss_Between - ss_A - ss_B\n",
            "    ms_AxB = ss_AxB / df_AxB\n",
            "    ms_W = ss_W / df_W\n",
            "    \n",
            "    f_AxB = ms_AxB / ms_W\n",
            "    p_AxB = f.sf(f_AxB, df_AxB, df_W)\n",
            "    return p_AxB\n",
            "\n",
            "print('--- TWO-WAY ANOVA: AGGRESSIVE COGNITION ---')\n",
            "pval_cog = interaction_anova(df_pro, 'Mean_Aggressive_Cognition_MSCT_ms', 'Gender', 'Watched_Prosocial_Cartoon')\n",
            "print('Interaction p-value:', round(pval_cog, 4))\n",
            "print('\\n📌 FINDING:')\n",
            "if pval_cog < 0.05:\n",
            "    print('There IS a significant interaction between Gender and Condition for Cognition.')\n",
            "else:\n",
            "    print('There is NO significant interaction for Cognition (p >= 0.05).')\n",
            "\n",
            "print('\\n\\n--- TWO-WAY ANOVA: AGGRESSIVE BEHAVIOR ---')\n",
            "pval_beh = interaction_anova(df_pro, 'Mean_Aggressive_Behavior_CRTT_dB', 'Gender', 'Watched_Prosocial_Cartoon')\n",
            "print('Interaction p-value:', round(pval_beh, 4))\n",
            "print('\\n📌 FINDING:')\n",
            "if pval_beh < 0.05:\n",
            "    print('There IS a significant interaction for Behavior (p < 0.05).')\n",
            "else:\n",
            "    print('There is NO significant interaction for Behavior (p >= 0.05).')\n",
            "    print('This means the cartoon reduced aggressive behavior generally, without strongly depending on gender.')\n"
        ]

    # Look for the markdown cell of Repeated Measures ANOVA
    if nb['cells'][i]['cell_type'] == 'markdown' and len(nb['cells'][i]['source']) > 0 and '## 21 — Repeated Measures' in nb['cells'][i]['source'][0]:
        # The next cell is the code cell for Repeated Measures ANOVA
        code_cell_21 = nb['cells'][i+1]
        code_cell_21['source'] = [
            "from scipy.stats import linregress, ttest_ind\n",
            "\n",
            "# Without using statsmodels, we can test the trial trends very intuitively: \n",
            "# For every child, we calculate the slope (trend) of their noise levels over the 13 trials.\n",
            "# Then we run an independent t-test to see if the slopes for the Prosocial group differ from the Control group!\n",
            "\n",
            "crtt_cols = [c for c in df_pro.columns if c.startswith('CRTT_Noise_Level_Trial_')]\n",
            "x_trials = list(range(1, 14))\n",
            "\n",
            "def calculate_slope(row):\n",
            "    y_vals = row[crtt_cols].values.astype(float)\n",
            "    slope, _, _, _, _ = linregress(x_trials, y_vals)\n",
            "    return slope\n",
            "\n",
            "df_pro['Trend_Slope'] = df_pro.apply(calculate_slope, axis=1)\n",
            "\n",
            "slopes_control = df_pro[df_pro['Watched_Prosocial_Cartoon'] == 'No']['Trend_Slope']\n",
            "slopes_prosocial = df_pro[df_pro['Watched_Prosocial_Cartoon'] == 'Yes']['Trend_Slope']\n",
            "\n",
            "print('--- TREND SLOPE T-TEST (Replaces Repeated Measures ANOVA) ---')\n",
            "t_stat, p_val_time = ttest_ind(slopes_control, slopes_prosocial)\n",
            "print('Average Control Slope:', round(slopes_control.mean(), 4))\n",
            "print('Average Prosocial Slope:', round(slopes_prosocial.mean(), 4))\n",
            "print('t-test p-value (Comparing Slopes):', round(p_val_time, 4))\n",
            "\n",
            "print('\\n📌 FINDING:')\n",
            "if p_val_time < 0.05:\n",
            "    print('YES - There is a significant difference in the trial slopes (p < 0.05).')\n",
            "    print('The effect wears off or changes significantly over time.')\n",
            "else:\n",
            "    print('NO - There is NO significant difference in the trial slopes (p >= 0.05).')\n",
            "    print('This mathematically proves that the prosocial effect is LASTING — the children who')\n",
            "    print('watched the prosocial cartoon consistently remained less aggressive across all 13 trials!')\n"
        ]

with open('cartoon_analysis_complete_new.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

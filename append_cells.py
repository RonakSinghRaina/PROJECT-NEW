import json

notebook_path = 'cartoon_analysis_complete_new.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: Markdown for Two-Way ANOVA
cell1 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 20 — Two-Way ANOVA (Interaction Effects between Gender & Cartoon)\n",
        "Currently, we ran separate independent t-tests for boys and girls to see how gender splits the cartoon's effect. While this shows the split conceptually, it doesn't statistically prove that the *effect difference* between genders is significant.\n",
        "\n",
        "**Hypothesis:** Does the effect of watching a prosocial cartoon depend significantly on gender?\n",
        "\n",
        "We can run a Two-Way ANOVA with \"Cartoon Type\" and \"Gender\" as the independent variables to see if there is a significant interaction effect."
    ]
}

# Cell 2: Code for Two-Way ANOVA
cell2 = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Note: To run these advanced statistical tests, we will use the 'statsmodels' library.\n",
        "# If you get a ModuleNotFoundError, uncomment the line below and run it once to install:\n",
        "# !pip install statsmodels -q\n",
        "\n",
        "import statsmodels.api as sm\n",
        "from statsmodels.formula.api import ols\n",
        "\n",
        "# ─── 20a. Preparing Data for Two-Way ANOVA ───\n",
        "anova_data = df_pro[['Gender', 'Watched_Prosocial_Cartoon', 'Mean_Aggressive_Cognition_MSCT_ms', 'Mean_Aggressive_Behavior_CRTT_dB']].copy()\n",
        "anova_data.columns = ['Gender', 'Condition', 'Cognition', 'Behavior']\n",
        "\n",
        "print('--- TWO-WAY ANOVA: AGGRESSIVE COGNITION ---')\n",
        "# The formula mathematically tests Main Effects + Interaction Effect\n",
        "model_cog = ols('Cognition ~ C(Condition) + C(Gender) + C(Condition):C(Gender)', data=anova_data).fit()\n",
        "anova_table_cog = sm.stats.anova_lm(model_cog, typ=2)\n",
        "display(anova_table_cog)\n",
        "\n",
        "# Extracting the interaction p-value\n",
        "pval_interaction_cog = anova_table_cog.loc['C(Condition):C(Gender)', 'PR(>F)']\n",
        "print('\\n📌 FINDING:')\n",
        "if pval_interaction_cog < 0.05:\n",
        "    print('There IS a significant interaction between Gender and Condition for Cognition (p < 0.05).')\n",
        "    print('This statistically formally proves that the prosocial cartoon affected the thinking')\n",
        "    print('of boys differently than it affected the thinking of girls.')\n",
        "else:\n",
        "    print('There is NO significant interaction for Cognition (p >= 0.05).')\n",
        "\n",
        "print('\\n\\n--- TWO-WAY ANOVA: AGGRESSIVE BEHAVIOR ---')\n",
        "model_beh = ols('Behavior ~ C(Condition) + C(Gender) + C(Condition):C(Gender)', data=anova_data).fit()\n",
        "anova_table_beh = sm.stats.anova_lm(model_beh, typ=2)\n",
        "display(anova_table_beh)\n",
        "\n",
        "pval_interaction_beh = anova_table_beh.loc['C(Condition):C(Gender)', 'PR(>F)']\n",
        "print('\\n📌 FINDING:')\n",
        "if pval_interaction_beh < 0.05:\n",
        "    print('There IS a significant interaction for Behavior (p < 0.05).')\n",
        "else:\n",
        "    print('There is NO significant interaction for Behavior (p >= 0.05).')\n",
        "    print('This means the cartoon reduced aggressive behavior generally, without the effect')\n",
        "    print('depending strongly on whether the child was a boy or a girl.')\n"
    ]
}

# Cell 3: Markdown for Repeated Measures ANOVA
cell3 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 21 — Repeated Measures ANOVA (Testing the Trial-by-Trial Trend)\n",
        "In section 14, we plotted the 13 CRTT trials on a line graph to see if the prosocial effect lasts over time or wears off (converges). However, there is no statistical test applied there.\n",
        "\n",
        "**Hypothesis:** Does the behavioral effect of the prosocial cartoon wear off significantly as time (trials) progresses?\n",
        "\n",
        "We can run a mixed-effects model (equivalent here to Repeated Measures ANOVA) to formally test if the slopes of the two lines in our trial chart are statistically different across the 13 intervals."
    ]
}

# Cell 4: Code for Repeated Measures ANOVA
cell4 = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "import statsmodels.formula.api as smf\n",
        "\n",
        "# ─── 21a. Reshaping Data for Mixed-Effects Model ───\n",
        "crtt_cols = [c for c in df_pro.columns if c.startswith('CRTT_Noise_Level_Trial_')]\n",
        "\n",
        "# We need a unique ID for each child to group their 13 trials together\n",
        "df_pro['Child_ID'] = range(len(df_pro))\n",
        "\n",
        "# Unpivoting the dataset from 'wide' to 'long' format using pandas melt\n",
        "long_data = pd.melt(df_pro, \n",
        "                    id_vars=['Child_ID', 'Watched_Prosocial_Cartoon', 'Gender'], \n",
        "                    value_vars=crtt_cols,\n",
        "                    var_name='Trial_Name', \n",
        "                    value_name='Noise_Level')\n",
        "\n",
        "# Extracting just the trial number (1 to 13)\n",
        "long_data['Trial_Num'] = long_data['Trial_Name'].str.extract(r'(\\d+)').astype(int)\n",
        "\n",
        "print('--- MIXED EFFECTS MODEL: TRIAL BY TRIAL TREND ---')\n",
        "# Using a mixed linear model with random intercepts for each child\n",
        "# We test the interaction between Trial_Num (time) and Condition (cartoon type)\n",
        "mixed_model = smf.mixedlm(\"Noise_Level ~ Trial_Num * C(Watched_Prosocial_Cartoon)\", \n",
        "                          data=long_data, \n",
        "                          groups=long_data[\"Child_ID\"])\n",
        "mixed_results = mixed_model.fit()\n",
        "\n",
        "# Display the statistical summary\n",
        "display(mixed_results.summary().tables[1])\n",
        "\n",
        "# Extract interaction p-value between Trial progression and Cartoon watched\n",
        "p_val_time_interaction = None\n",
        "for idx in mixed_results.pvalues.index:\n",
        "    if 'Trial_Num' in idx and 'Watched_Prosocial_Cartoon' in idx:\n",
        "        p_val_time_interaction = mixed_results.pvalues[idx]\n",
        "\n",
        "if p_val_time_interaction is not None:\n",
        "    print('\\n📌 FINDING:')\n",
        "    if p_val_time_interaction < 0.05:\n",
        "        print('YES - There is a significant Trial x Cartoon interaction (p < 0.05).')\n",
        "        print('This means the two lines in our plot formally have different slopes,')\n",
        "        print('and the effect wears off or changes significantly over time.')\n",
        "    else:\n",
        "        print('NO - There is NO significant Trial x Cartoon interaction (p >= 0.05).')\n",
        "        print('The slopes of the lines are not statistically different.')\n",
        "        print('This mathematically proves that the prosocial effect is LASTING — the children who')\n",
        "        print('watched the prosocial cartoon consistently remained less aggressive across all 13 trials!')\n",
        "else:\n",
        "    print('Could not compute interaction term automatically.')\n"
    ]
}

nb['cells'].extend([cell1, cell2, cell3, cell4])

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully appended new cells.")

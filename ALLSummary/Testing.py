import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from EXP1Summary import (
    checkdeletedrows_forallcsv, 
    calculate_metrics
)

def testEXP1():
    """Processes EXP1 results, performs statistical tests, and prints ANOVA and post-hoc analysis results."""

    # 🔹 **Step 1: Load and Process Data**
    combined_deleted_df, all_balanced_metrics, final_balanced_df = checkdeletedrows_forallcsv()

    # 🔹 **Step 2: Split Data into Individual DataFrames by Task**
    task_names = [
        "df_volume", "df_area", "df_direction", "df_length", "df_position_common_scale",
        "df_position_non_aligned_scale", "df_angle", "df_curvature", "df_shading"
    ]
    task_dfs = {task: final_balanced_df[final_balanced_df["task"] == task] for task in task_names}

    # 🔹 **Step 3: Compute Metrics and Extract MLAE Values**
    metrics_table, mlae_df = calculate_metrics(*task_dfs.values())

    # Group MLAE values by model
    mlae_groups = [group["MLAE"].values for _, group in mlae_df.groupby("Model")]

    # 🔹 **Step 4: Perform O'Brien's Test for Homogeneity of Variances**
    obrien_transformed = stats.obrientransform(*mlae_groups)
    obrien_f, obrien_p = stats.f_oneway(*obrien_transformed)

    print("\n🔬 O'Brien Test for Homogeneity of Variances:")
    print(f"F-statistic: {obrien_f:.4f}")
    print(f"P-value: {obrien_p:.4e}")

    if obrien_p > 0.05:
        print("✅ Variances are equal (homoscedasticity holds). Proceeding with one-way ANOVA.")
        use_welch = False
    else:
        print("⚠️ Variances are not equal (heteroscedasticity detected). Using Welch’s ANOVA.")
        use_welch = True

    # 🔹 **Step 5: Perform One-Way ANOVA or Welch’s ANOVA**
    if not use_welch:
        f_stat, p_value = stats.f_oneway(*mlae_groups)
        print("\n📊 One-Way ANOVA Results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4e}")
    else:
        welch_anova_results = pg.welch_anova(dv='MLAE', between='Model', data=mlae_df)
        print("\n📊 Welch's ANOVA Results:")
        print(welch_anova_results)

    # 🔹 **Step 6: Perform Post-hoc Tukey’s HSD Test**
    print("\n🔬 Performing Tukey's HSD Post-hoc Test...")
    tukey_results = pairwise_tukeyhsd(endog=mlae_df["MLAE"], groups=mlae_df["Model"], alpha=0.05)
    print("\n📊 Tukey's HSD Post-hoc Test Results:")
    print(tukey_results)

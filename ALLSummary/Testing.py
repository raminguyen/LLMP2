import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import mean_absolute_error

from EXP1Summary import (
    checkdeletedrows_forallcsv as test_exp1, 
    calculate_metrics
) 

from EXP2Summary import (
    checkdeletedrows_forallcsv as test_exp2, 
    calculate_metrics
)


""" EXP1 """


"""Use to download CSV

def testEXP1():

    #for i, item in enumerate(final_balanced_df):
    #print(f"Element {i}: Type = {type(item)}")

    #final_balanced_df = test_exp1() 

"""


def calculate_mlae_individual_EXP1():

    df = pd.read_csv("finalEXP1.csv") 

    print(len(df))

    # Ensure numeric conversion of ground truth and cleaned answers
    df['ground_truth_num'] = pd.to_numeric(df['ground_truth'], errors='coerce')
    df['cleaned_answers_num'] = pd.to_numeric(df['cleaned_answers'], errors='coerce')

    # Drop rows with missing values in necessary columns
    df = df.dropna(subset=['ground_truth_num', 'cleaned_answers_num'])

    # Compute MLAE for each row
    def compute_mlae(row):
        return np.log2(mean_absolute_error([row['ground_truth_num']], [row['cleaned_answers_num']]) + 0.125)

    df['mlae'] = df.apply(compute_mlae, axis=1)

    return df


"""==================================="""
    

""" EXP2 """

def testEXP2():
    """Processes EXP2 results, performs statistical tests, and prints ANOVA and post-hoc analysis results."""

    # 🔹 **Step 1: Load and Process Data**
    print("\n📌 Processing Data...")
    test_exp2()


def calculate_mlae_individual_EXP2():
    """Calculate individual MLAE values for each row in a dataframe."""
    
    df = pd.read_csv("finalEXP2.csv") 
        
    df['ground_truth_num'] = df['ground_truth'].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)
    df['cleaned_answers_num'] = df['cleaned_answers'].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)
    
    df = df.dropna(subset=['ground_truth_num', 'cleaned_answers_num'])
    
    df['mlae'] = df.apply(
        lambda row: np.log2(mean_absolute_error(
            [row['ground_truth_num']], 
            [row['cleaned_answers_num']]
        ) + 0.125),
        axis=1
    )
    
    return df


""" EXP3 """


def calculate_mlae_individual_EXP3():

    df = pd.read_csv("finalEXP3.csv") 

    print(len(df))

    # Ensure numeric conversion of ground truth and cleaned answers
    df['ground_truth_num'] = pd.to_numeric(df['ground_truth'], errors='coerce')
    df['cleaned_answers_num'] = pd.to_numeric(df['cleaned_answers'], errors='coerce')

    # Drop rows with missing values in necessary columns
    df = df.dropna(subset=['ground_truth_num', 'cleaned_answers_num'])

    # Compute MLAE for each row
    def compute_mlae(row):
        return np.log2(mean_absolute_error([row['ground_truth_num']], [row['cleaned_answers_num']]) + 0.125)

    df['mlae'] = df.apply(compute_mlae, axis=1)

    return df

""" EXP4 """

def calculate_mlae_individual_EXP4():
    """Calculate individual MLAE values for each row in a dataframe."""
    
    df = pd.read_csv("finalEXP4.csv") 
    
    df['ground_truth_num'] = df['ground_truth'].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)
    df['cleaned_answers_num'] = df['cleaned_answers'].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)
    
    df = df.dropna(subset=['ground_truth_num', 'cleaned_answers_num'])
    
    df['mlae'] = df.apply(
        lambda row: np.log2(mean_absolute_error(
            [row['ground_truth_num']], 
            [row['cleaned_answers_num']]
        ) + 0.125),
        axis=1
    )
    
    return df

""" Perform EXP5 """

def calculate_mlae_individual_EXP5():

    df = pd.read_csv("finalEXP5.csv") 

    print(len(df))

    # Ensure numeric conversion of ground truth and cleaned answers
    df['ground_truth_num'] = pd.to_numeric(df['ground_truth'], errors='coerce')
    df['cleaned_answers_num'] = pd.to_numeric(df['cleaned_answers'], errors='coerce')

    # Drop rows with missing values in necessary columns
    df = df.dropna(subset=['ground_truth_num', 'cleaned_answers_num'])

    # Compute MLAE for each row
    def compute_mlae(row):
        return np.log2(mean_absolute_error([row['ground_truth_num']], [row['cleaned_answers_num']]) + 0.125)

    df['mlae'] = df.apply(compute_mlae, axis=1)

    return df

""" PERFORM STATISTICAL TESTING"""

def perform_statistical_testing(balanced_df):
    """
    Performs statistical tests on MLAE values:
    1. O'Brien's test for homogeneity of variances
    2. One-way ANOVA or Welch’s ANOVA
    3. Tukey's HSD post-hoc test
    
    Args:
        balanced_df (pd.DataFrame): The processed DataFrame containing model results.

    Returns:
        dict: A dictionary with statistical results.
    """

    # 🔹 **Step 1: Extract MLAE Values by Model**
    mlae_groups = [group["mlae"].values for _, group in balanced_df.groupby("model_name")]

    # 🔹 **Step 2: Perform O'Brien's Test for Homogeneity of Variances**
    print("\n🔬 O'Brien Test for Homogeneity of Variances:")
    obrien_transformed = stats.obrientransform(*mlae_groups)
    obrien_f, obrien_p = stats.f_oneway(*obrien_transformed)

    print(f"F-statistic: {obrien_f:.4f}")
    print(f"P-value: {obrien_p:.4e}")

    use_welch = obrien_p <= 0.01  # Use Welch's ANOVA if variances are not equal

    if use_welch:
        print("⚠️ Variances are not equal (heteroscedasticity detected). Using Welch’s ANOVA.")
    else:
        print("✅ Variances are equal (homoscedasticity holds). Proceeding with one-way ANOVA.")

    # 🔹 **Step 3: Perform One-Way ANOVA or Welch’s ANOVA**
    if use_welch:
        print("\n📊 Welch's ANOVA Results:")
        anova_results = pg.welch_anova(dv='mlae', between='model_name', data=balanced_df)
        print(anova_results)

        # Extract p-value from Welch’s ANOVA results
        p_value = anova_results['p-unc'][0]  # Get the first (only) p-value

        # Print p-value if it's less than 0.01
        if p_value < 0.01:
            print(f"⚠️ Significant result: p-value = {p_value:.4e} (p < 0.01)")

    else:
        f_stat, p_value = stats.f_oneway(*mlae_groups)
        print("\n📊 One-Way ANOVA Results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4e}")

        anova_results = {"F-statistic": f_stat, "P-value": p_value}

    # 🔹 **Step 4: Perform Post-hoc Tukey’s HSD Test**
    print("\n🔬 Performing Tukey's HSD Post-hoc Test...")
    tukey_results = pairwise_tukeyhsd(endog=balanced_df["mlae"], groups=balanced_df["model_name"], alpha=0.01)
    print("\n📊 Tukey's HSD Post-hoc Test Results:")
    print(tukey_results)

   



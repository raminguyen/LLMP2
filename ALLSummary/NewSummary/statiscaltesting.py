""" PERFORM AVERAGE MLAE, INDIVIDUAL MLAE, AND STATISTICAL TESTING"""

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics import mean_absolute_error
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
import matplotlib.ticker as mticker


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def calculate_average_mlae_EXP1(df):
    """
    Calculate metrics for each dataset and model.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing 'task_name', 'model_name', 'ground_truth', and 'cleaned_answers'.
    
    Returns:
    pandas.DataFrame: Table of metrics for all models and datasets.
    """

    # Dictionary to store all metrics
    metrics_summary = []
    
    for df_name, df_task in df.groupby('task_name'):  # Group by task (dataset)
        for model_name, data in df_task.groupby('model_name'):  # Group by model
            
            # Convert to numeric 
            data['ground_truth_num'] = pd.to_numeric(data['ground_truth'], errors='coerce')
            data['cleaned_answers_num'] = pd.to_numeric(data['cleaned_answers'], errors='coerce')
            
            # Compute absolute error
            data['error'] = (data['ground_truth_num'] - data['cleaned_answers_num']).abs()

            # Compute MLAE
            data['mlae'] = np.log2(data['error'] + 0.125)

            mlae_values = data['mlae'].tolist()
            mae_values = data['error'].tolist()

            # Compute aggregated metrics
            avg_mlae = np.mean(mlae_values)
            std_mlae = np.std(mlae_values)
            avg_mae = np.mean(mae_values)

            # Compute confidence interval using bootstrapping
            try:
                bootstrap_result = bs.bootstrap(np.array(mlae_values), stat_func=bs_stats.std)
                confidence_value = 1.96 * bootstrap_result.value
            except:
                confidence_value = np.nan

            # Store metrics as a dictionary
            metrics_summary.append({
                'Dataset': df_name,
                'Model': model_name,
                'Average MLAE': round(avg_mlae, 2),
                'Average MAE': round(avg_mae, 2),
                'Std MLAE': round(std_mlae, 2),
                'Confidence Interval (95%)': round(confidence_value, 2)
            })
    
    # Convert list of dicts to DataFrame
    metrics_summary_df = pd.DataFrame(metrics_summary)
    
    return metrics_summary_df

def calculate_mlae_individual_EXP1():
    """Calculate individual MLAE values for each row in finalEXP1.csv."""

    df = pd.read_csv("/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/EXP1/finalEXP1.csv")

    # Convert ground truth and cleaned answers to numeric, handling lists/strings safely
    df['ground_truth_num'] = df['ground_truth'].apply(lambda x: pd.eval(x) if isinstance(x, str) and x.strip().startswith('[') else pd.to_numeric(x, errors='coerce'))
    df['cleaned_answers_num'] = df['cleaned_answers'].apply(lambda x: pd.eval(x) if isinstance(x, str) and x.strip().startswith('[') else pd.to_numeric(x, errors='coerce'))

    # Drop rows with missing values after conversion
    df = df.dropna(subset=['ground_truth_num', 'cleaned_answers_num'])

    # Compute absolute error
    df['error'] = (df['ground_truth_num'] - df['cleaned_answers_num']).abs()

    # Compute MLAE, keeping NaN values
    df['mlae'] = np.log2(df['error'] + 0.125)

    return df



def calculate_average_mlae_EXP2(df):
    """
    Calculate metrics for each dataset and model.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing 'task_name', 'model_name', 'ground_truth', and 'cleaned_answers'.
    
    Returns:
    pandas.DataFrame: Table of metrics for all models and datasets.
    """

    # Dictionary to store all metrics
    metrics_summary = []
    
    for df_name, df_task in df.groupby('task_name'):  # Group by task (dataset)
        for model_name, data in df_task.groupby('model_name'):  # Group by model
            
            data = data.copy()

            # Safe function to evaluate values and handle tuples (ranges)
            def safe_eval(x):
                try:
                    result = pd.eval(x) if isinstance(x, str) else x
                    if isinstance(result, (list, tuple, np.ndarray)):  
                        return sum(result) / len(result)  # Take the average if it's a range
                    return float(result)  # Convert to float for calculations
                except:
                    return np.nan  # Keep NaN values if conversion fails

            data['ground_truth_num'] = data['ground_truth'].apply(safe_eval).astype(float)

            data['cleaned_answers_num'] = data['cleaned_answers'].apply(safe_eval).astype(float)
           
            # Compute absolute error
            data['error'] = (data['ground_truth_num'] - data['cleaned_answers_num']).abs()

            # Compute MLAE
            data['mlae'] = np.log2(data['error'] + 0.125)

            mlae_values = data['mlae'].tolist()
            mae_values = data['error'].tolist()

            # Compute aggregated metrics
            avg_mlae = np.mean(mlae_values)
            std_mlae = np.std(mlae_values)
            avg_mae = np.mean(mae_values)

            # Compute confidence interval using bootstrapping
            try:
                bootstrap_result = bs.bootstrap(np.array(mlae_values), stat_func=bs_stats.std)
                confidence_value = 1.96 * bootstrap_result.value
            except:
                confidence_value = np.nan

            # Store metrics as a dictionary
            metrics_summary.append({
                'Dataset': df_name,
                'Model': model_name,
                'Average MLAE': round(avg_mlae, 2),
                'Average MAE': round(avg_mae, 2),
                'Std MLAE': round(std_mlae, 2),
                'Confidence Interval (95%)': round(confidence_value, 2)
            })
    
    # Convert list of dicts to DataFrame
    metrics_summary_df = pd.DataFrame(metrics_summary)
    
    return metrics_summary_df

def calculate_mlae_individual_EXP2():
    """Calculate individual MLAE values for each row in a dataframe."""

    df = pd.read_csv("/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/EXP2/finalEXP2.csv")

    # Safe function to evaluate values and handle tuples (ranges)
    def safe_eval(x):
        try:
            result = pd.eval(x) if isinstance(x, str) else x
            if isinstance(result, (list, tuple, np.ndarray)):  
                return sum(result) / len(result)  # Take the average if it's a range
            return float(result)  # Convert to float for calculations
        except:
            return np.nan  # Keep NaN values if conversion fails

    # Convert ground truth and cleaned answers to numeric values
    df['ground_truth_num'] = df['ground_truth'].apply(safe_eval)
    df['cleaned_answers_num'] = df['cleaned_answers'].apply(safe_eval)

    # Compute absolute error
    df['error'] = (df['ground_truth_num'] - df['cleaned_answers_num']).abs()

    # Compute MLAE, keeping NaN values
    df['mlae'] = np.log2(df['error'] + 0.125)


    return df


def calculate_average_mlae_EXP3(df):
    """
    Calculate metrics for each dataset and model.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing 'task_name', 'model_name', 'ground_truth', and 'cleaned_answers'.
    
    Returns:
    pandas.DataFrame: Table of metrics for all models and datasets.
    """

    # Dictionary to store all metrics
    metrics_summary = []
    
    for df_name, df_task in df.groupby('task_name'):  # Group by dataset
        for model_name, group in df_task.groupby('model_name'):  # Group by model
            
            # Create a copy to avoid modifying the original DataFrame
            data = group.copy()

            # Convert 'ground_truth' and 'cleaned_answers' to numeric safely
            def safe_eval(x):
                try:
                    result = pd.eval(x) if isinstance(x, str) else x
                    if isinstance(result, (list, np.ndarray)):  
                        return result[0]  # Extract first value if it's a list/array
                    return result
                except:
                    return np.nan  # Assign NaN if conversion fails

            data['ground_truth_num'] = data['ground_truth'].apply(safe_eval).astype(float)

            data['cleaned_answers_num'] = data['cleaned_answers'].apply(safe_eval).astype(float)
        
            # Compute absolute error
            data['error'] = (data['ground_truth_num'] - data['cleaned_answers_num']).abs()

            # Compute MLAE
            data['mlae'] = np.log2(data['error'] + 0.125)

            # Get values as lists
            mlae_values = data['mlae'].tolist()
            mae_values = data['error'].tolist() 

            # Compute aggregated metrics
            avg_mlae = np.mean(mlae_values) if mlae_values else np.nan
            std_mlae = np.std(mlae_values) if mlae_values else np.nan
            avg_mae = np.mean(mae_values) if mae_values else np.nan

            # Compute confidence interval using bootstrapping
            confidence_value = np.nan  # Default to NaN
            if mlae_values:
                try:
                    bootstrap_result = bs.bootstrap(np.array(mlae_values), stat_func=bs_stats.std)
                    confidence_value = 1.96 * bootstrap_result.value
                except Exception as e:
                    print(f"Bootstrap failed for {df_name} - {model_name}: {e}")

            # Store metrics as a dictionary
            metrics_summary.append({
                'Dataset': df_name,
                'Model': model_name,
                'Average MLAE': round(avg_mlae, 2),
                'Average MAE': round(avg_mae, 2),
                'Std MLAE': round(std_mlae, 2),
                'Confidence Interval (95%)': round(confidence_value, 2)
            })
    
    # Convert list of dicts to DataFrame
    metrics_summary_df = pd.DataFrame(metrics_summary)
    
    return metrics_summary_df

def calculate_mlae_individual_EXP3():

    df = pd.read_csv("/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/EXP3/finalEXP3.csv") 

    print(len(df))

    # Ensure numeric conversion of ground truth and cleaned answers
    df['ground_truth_num'] = pd.to_numeric(df['ground_truth'], errors='coerce')
    df['cleaned_answers_num'] = pd.to_numeric(df['cleaned_answers'], errors='coerce')

    # Compute absolute error
    df['error'] = (df['ground_truth_num'] - df['cleaned_answers_num']).abs()

    # Compute MLAE, keeping NaN values
    df['mlae'] = np.log2(df['error'] + 0.125)
    
    return df



def calculate_average_mlae_EXP4(df):
    """
    Calculate MLAE metrics for each dataset and model.

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'task_name', 'model_name', 'ground_truth', and 'cleaned_answers'.

    Returns:
    pandas.DataFrame: Table of metrics for all models and datasets.
    """

    metrics_summary = []  # Store results

    for df_name, df_task in df.groupby('task_name'):  # Group by dataset
        for model_name, group in df_task.groupby('model_name'):  # Group by model
            
            # Create a copy to avoid modifying the original DataFrame
            data = group.copy()

            # Convert 'ground_truth' and 'cleaned_answers' to numeric safely
            def safe_eval(x):
                try:
                    result = pd.eval(x) if isinstance(x, str) else x
                    if isinstance(result, (list, np.ndarray)):  
                        return result[0]  # Extract first value if it's a list/array
                    return result
                except:
                    return np.nan  # Assign NaN if conversion fails

            data['ground_truth_num'] = data['ground_truth'].apply(safe_eval).astype(float)

            data['cleaned_answers_num'] = data['cleaned_answers'].apply(safe_eval).astype(float)
           
            # Compute absolute error
            data['error'] = (data['ground_truth_num'] - data['cleaned_answers_num']).abs()

            # Compute MLAE
            data['mlae'] = np.log2(data['error'] + 0.125)

            # Get values as lists
            mlae_values = data['mlae'].tolist()
            mae_values = data['error'].tolist() 

            # Compute aggregated metrics
            avg_mlae = np.mean(mlae_values) if mlae_values else np.nan
            std_mlae = np.std(mlae_values) if mlae_values else np.nan
            avg_mae = np.mean(mae_values) if mae_values else np.nan

            # Compute confidence interval using bootstrapping
            confidence_value = np.nan  # Default to NaN
            if mlae_values:
                try:
                    bootstrap_result = bs.bootstrap(np.array(mlae_values), stat_func=bs_stats.std)
                    confidence_value = 1.96 * bootstrap_result.value
                except Exception as e:
                    print(f"Bootstrap failed for {df_name} - {model_name}: {e}")

            # Store metrics as a dictionary
            metrics_summary.append({
                'Dataset': df_name,
                'Model': model_name,
                'Average MLAE': round(avg_mlae, 2) if not np.isnan(avg_mlae) else np.nan,
                'Average MAE': round(avg_mae, 2) if not np.isnan(avg_mae) else np.nan,
                'Std MLAE': round(std_mlae, 2) if not np.isnan(std_mlae) else np.nan,
                'Confidence Interval (95%)': round(confidence_value, 2) if not np.isnan(confidence_value) else np.nan
            })

    # Convert list of dicts to DataFrame
    return pd.DataFrame(metrics_summary)

def calculate_mlae_individual_EXP4():
    """Calculate individual MLAE values for each row in a dataframe, keeping NaN values."""

    # Load the dataset
    df = pd.read_excel("/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/EXP4/finalEXP4.xlsx")


    # Safe function to evaluate values and handle tuples (ranges)
    def safe_eval(x):
        try:
            result = pd.eval(x) if isinstance(x, str) else x
            if isinstance(result, (list, tuple, np.ndarray)):  
                return sum(result) / len(result)  # Take the average if it's a range
            return float(result)  # Convert to float for calculations
        except:
            return np.nan  # Keep NaN values if conversion fails

    # Convert ground truth and cleaned answers to numeric values
    df['ground_truth_num'] = df['ground_truth'].apply(safe_eval)
    df['cleaned_answers_num'] = df['cleaned_answers'].apply(safe_eval)

    # Compute absolute error
    df['error'] = (df['ground_truth_num'] - df['cleaned_answers_num']).abs()

    # Compute MLAE, keeping NaN values
    df['mlae'] = np.log2(df['error'] + 0.125)

    return df

""" EXP5 """

def calculate_average_mlae_EXP5(df):
    """
    Calculate MLAE metrics for each dataset and model.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the results for different tasks.

    Returns:
    pandas.DataFrame: Table of metrics for all models and datasets.
    """
    
    # Dictionary to store metrics for each dataset
    metrics_summary = {}

    # Ensure ground truth and cleaned answers are numeric
    def safe_eval(x):
        try:
            result = ast.literal_eval(x) if isinstance(x, str) else x
            if isinstance(result, (list, tuple, np.ndarray)):  
                return sum(result) / len(result)  # Take the average if it's a range
            return float(result)  # Convert to float
        except:
            return np.nan  # Assign NaN if conversion fails

    df['ground_truth_num'] = df['ground_truth'].apply(safe_eval)
    df['cleaned_answers_num'] = df['cleaned_answers'].apply(safe_eval)

    # Iterate over each unique task_name (instead of requiring df_10, df_100, df_1000 separately)
    for task_name, df_task in df.groupby('task_name'):
        model_metrics = {}

        for model_name, data in df_task.groupby('model_name'):
            # Compute absolute error
            data['error'] = (data['ground_truth_num'] - data['cleaned_answers_num']).abs()

            # Compute MLAE
            data['mlae'] = np.log2(data['error'] + 0.125)

            mlae_values = data['mlae'].dropna().tolist()  # Remove NaN values
            mae_values = data['error'].dropna().tolist()

            # Compute aggregated metrics
            avg_mlae = np.mean(mlae_values) if mlae_values else np.nan
            std_mlae = np.std(mlae_values) if mlae_values else np.nan
            avg_mae = np.mean(mae_values) if mae_values else np.nan

            # Compute confidence interval using bootstrapping
            confidence_value = np.nan
            if mlae_values:
                try:
                    bootstrap_result = bs.bootstrap(np.array(mlae_values), stat_func=bs_stats.std)
                    confidence_value = 1.96 * bootstrap_result.value
                except Exception as e:
                    print(f"Bootstrap failed for {task_name} - {model_name}: {e}")

            model_metrics[model_name] = {
                'Dataset': task_name,
                'Model': model_name,
                'Average MLAE': round(avg_mlae, 2) if not np.isnan(avg_mlae) else np.nan,
                'Average MAE': round(avg_mae, 2) if not np.isnan(avg_mae) else np.nan,
                'Std MLAE': round(std_mlae, 2) if not np.isnan(std_mlae) else np.nan,
                'Confidence Interval (95%)': round(confidence_value, 2) if not np.isnan(confidence_value) else np.nan
            }
        
        if model_metrics:
            metrics_summary[task_name] = model_metrics

    # Convert to DataFrame
    metrics_table = pd.DataFrame([
        metrics 
        for dataset_metrics in metrics_summary.values() 
        for metrics in dataset_metrics.values()
    ])
    
    # Sort by dataset and MLAE
    if not metrics_table.empty:
        metrics_table = metrics_table.sort_values(['Dataset', 'Average MLAE'])
    
    return metrics_table


def calculate_mlae_individual_EXP5():
    """Calculate individual MLAE values for each row in finalEXP5.csv."""

    df = pd.read_csv("/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/EXP5/finalEXP5.csv")

    # Convert ground truth and cleaned answers to numeric, handling lists/strings safely
    df['ground_truth_num'] = df['ground_truth'].apply(lambda x: pd.eval(x) if isinstance(x, str) and x.strip().startswith('[') else pd.to_numeric(x, errors='coerce'))
    df['cleaned_answers_num'] = df['cleaned_answers'].apply(lambda x: pd.eval(x) if isinstance(x, str) and x.strip().startswith('[') else pd.to_numeric(x, errors='coerce'))

    # Drop rows with missing values after conversion
    df = df.dropna(subset=['ground_truth_num', 'cleaned_answers_num'])

    # Compute absolute error
    df['error'] = (df['ground_truth_num'] - df['cleaned_answers_num']).abs()

    # Compute MLAE, keeping NaN values
    df['mlae'] = np.log2(df['error'] + 0.125)

    return df



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
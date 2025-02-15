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

def clean_raw_answers(file_path):
    """
    Clean raw answers from CSV file, focusing only on extracting digits.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    tuple: (df_framed, df_unframed, deleted_rows) - Two DataFrames and list of deleted rows
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    

    print("Rows before cleaning:")
    print(f"  📥 Total rows: {len(df)}")

    for task in df['task_name'].unique():
        number_unique_images = df[df['task_name'] == task]['image_path'].nunique()
        print(f"Task {task}: {number_unique_images} unique images")
    
    deleted_rows = []
    total_tasks = len(df)

    # Tracking unique images per task
    print("\nUnique images per task:")
    for task in df['task_name'].unique():
        number_unique_images = df[df['task_name'] == task]['image_path'].nunique()
        print(f"Task {task}: {number_unique_images} unique images")
        
    def extract_digits_exp5(x):
        if pd.isna(x):
            return np.nan
        # Convert to string
        x = str(x)
        # Remove newline characters and whitespace
        x = x.strip().replace('\n', '')
        # If string starts with "user", extract the last number
        if x.startswith('user'):
            numbers = re.findall(r'\d+\.?\d*', x)
            return float(numbers[-1]) if numbers else np.nan
        # Extract first number found
        numbers = re.findall(r'\d+\.?\d*', x)
        return float(numbers[0]) if numbers else np.nan

    # Verify column names in DataFrame
    answer_column = 'answer' if 'answer' in df.columns else 'raw_answer'

    # Create new column with cleaned values
    df['parsed_answers'] = df[answer_column].apply(extract_digits_exp5)
    
    # Drop rows with NaN in parsed_answers
    df = df.dropna(subset=['parsed_answers'])
    
    # Track counts after dropping NaN
    print("\nCounts after dropping NaN values:")
    for task in df['task_name'].unique():
        clean_count = len(df[df['task_name'] == task])
        print(f"Task {task}: {clean_count} rows remaining")
    
    # Format cleaned values as strings with one decimal point
    df['parsed_answers'] = df['parsed_answers'].apply(
        lambda x: '{:.1f}'.format(x) if not pd.isna(x) else x
    )
    
    # Process rows for deletion
    rows_to_delete = []
    for index, row in df.iterrows():
        raw_text = row[answer_column]
        
        # Extract digits
        cleaned_answer = extract_digits_exp5(raw_text)
        
        # Check if answer is valid, otherwise mark for deletion
        if cleaned_answer is None:
            deleted_rows.append(row)
            rows_to_delete.append(index)
    
    # Drop rows marked for deletion
    df.drop(rows_to_delete, inplace=True)
    
    # Print deletion summary
    print(f"\nDeletion Summary:")
    print(f"Total initial tasks: {total_tasks}")
    print(f"Rows marked for deletion: {len(deleted_rows)}")
    
    # Process the dataframe with extracted digits
    df['cleaned_answers'] = df[answer_column].apply(extract_digits_exp5)
    
    # Split the dataframe by task
    df_10 = df[df['task_name'] == 10].copy()
    df_100 = df[df['task_name'] == 100].copy()
    df_1000 = df[df['task_name'] == 1000].copy()
    
    return df_10, df_100, df_1000, deleted_rows

def calculate_metrics(df_10, df_100, df_1000):
    """
    Calculate metrics for each dataset and model.

    Parameters:
    df_10, df_100, df_1000: DataFrames containing the results for each task.

    Returns:
    pandas.DataFrame: Table of metrics for all models and datasets.
    """
    # Dictionary to store metrics for each dataset
    metrics_summary = {}

    # List of DataFrames to process
    dataframes = {
        'Task_10': df_10, 
        'Task_100': df_100, 
        'Task_1000': df_1000
    }

    # Loop through each dataset
    for df_name, df in dataframes.items():
        model_metrics = {}
        
        for model_name, data in df.groupby('model_name'):
            # Convert ground truth and parsed answers to numeric
            data['ground_truth'] = pd.to_numeric(data['ground_truth'], errors='coerce')
            data['parsed_answers'] = pd.to_numeric(data['parsed_answers'], errors='coerce')
            data = data.dropna(subset=['ground_truth', 'parsed_answers'])
            
            if len(data) == 0:
                continue

            # Calculate MAE
            avg_mae = mean_absolute_error(data['ground_truth'], data['parsed_answers'])

            # Calculate MLAE
            data['mlae'] = data.apply(
                lambda row: np.log2(mean_absolute_error(
                    [row['ground_truth']], 
                    [row['parsed_answers']]
                ) + 0.125),
                axis=1
            )
            
            avg_mlae = data['mlae'].mean()
            std_mlae = data['mlae'].std()

            # Bootstrap confidence interval
            try:
                mlae_values = data['mlae'].dropna().values
                bootstrap_result = bs.bootstrap(
                    np.array(mlae_values), 
                    stat_func=bs_stats.std
                )
                confidence_value = 1.96 * bootstrap_result.value
            except Exception as e:
                print(f"Bootstrap error for {model_name} in {df_name}: {e}")
                confidence_value = np.nan
            
            # Store metrics
            model_metrics[model_name] = {
                'Dataset': df_name,
                'Model': model_name,
                'Average MLAE': round(avg_mlae, 2),
                'Average MAE': round(avg_mae, 2),
                'Std MLAE': round(std_mlae, 2),
                'Confidence Interval (95%)': round(confidence_value, 2)
            }
        
        metrics_summary[df_name] = model_metrics

    # Combine all metrics into a DataFrame
    metrics_table = pd.DataFrame([
        metrics 
        for dataset_metrics in metrics_summary.values() 
        for metrics in dataset_metrics.values()
    ])
    
    # Sort by dataset and MLAE
    if not metrics_table.empty:
        metrics_table = metrics_table.sort_values(['Dataset', 'Average MLAE'])
    
    return metrics_table


def process_plot(metrics_table):
    """
    Process and create the plot for given metrics table
    """
    print("\nGenerating plot...")
    plot_results(metrics_table)
   

""" Average 3 running """
def average_metrics(metrics_list):

    """Helper function to average metrics across multiple runs"""
   
    all_metrics_df = pd.concat(metrics_list)
    
    # Group by Dataset and Model, then calculate averages
    averaged_metrics = all_metrics_df.groupby(['Dataset', 'Model']).agg({
        'Average MLAE': 'mean',
        'Average MAE': 'mean',  # Include Average MAE
        'Std MLAE': 'mean',
        'Confidence Interval (95%)': 'mean'
    }).reset_index()
    
    for col in ['Average MLAE', 'Std MLAE', 'Confidence Interval (95%)']:
        averaged_metrics[col] = averaged_metrics[col].round(2)
    
    return averaged_metrics

def process_and_plot_multiplerun():
    """Process three EXP5 result files and create averaged plot"""
    # Define file paths
    file_paths = [
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberone/EXP-Results/EXP5results55images.csv",
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numbertwo/EXP-Results/EXP5results55images.csv",
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberthree/EXP-Results/EXP5results55images.csv"
    ]

    # Calculate metrics for each file and store them
    all_metrics = []
    for i, file_path in enumerate(file_paths, 1):
        print(f"\nProcessing File {i}: {os.path.basename(file_path)}")
        
        # Process each file
        df_10, df_100, df_1000, deleted_rows = clean_raw_answers(file_path)
        
        # Print number of rows for each task
        print(f"\nNumber of rows in each task:")
        print(f"Task 10 rows: {len(df_10)}")
        print(f"Task 100 rows: {len(df_100)}")
        print(f"Task 1000 rows: {len(df_1000)}")
        
        # Calculate metrics
        metrics = calculate_metrics(df_10, df_100, df_1000)
        print("\nMetrics for this file:")
        print(metrics)
        
        all_metrics.append(metrics)

    # Calculate and print averaged metrics
    print("\nAveraged Metrics across all files:")
    averaged_metrics_table = average_metrics(all_metrics)
    print(averaged_metrics_table)

    # Plot using the averaged metrics
    plot_results(averaged_metrics_table)

    return averaged_metrics_table

def balance_datasets(df_10, df_100, df_1000, target_size=824):
    """
    Balance datasets to specified target sizes, using replacement if necessary.
    
    Args:
        df_10: DataFrame for 10-sample task
        df_100: DataFrame for 100-sample task
        df_1000: DataFrame for 1000-sample task
        target_size: Optional target size override
    
    Returns:
        Tuple of balanced DataFrames (df_10_balanced, df_100_balanced, df_1000_balanced)
    """
    # If no target size specified, use the intended size for each dataset
    if target_size is None:
        df_10_balanced = df_10.sample(n=10, replace=len(df_10) < 10, random_state=42)
        df_100_balanced = df_100.sample(n=100, replace=len(df_100) < 100, random_state=42)
        df_1000_balanced = df_1000.sample(n=1000, replace=len(df_1000) < 1000, random_state=42)
    else:
        # If target size is specified, use it for all datasets
        df_10_balanced = df_10.sample(n=target_size, replace=len(df_10) < target_size, random_state=42)
        df_100_balanced = df_100.sample(n=target_size, replace=len(df_100) < target_size, random_state=42)
        df_1000_balanced = df_1000.sample(n=target_size, replace=len(df_1000) < target_size, random_state=42)
    
    return df_10_balanced, df_100_balanced, df_1000_balanced

def clean_and_balance_csv_files():
    """Process and check deleted rows across all CSV files"""
    file_paths = [
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberone/EXP-Results/EXP5results55images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numbertwo/EXP-Results/EXP5results55images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberthree/EXP-Results/EXP5results55images.csv'
    ]
    all_balanced_metrics = []
    all_deleted_dfs = []
    balanced_dataframes = []
    
    for file_path in file_paths:
        df_10, df_100, df_1000, deleted_rows = clean_raw_answers(file_path)
        print(f"\n📂 Processing: {file_path.split('/')[-1]}")
        print(f"\nNumber of rows in each task:")
        print(f"Task 10 samples rows: {len(df_10)}")
        print(f"Task 100 samples rows: {len(df_100)}")
        print(f"Task 1000 samples rows: {len(df_1000)}")
        
        # Balance datasets
        df_10, df_100, df_1000 = balance_datasets(df_10, df_100, df_1000)

        # Store the balanced DataFrames for later use
        balanced_dataframes.append((df_10, df_100, df_1000))
    

        # Flatten the list of tuples and merge all into one DataFrame
        balanced_df = pd.concat([df for group in balanced_dataframes for df in group], ignore_index=True)

        # Now you can save it to an Excel file
        balanced_df.to_csv("finalEXP5.csv", index=False)

        print("\nBalanced datasets:")
        print(f"Task 10 samples balanced rows: {len(df_10)}")
        print(f"Task 100 samples balanced rows: {len(df_100)}")
        print(f"Task 1000 samples balanced rows: {len(df_1000)}")
        
        metrics_table = calculate_metrics(df_10, df_100, df_1000)
        all_balanced_metrics.append(metrics_table)
        
        if deleted_rows:
           print(f"Deleted rows from {file_path.split('/')[-1]}:")
           print(pd.DataFrame(deleted_rows)[['raw_answer', 'model_name']])  # Or just `deleted_rows` if it's a list of dicts
           deleted_df = pd.DataFrame(deleted_rows)[['raw_answer', 'model_name']]
           deleted_df['file'] = file_path.split('/')[-1]
           all_deleted_dfs.append(deleted_df)

    
    # Combine metrics
    combined_metrics = pd.concat(all_balanced_metrics, ignore_index=True) if all_balanced_metrics else pd.DataFrame()
    
    # Combine deleted rows
    if all_deleted_dfs:
        combined_deleted_df = pd.concat(all_deleted_dfs, ignore_index=True)
    else:
        combined_deleted_df = pd.DataFrame(columns=['file', 'raw_answer', 'model_name'])

    return combined_metrics, balanced_dataframes, combined_deleted_df


def plot_multiplerun(metrics_table):
    """
    Plot results for multiple runs using balanced datasets metrics.
    
    Args:
        metrics_table (pd.DataFrame): Metrics table containing all runs
    
    Returns:
        pd.DataFrame: Averaged metrics table
    """
    print("\nAveraged Metrics (Balanced Datasets):")
    
    # Convert the metrics_table DataFrame into a list with a single DataFrame
    metrics_list = [metrics_table]
    
    # Calculate averaged metrics
    averaged_metrics_table = average_metrics(metrics_list)
    
    # Plot the results
    plot_results(averaged_metrics_table)
    
    return averaged_metrics_table


def find_task_images(base_path, task_types=None):

    """
    Automatically find images for each task in the given directory
    Parameters:
    base_path (str): Base directory path where images are stored
    task_types (list): List of task types (e.g., ['unframed', 'framed']). If None, will look for both.
    Returns:
    dict: Dictionary mapping task names to image paths
    """
    task_images = {}
    image_extensions = ['.jpg', '.jpeg', '.png']

    if task_types is None:
        task_types = ['type1', 'type2', 'type3', 'type4', 'type5']

    for task in task_types:
        task_pattern = f"{task}_"
        
        for file in os.listdir(base_path):
            if file.startswith(task_pattern) and any(file.lower().endswith(ext) for ext in image_extensions):
                task_images[task] = os.path.join(base_path, file)
                break

    return task_images

import ast

def check_combined_unique_answers(df):
    """
    Analyze unique values in `cleaned_answers` and compare Pretrained vs. Finetuned models.
    """

    # Define model groups
    pretrained_models = {'Gemini1_5Flash', 'GeminiProVision', 'LLaMA', 'gpt4o'}
    finetuned_models = {'CustomLLaMA'}

    # Function to safely evaluate cleaned_answers values
    def safe_eval(x):
        try:
            if isinstance(x, str):
                x = x.strip()  # Remove unwanted spaces/newlines
                return [float(i) for i in ast.literal_eval(x)] if x.startswith("[") else [float(x)]
            elif isinstance(x, (int, float)):
                return [x]  # Convert single numbers to list for uniform processing
            elif isinstance(x, list):
                return x  # Already a list, return as is
            return np.nan
        except:
            return np.nan

    # Convert cleaned_answers values to proper lists/numeric values
    df['cleaned_answers'] = df['cleaned_answers'].apply(safe_eval)

    # Assign model type
    df['new_model_type'] = df['model_name'].apply(
        lambda x: 'Pretrained' if x in pretrained_models else 'Finetuned' if x in finetuned_models else 'Unknown'
    )

    # Dictionary to store grouped data dynamically
    grouped_tasks = {
        f"{str(task_name)}_pretrained": df[(df["task_name"] == task_name) & (df["new_model_type"] == "Pretrained")]
        for task_name in df["task_name"].unique()
    }
    grouped_tasks.update({
        f"{str(task_name)}_finetuned": df[(df["task_name"] == task_name) & (df["new_model_type"] == "Finetuned")]
        for task_name in df["task_name"].unique()
    })

    # Define unique colors for each task_name and model_type combination
    task_colors = {
        ('10', 'pretrained'): '#3A50A1',  # Dark Blue
        ('100', 'pretrained'): '#1F77B4',  # Medium Blue
        ('1000', 'pretrained'): '#7EA6E0',  # Light Blue
        ('10', 'finetuned'): '#4C8C2B',  # Dark Green
        ('100', 'finetuned'): '#2CA02C',  # Medium Green
        ('1000', 'finetuned'): '#A3D977',  # Light Green
    }

    # Define marker styles
    task_markers = {'pretrained': 'o', 'finetuned': '^'}

    # Prepare figure: one plot for Pretrained, one for Finetuned
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ranges = [(1, 10), (16, 7000)] 
    model_types = ['Pretrained', 'Finetuned']

    # Loop through model types and axes
    for ax, model_type in zip(axes, model_types):
        for range_min, range_max in ranges:
            for task_name, task_data in grouped_tasks.items():
                if model_type.lower() not in task_name:
                    continue  # Skip irrelevant tasks

                # Flatten cleaned_answers
                flattened_answers = task_data['cleaned_answers'].dropna().explode()
                flattened_answers = flattened_answers.apply(
                    lambda x: x if isinstance(x, (int, float)) else x[0] if isinstance(x, list) and len(x) > 0 else None
                ).dropna()

                # Convert to numeric and count unique values
                flattened_answers = pd.to_numeric(flattened_answers, errors='coerce').dropna()
                value_counts = flattened_answers.value_counts().sort_index()

                # Filter by range
                filtered_counts = value_counts[
                    (value_counts.index >= range_min) & (value_counts.index <= range_max)
                ]

                if filtered_counts.empty:
                    print(f"⚠️ No values found for {model_type} in range {range_min} to {range_max} for {task_name}")
                    continue

                # Define marker and color for each task_name
                task_key = (str(task_name).split('_')[0], model_type.lower())  # Extract task number
                color = task_colors.get(task_key, "#808080")  # Default gray if missing
                marker = task_markers[model_type.lower()]

                # Plot points
                ax.scatter(
                    filtered_counts.index, filtered_counts.values,
                    label=f"{task_key[0]} {model_type.lower()}",
                    color=color, marker=marker, alpha=0.7, s=50
                )

        # Plot settings
        ax.set_xlabel("Unique Values")
        ax.set_ylabel("Counts")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Task Type", fontsize=8, loc="upper right")

    # Save and show plot
    plt.savefig('analysisexp5.png', bbox_inches='tight', format='png', transparent=False, dpi=300)
    plt.show()



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

# Single CSV file to process
csv_file = '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP3/finetuning-EXP3numberone/EXP-Results/EXP3numberone_results.csv'

import pandas as pd
import numpy as np
import re

# Display all columns
pd.set_option('display.max_columns', None)

# Display all rows
pd.set_option('display.max_rows', None)

# Width of the display in characters
pd.set_option('display.width', None)

# Don't wrap long strings
pd.set_option('display.max_colwidth', None)

def clean_raw_answers(file_path):
    """
    Clean raw answers from CSV file, focusing only on extracting digits.
    Parameters:
    file_path (str): Path to the CSV file
    Returns:
    pandas.DataFrame: DataFrame with raw and cleaned answers
    """
    deleted_rows = []

    # Read the CSV file
    df = pd.read_csv(file_path)

    print("Rows before cleaning:")
    print(f"  📥 Total rows: {len(df)}")

    for task in df['task_name'].unique():
        number_unique_images = df[df['task_name'] == task]['image_path'].nunique()
        print(f"Task {task}: {number_unique_images} unique images")


    # First check your data
    #print("DataFrame columns:", df.columns)

    deleted_rows = []
    total_tasks = len(df)

    def extract_digits_exp3(raw_text):
        """
        Extracts the last valid decimal value or fraction from the raw answer. 
        Handles fractions by evaluating and formatting them properly.

        Parameters:
        raw_text (str): The raw answer as a string.

        Returns:
        float or np.nan: Extracted value or NaN if no valid value is found.
        """
        if pd.isna(raw_text):
            return np.nan
        
        # Convert raw_answer to string to handle all cases
        raw_text = str(raw_text)
        
        # Find fractions and decimals in the raw answer
        fraction_matches = re.findall(r'\d+/\d+', raw_text)
        decimal_matches = re.findall(r'0\.\d{1,3}', raw_text)
        
        # Combine matches
        all_matches = fraction_matches + decimal_matches
        
        # Process the last match if available
        if all_matches:
            last_match = all_matches[-1]
            if '/' in last_match:  # If it's a fraction
                try:
                    fraction_value = eval(last_match)  # Evaluate the fraction
                    return round(fraction_value, 2)  # Round to 2 decimal places
                except ZeroDivisionError:
                    return np.nan
            else:  # It's a decimal
                return round(float(last_match), 2)
        return np.nan

    # Initialize variables
    total_tasks = len(df)
    deleted_rows = []
    rows_to_delete = []

    # First pass to identify rows to delete
    for index, row in df.iterrows():
        raw_text = row['raw_answer']  # Changed from row.get()
        cleaned_answer = extract_digits_exp3(raw_text)
        
        if pd.isna(cleaned_answer):
            deleted_rows.append(row)
            rows_to_delete.append(index)

    # Drop identified rows once
    if rows_to_delete:
        df.drop(rows_to_delete, inplace=True)

    # Print summary once
    print(f"Total tasks: {total_tasks}")
    print(f"Rows deleted: {len(deleted_rows)}")


    # Apply cleaning function to the 'raw_answer' column
    df['cleaned_answers'] = df['raw_answer'].apply(extract_digits_exp3)

    # Split the dataframe by task
    df_type1 = df[df['task_name'] == 'type1'].copy()
    df_type2 = df[df['task_name'] == 'type2'].copy()
    df_type3 = df[df['task_name'] == 'type3'].copy()
    df_type4 = df[df['task_name'] == 'type4'].copy()
    df_type5 = df[df['task_name'] == 'type5'].copy()

    return df_type1, df_type2, df_type3, df_type4, df_type5, deleted_rows
def plot_results(metrics_table):
    """
    Plot the results from the metrics table with human benchmark values
    """
    summary_stats_by_task = {df_name: metrics_table[metrics_table['Dataset'] == df_name] 
                            for df_name in metrics_table['Dataset'].unique()}

    # Define display names for tasks
    task_display_names = {
        'type1': 'TYPE 1',
        'type2': 'TYPE 2',
        'type3': 'TYPE 3',
        'type4': 'TYPE 4',
        'type5': 'TYPE 5',
    }

    # Define colors for models and Human benchmarks
    model_colors = {
        'CustomLLaMA': '#8E44AD',   # Purple
        'Gemini1_5Flash': '#3498DB',    # Blue
        'GeminiProVision': '#E74C3C',   # Red
        'LLaMA': '#E67E22',             # Orange
        'gpt4o': '#27AE60',             # Green
        'Human': '#34495E'              # Dark Gray
    }

    # Define Human benchmark data
    human_data = {
        'type1': (2.72, 0.155),
        'type2': (2.35, 0.175),
        'type3': (1.84, 0.16),
        'type4': (1.72, 0.2),
        'type5': (1.4, 0.14)
    }

    # Get task images
    base_path = '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP3/finetuning-EXP3numberone/images'
    task_images = find_task_images(base_path)
    
    num_tasks = len(summary_stats_by_task)
    fig, axes = plt.subplots(num_tasks, 3, figsize=(14, 4 * num_tasks), 
                            gridspec_kw={'width_ratios': [1, 4, 1]}, sharex=False)
    fig.subplots_adjust(hspace=0.8, left=0.05, right=0.95)
    fig.patch.set_facecolor('white')

    # Handle both single and multiple subplot cases
    if num_tasks == 1:
        axes = np.array([axes])  # Convert to 2D array with one row

    for i, (task_name, task_data) in enumerate(summary_stats_by_task.items()):
        ax_img, ax_plot, ax_label = axes[i]

        if task_name in task_images:
            img_path = task_images[task_name]
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("L")
                img_inverted = ImageOps.invert(img)
                img_with_border = ImageOps.expand(img_inverted.convert("RGB"), border=1, fill="black")
                ax_img.imshow(img_with_border)
                ax_img.set_facecolor("white")
            else:
                ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=10, color="black")
                ax_img.set_facecolor("white")

        ax_img.axis('off')
        ax_img.set_title(task_display_names.get(task_name, task_name), loc="left", fontsize=12, color="black")

        sorted_model_names = sorted(task_data['Model'].unique())
        y_positions = np.arange(len(sorted_model_names))

        for j, model_name in enumerate(sorted_model_names):
            model_data = task_data[task_data['Model'] == model_name]
            mlae_value = model_data['Average MLAE'].values[0]
            confidence_interval = model_data['Confidence Interval (95%)'].values[0]

            ax_plot.errorbar(mlae_value, j, xerr=confidence_interval, fmt='o', 
                           color=model_colors.get(model_name, 'gray'), capsize=5, 
                           label=f"{model_name}" if i == 0 else None)

        # Plot human benchmark as horizontal error bar
        if task_name in human_data:
            human_value, human_std = human_data[task_name]
            human_interval = human_std * 1.96
            y_pos = len(sorted_model_names) + 0.5
            
            ax_plot.errorbar(human_value, y_pos, xerr=human_interval, 
                           fmt='s', color=model_colors['Human'], 
                           capsize=5, capthick=1.5,
                           markersize=7, label='Human' if i == 0 else None)

        ax_plot.axvline(-4, color="black", linewidth=1)
        ax_plot.axvline(-14, color="black", linewidth=1)
        ax_plot.grid(False)

        for offset in np.linspace(-0.05, 0.05, 10):
            ax_plot.axvline(0 + offset, color="gray", alpha=0.1, linewidth=0.5)

        ax_plot.spines['top'].set_visible(False)
        ax_plot.spines['right'].set_visible(False)
        ax_plot.spines['left'].set_visible(False)
        ax_plot.spines['bottom'].set_position(('outward', 10))

        ax_plot.set_yticks(y_positions)
        ax_plot.set_yticklabels([])
        ax_plot.set_xlim(-4, 4)
        ax_plot.invert_yaxis()

        ax_label.set_yticks(y_positions)
        ax_label.set_yticklabels(sorted_model_names, fontsize=10)
        ax_label.tick_params(left=False, right=False, labelleft=False, labelright=True)
        ax_plot.tick_params(axis='y', which='both', left=False, right=False)
        ax_label.set_ylim(ax_plot.get_ylim())
        ax_label.axis("off")

    if num_tasks > 0:  # Only add legend if there are tasks
        axes[0, 1].legend(loc="best", frameon=False)
    plt.show()


def process_plot(metrics_table):
    """
    Process and create the plot for given metrics table
    """
    print("\nGenerating plot...")
    plot_results(metrics_table)

def calculate_metrics(df_type1, df_type2, df_type3, df_type4, df_type5):
    """
    Calculate metrics for each dataset and model.
    Parameters:
    df_type1, df_type2, df_type3, df_type4, df_type5: DataFrames containing the results for each task
    Returns:
    pandas.DataFrame: Table of metrics for all models and datasets
    """
    # Dictionary to store metrics for each dataset
    metrics_summary = {}

    # List of DataFrames to process
    dataframes = {
        'type1': df_type1, 
        'type2': df_type2, 
        'type3': df_type3,
        'type4': df_type4, 
        'type5': df_type5,
    }

    # Loop through each dataset
    for df_name, df in dataframes.items():
        model_metrics = {}
        
        for model_name, group in df.groupby('model_name'):
            data = group.copy()
            
            data['ground_truth_num'] = data['ground_truth'].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)
            data['cleaned_answers_num'] = data['cleaned_answers'].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)
            
            data = data.dropna(subset=['ground_truth_num', 'cleaned_answers_num'])
            
            if len(data) == 0:
                continue
                
            mlae_values = []
            mae_values = []
            for _, row in data.iterrows():
                # Calculate MLAE
                mlae = np.log2(mean_absolute_error(
                    [row['ground_truth_num']], 
                    [row['cleaned_answers_num']]
                ) + 0.125)
                mlae_values.append(mlae)
                
                # Calculate MAE
                mae = mean_absolute_error(
                    [row['ground_truth_num']], 
                    [row['cleaned_answers_num']]
                )
                mae_values.append(mae)
            
            data.loc[:, 'mlae'] = mlae_values
            data.loc[:, 'mae'] = mae_values
            
            avg_mlae = np.mean(mlae_values)
            std_mlae = np.std(mlae_values)
            avg_mae = np.mean(mae_values)
            
            try:
                bootstrap_result = bs.bootstrap(
                    np.array(mlae_values), 
                    stat_func=bs_stats.std
                )
                confidence_value = 1.96 * bootstrap_result.value
            except:
                confidence_value = np.nan
            
            model_metrics[model_name] = {
                'Dataset': df_name,
                'Model': model_name,
                'Average MLAE': round(avg_mlae, 2),
                'Average MAE': round(avg_mae, 2),
                'Std MLAE': round(std_mlae, 2),
                'Confidence Interval (95%)': round(confidence_value, 2)
            }
        
        if model_metrics:
            metrics_summary[df_name] = model_metrics

    metrics_table = pd.DataFrame([
        metrics 
        for dataset_metrics in metrics_summary.values() 
        for metrics in dataset_metrics.values()
    ])
    
    if not metrics_table.empty:
        metrics_table = metrics_table.sort_values(['Dataset', 'Average MLAE'])
    
    return metrics_table

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
""" Average 3 running """


def average_metrics(metrics_list):
    """
    Helper function to average metrics across multiple runs, including MAE.
    
    Parameters:
    metrics_list (list of DataFrames): List of metrics DataFrames to average.
    
    Returns:
    pandas.DataFrame: DataFrame with averaged metrics.
    """
    # Combine all metrics DataFrames
    all_metrics_df = pd.concat(metrics_list)
    
    # Group by Dataset and Model, then calculate averages
    averaged_metrics = all_metrics_df.groupby(['Dataset', 'Model']).agg({
        'Average MLAE': 'mean',
        'Average MAE': 'mean',  # Include Average MAE
        'Std MLAE': 'mean',
        'Confidence Interval (95%)': 'mean'
    }).reset_index()
    
    # Round the values for presentation
    for col in ['Average MLAE', 'Average MAE', 'Std MLAE', 'Confidence Interval (95%)']:
        averaged_metrics[col] = averaged_metrics[col].round(2)
    
    return averaged_metrics

def balance_datasets(df_type1, df_type2, df_type3, df_type4, df_type5, target_size=737):

    # Determine the target size based on the smallest dataset or provided value
    if target_size is None:
        min_samples = min(len(df_type1), len(df_type2), len(df_type3), len(df_type4), len(df_type5))
    else:
        min_samples = target_size
    
    # Balance each dataset
    df_type1_balanced = (
        df_type1.sample(n=min_samples, random_state=42) if len(df_type1) > min_samples else df_type1
    )
    df_type2_balanced = (
        df_type2.sample(n=min_samples, random_state=42) if len(df_type2) > min_samples else df_type2
    )
    df_type3_balanced = (
        df_type3.sample(n=min_samples, random_state=42) if len(df_type3) > min_samples else df_type3
    )
    df_type4_balanced = (
        df_type4.sample(n=min_samples, random_state=42) if len(df_type4) > min_samples else df_type4
    )
    df_type5_balanced = (
        df_type5.sample(n=min_samples, random_state=42) if len(df_type5) > min_samples else df_type5
    )
    
    return df_type1_balanced, df_type2_balanced, df_type3_balanced, df_type4_balanced, df_type5_balanced

def checkdeletedrows_forallcsv():
    """Process and check deleted rows across all CSV files"""
    file_paths = [
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP3/finetuning-EXP3numberone/EXP-Results/EXP3results55images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP3/finetuning-EXP3numbertwo/EXP-Results/EXP3results55images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP3/finetuning-EXP3numberthree/EXP-Results/EXP3results55images.csv'
    ]
    all_balanced_metrics = []
    all_deleted_dfs = []
    
    for file_path in file_paths:
        df_type1, df_type2, df_type3, df_type4, df_type5, deleted_rows = clean_raw_answers(file_path)
        print(f"\n📂 Processing: {file_path.split('/')[-1]}")
        print(f"\nNumber of rows in each task:")
        print(f"Task type1 rows: {len(df_type1)}")
        print(f"Task type2 rows: {len(df_type2)}")
        print(f"Task type3 rows: {len(df_type3)}")
        print(f"Task type4 rows: {len(df_type4)}")
        print(f"Task type5 rows: {len(df_type5)}")
        
        # Balance datasets
        df_type1, df_type2, df_type3, df_type4, df_type5 = balance_datasets(
            df_type1, df_type2, df_type3, df_type4, df_type5
        )
        print("\nBalanced datasets:")
        print(f"Task type1 balanced rows: {len(df_type1)}")
        print(f"Task type2 balanced rows: {len(df_type2)}")
        print(f"Task type3 balanced rows: {len(df_type3)}")
        print(f"Task type4 balanced rows: {len(df_type4)}")
        print(f"Task type5 balanced rows: {len(df_type5)}")
        
        metrics_table = calculate_metrics(df_type1, df_type2, df_type3, df_type4, df_type5)
        all_balanced_metrics.append(metrics_table)
        
        if deleted_rows:
            deleted_df = pd.DataFrame(deleted_rows)[['raw_answer', 'model_name']]
            deleted_df['file'] = file_path.split('/')[-1]
            all_deleted_dfs.append(deleted_df)
    
    # Combine metrics
    combined_metrics = pd.concat(all_balanced_metrics, ignore_index=True) if all_balanced_metrics else pd.DataFrame()
    
    # Combine deleted rows
    if all_deleted_dfs:
        combined_deleted_df = pd.concat(all_deleted_dfs, ignore_index=True)
        return combined_metrics, combined_deleted_df[['file', 'raw_answer', 'model_name']]
    return combined_metrics, pd.DataFrame(columns=['file', 'raw_answer', 'model_name'])


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

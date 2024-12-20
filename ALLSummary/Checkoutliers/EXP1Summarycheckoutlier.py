

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

from matplotlib.ticker import ScalarFormatter


""" Clean raw answers and clean outliers """

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

    task_counts = {}
    for task in df['task_name'].unique():
        task_rows = len(df[df['task_name'] == task])
        task_counts[task] = task_rows
        number_unique_images = df[df['task_name'] == task]['image_path'].nunique()
        print(f"  Task {task}: {task_rows} rows, {number_unique_images} unique images")

    print(f" Total tasks: {sum(task_counts.values())}")


    def extract_digits_exp1(raw_text):
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
        
        
        # Convert to string and clean
        raw_text = str(raw_text).strip().replace('\n', '')
        
        # If string starts with "user", extract the last number
        if raw_text.startswith('user'):
            numbers = re.findall(r'\d+\.?\d*', raw_text)
            return float(numbers[-1]) if numbers else np.nan
            
        # Extract the first number found otherwise
        numbers = re.findall(r'\d+\.?\d*', raw_text)
        return float(numbers[0]) if numbers else np.nan

    # Initialize variables
    total_tasks = len(df)
    deleted_rows = []
    rows_to_delete = []

    # First pass to identify rows to delete
    for index, row in df.iterrows():
        raw_text = row['raw_answer']  # Changed from row.get()
        cleaned_answer = extract_digits_exp1(raw_text)
        
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
    df['cleaned_answers'] = df['raw_answer'].apply(extract_digits_exp1)

    # Split the dataframe by task
    df_volume = df[df['task_name'] == 'volume'].copy()
    df_area = df[df['task_name'] == 'area'].copy()
    df_direction = df[df['task_name'] == 'direction'].copy()
    df_length = df[df['task_name'] == 'length'].copy()
    df_position_common_scale = df[df['task_name'] == 'position_common_scale'].copy()
    df_position_non_aligned_scale = df[df['task_name'] == 'position_non_aligned_scale'].copy()
    df_angle = df[df['task_name'] == 'angle'].copy()
    df_curvature = df[df['task_name'] == 'curvature'].copy()
    df_shading = df[df['task_name'] == 'shading'].copy()

    return df_volume, df_area, df_direction, df_length, df_position_common_scale, df_position_non_aligned_scale, df_angle, df_curvature, df_shading, deleted_rows

# Process one file
def process_one_file(file_path):
    print(f"\nProcessing file: {file_path}")
    
    # Clean raw answers
    df_volume, df_area, df_direction, df_length, df_position_common_scale, \
    df_position_non_aligned_scale, df_angle, df_curvature, df_shading, deleted_rows = clean_raw_answers(file_path)

    # Combine all tasks into a single DataFrame for this file
    file_cleaned_df = pd.concat([
        df_volume, df_area, df_direction, df_length, df_position_common_scale,
        df_position_non_aligned_scale, df_angle, df_curvature, df_shading
    ], ignore_index=True)

    return file_cleaned_df, deleted_rows

# Process each file individually
def process_files_individually(file_paths):
    individual_file_results = {}  # Dictionary to store results for each file

    for i, file_path in enumerate(file_paths):
        print(f"\n### Processing File {i + 1}/{len(file_paths)} ###")
        
        # Process the current file
        file_cleaned_df, deleted_rows = process_one_file(file_path)
        
        # Store the results separately for each file
        individual_file_results[file_path] = {
            "cleaned_df": file_cleaned_df,
            "deleted_rows": deleted_rows
        }
        
        # Print file-specific stats
        print(f"File {file_path}:")
        print(f"  Cleaned DataFrame Shape: {file_cleaned_df.shape}")
        print(f"  Deleted Rows: {len(deleted_rows)}")

    return individual_file_results

def visualize_clean_data(individual_results):
    """
    Function to visualize distribution of clean data across multiple tasks.
    
    Args:
        individual_results (dict): Dictionary containing DataFrames for each file
    """
    tasks = [
        'position_common_scale', 'position_non_aligned_scale',
        'length', 'direction', 'angle', 'area', 'volume',
        'curvature', 'shading'
    ]
    
    colors = ['blue', 'green', 'purple', 'orange', 'red', 
              'cyan', 'magenta', 'brown', 'gray']

    def plot_task_distribution(ax, task_name, cleaned_df, color):
        """Plot the distribution of cleaned answers for a specific task."""
        task_df = cleaned_df[cleaned_df['task_name'] == task_name].copy()
        
        # Ensure consistent rounding for specific tasks like 'area'
        if task_name == 'volume':
            task_df['cleaned_answers'] = task_df['cleaned_answers'].astype(float).astype(int)
        elif task_name == 'curvature':
            # For curvature, no changes required since it uses float values directly
            pass
        else:
            # Round cleaned_answers to three decimal places for Area and other tasks
            task_df['cleaned_answers'] = task_df['cleaned_answers'].astype(float).round()

        # Calculate unique counts for the plot
        answer_counts = task_df['cleaned_answers'].value_counts().sort_index()
        
        # Plot scatter
        ax.scatter(answer_counts.index, answer_counts.values, color=color, alpha=0.7)
        ax.set_title(task_name.replace('_', ' ').title(), fontsize=10)
        ax.set_xlabel("Cleaned Answers", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(axis='both', linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(useOffset=False, style='plain', axis='x')

        # Print summary for verification
        print(f"\nPlotting '{task_name}':")
        print(f"Counts used for plotting:\n{answer_counts}")

    def plot_tasks_for_file(cleaned_df, file_name):
        """Plot all tasks for a specific file."""
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, task_name in enumerate(tasks):
            plot_task_distribution(axes[i], task_name, cleaned_df, colors[i])
        
        plt.tight_layout()
        fig.suptitle(f"Task Distribution for {file_name}", fontsize=14, y=1.02)
        plt.show()

    def process_plot_run(results):
        """Process and plot results for each file."""
        for file_path, result in results.items():
            print(f"\n### Processing File: {file_path.split('/')[-1]} ###")
            
            # Only use the cleaned data
            cleaned_df = result['cleaned_df']
            print(f"Plotting distribution for cleaned data with {len(cleaned_df)} rows")
            
            # Plot the cleaned data directly
            plot_tasks_for_file(cleaned_df, file_path.split('/')[-1])

    # Execute the analysis
    process_plot_run(individual_results)





"""" AFTER Balancing"""

def calculate_metrics(individual_file_results):
    """
    Calculate metrics for each dataset and model.
    Parameters:
    df_volume, df_area, df_direction, df_length, df_position_common_scale, df_position_non_aligned_scale, df_angle, df_curvature, df_shading: DataFrames containing the results for each task
    Returns:
    pandas.DataFrame: Table of metrics for all models and datasets
    """
    # Dictionary to store metrics for each dataset
    metrics_summary = {}

    for file_path, result in individual_file_results.items():
        # Extract filtered DataFrame for the current file
        filtered_df = result['filtered_df']

        # Group the filtered DataFrame by tasks
        tasks = {
            'volume': filtered_df[filtered_df['task'] == 'df_volume'],
            'area': filtered_df[filtered_df['task'] == 'df_area'],
            'direction': filtered_df[filtered_df['task'] == 'df_direction'],
            'length': filtered_df[filtered_df['task'] == 'df_length'],
            'position_common_scale': filtered_df[filtered_df['task'] == 'df_position_common_scale'],
            'position_non_aligned_scale': filtered_df[filtered_df['task'] == 'df_position_non_aligned_scale'],
            'angle': filtered_df[filtered_df['task'] == 'df_angle'],
            'curvature': filtered_df[filtered_df['task'] == 'df_curvature'],
            'shading': filtered_df[filtered_df['task'] == 'df_shading']
        }

        # Loop through each dataset
        for task_name, df in tasks.items():
            model_metrics = {}

            # Group by model_name and calculate metrics
            for model_name, data in df.groupby('model_name'):
                # Ensure numeric conversion for ground_truth and cleaned_answers
                data['ground_truth_num'] = pd.to_numeric(data['ground_truth'], errors='coerce')
                data['cleaned_answers_num'] = pd.to_numeric(data['cleaned_answers'], errors='coerce')
                data = data.dropna(subset=['ground_truth', 'cleaned_answers_num'])

                # Calculate MLAE
                data.loc[:, 'mlae'] = data.apply(
                    lambda row: np.log2(mean_absolute_error(
                        [row['ground_truth_num']], 
                        [row['cleaned_answers_num']]
                    ) + 0.125),
                    axis=1
                )

                # Aggregate metrics
                avg_mlae = data['mlae'].mean()
                std_mlae = data['mlae'].std()

                # Bootstrap confidence intervals
                try:
                    mlae_values = data['mlae'].dropna().values
                    bootstrap_result = bs.bootstrap(
                        np.array(mlae_values), 
                        stat_func=bs_stats.std
                    )
                    confidence_value = 1.96 * bootstrap_result.value
                except:
                    confidence_value = np.nan

                model_metrics[model_name] = {
                    'Dataset': task_name,
                    'Model': model_name,
                    'Average MLAE': round(avg_mlae, 2),
                    'Std MLAE': round(std_mlae, 2),
                    'Confidence Interval (95%)': round(confidence_value, 2)
                }

            if model_metrics:  # Only add if there are metrics
                if task_name not in metrics_summary:
                    metrics_summary[task_name] = []
                metrics_summary[task_name].extend(model_metrics.values())

    # Flatten metrics_summary into a single DataFrame
    metrics_table = pd.DataFrame([
        metric
        for dataset_metrics in metrics_summary.values()
        for metric in dataset_metrics
    ])

    # Sort the metrics table if not empty
    if not metrics_table.empty:
        metrics_table = metrics_table.sort_values(['Dataset', 'Average MLAE'])
    
    return metrics_table

def plot_results(metrics_table):
    """
    Plot the results from the metrics table with human benchmark values
    """
    summary_stats_by_task = {df_name: metrics_table[metrics_table['Dataset'] == df_name] 
                            for df_name in metrics_table['Dataset'].unique()}

   # Define display names for tasks
    task_display_names = {
        'volume': 'VOLUME',
        'area': 'AREA',
        'direction': 'DIRECTION',
        'length': 'LENGTH',
        'position_common_scale': 'POSITION COMMON SCALE',
        'position_non_aligned_scale': 'POSITION NON-ALIGNED SCALE',
        'angle': 'ANGLE',
        'curvature': 'CURVATURE',
        'shading': 'SHADING',
    }


    # Define colors for models and Human benchmarks
    model_colors = {
        'Finetuned Llama': '#8E44AD',   # Purple
        'Gemini1_5Flash': '#3498DB',    # Blue
        'GeminiProVision': '#E74C3C',   # Red
        'LLaMA': '#E67E22',             # Orange
        'gpt4o': '#27AE60',             # Green
        'Human': '#34495E'              # Dark Gray
    }

    # Define Human benchmark data
    human_data = {
        'angle': (3.22, 0.54),
        'area': (3.64, 0.38),
        'volume': (5.18, 0.40),
        'curvature': (4.13, 0.10),
        'shading': (4.22, 0.23),
        'position_common_scale': (3.35, 0.49),
        'position_non_aligned_scale': (3.06, 0.74),
        'length': (3.51, 0.44),
        'direction': (3.75, 0.39)
    }

    # Get task images
    base_path = '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/finetuning-EXP1numberone/images'
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
        ax_plot.set_xlim(-4, 20)
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

    """Helper function to average metrics across multiple runs"""
   
    all_metrics_df = pd.concat(metrics_list)
    
    averaged_metrics = all_metrics_df.groupby(['Dataset', 'Model']).agg({
        'Average MLAE': 'mean',
        'Std MLAE': 'mean',
        'Confidence Interval (95%)': 'mean'
    }).reset_index()
    
    for col in ['Average MLAE', 'Std MLAE', 'Confidence Interval (95%)']:
        averaged_metrics[col] = averaged_metrics[col].round(2)
    
    return averaged_metrics



def balance_datasets_exp1(df_volume, df_area, df_direction, df_length, df_position_common_scale,
                         df_position_non_aligned_scale, df_angle, df_curvature, df_shading):
    """
    Balance all datasets to ensure each model has the same number of samples.
    Returns a dictionary with balanced DataFrames, using the 'df_' prefix to match calculate_metrics parameters.
    """
    # Create dictionary of input DataFrames with the 'df_' prefix
    datasets = {
        'df_volume': df_volume,
        'df_area': df_area,
        'df_direction': df_direction,
        'df_length': df_length,
        'df_position_common_scale': df_position_common_scale,
        'df_position_non_aligned_scale': df_position_non_aligned_scale,
        'df_angle': df_angle,
        'df_curvature': df_curvature,
        'df_shading': df_shading
    }
    
    balanced_dfs = {}
    
    # Find minimum number of samples per model across all datasets
    min_samples = float('inf')
    for df_name, df in datasets.items():
        model_counts = df.groupby('model_name').size()
        min_samples = min(min_samples, model_counts.min())
    
    # Balance each dataset
    for df_name, df in datasets.items():
        balanced_df = pd.DataFrame()
        for model_name, group in df.groupby('model_name'):
            # Randomly sample the minimum number of rows for each model
            sampled = group.sample(n=min_samples, random_state=42)
            balanced_df = pd.concat([balanced_df, sampled])
        
        # Store the balanced DataFrame with the 'df_' prefix
        balanced_dfs[df_name] = balanced_df
    
    return balanced_dfs

def checkdeletedrows_forallcsv():
    """Process and check deleted rows across all CSV files"""
    file_paths = [
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/finetuning-EXP1numberfour/EXP-Results/EXP1results55images.csv',
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/finetuning-EXP1numberfour/EXP-Results/EXP1results55images.csv",
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/finetuning-EXP1numbersix/EXP-Results/EXP1results55images.csv'
    ]
    
    all_deleted_dfs = []
    all_balanced_metrics = []
    combined_balanced_dfs = []  # To store combined DataFrames across files

    for file_path in file_paths:
        # Get the original DataFrames
        df_volume, df_area, df_direction, df_length, df_position_common_scale, \
        df_position_non_aligned_scale, df_angle, df_curvature, df_shading, deleted_rows = clean_raw_answers(file_path)
        
        # Store original lengths in a dictionary for reference
        original_lengths = {
            'df_volume': len(df_volume),
            'df_area': len(df_area),
            'df_direction': len(df_direction),
            'df_length': len(df_length),
            'df_position_common_scale': len(df_position_common_scale),
            'df_position_non_aligned_scale': len(df_position_non_aligned_scale),
            'df_angle': len(df_angle),
            'df_curvature': len(df_curvature),
            'df_shading': len(df_shading)
        }
        
        # Balance the datasets
        balanced_dfs = balance_datasets_exp1(
            df_volume, df_area, df_direction, df_length, df_position_common_scale,
            df_position_non_aligned_scale, df_angle, df_curvature, df_shading
        )
        
        # Add a 'file' column to each balanced DataFrame for file identification
        for task_name, df in balanced_dfs.items():
            df['task'] = task_name  # Add task name
            df['file'] = file_path.split('/')[-1]  # Add file name
            combined_balanced_dfs.append(df)  # Collect for final concatenation

        # Print comparison using the stored original lengths
        for task_name, df in balanced_dfs.items():
            print(f"Task {task_name}: Original={original_lengths[task_name]} -> Balanced={len(df)}")
        print(f"Deleted rows: {len(deleted_rows)}")
        
        # Calculate metrics
        metrics_table = calculate_metrics(**balanced_dfs)
        all_balanced_metrics.append(metrics_table)

        # Only append to all_deleted_dfs if there are actually deleted rows
        if deleted_rows:
            deleted_df = pd.DataFrame(deleted_rows)[['raw_answer', 'model_name']]
            deleted_df['file'] = file_path.split('/')[-1]
            all_deleted_dfs.append(deleted_df)
    
    # Combine all deleted rows if there are any
    if all_deleted_dfs:
        combined_deleted_df = pd.concat(all_deleted_dfs, ignore_index=True)
    else:
        # Return an empty DataFrame with the expected columns if no deleted rows
        combined_deleted_df = pd.DataFrame(columns=['file', 'raw_answer', 'model_name'])

    # Combine all balanced DataFrames into one DataFrame
    combined_balanced_df = pd.concat(combined_balanced_dfs, ignore_index=True)

    # Return all results
    return combined_deleted_df[['file', 'raw_answer', 'model_name']], all_balanced_metrics, combined_balanced_df

def process_and_plot_multiplerun():
    """Process three EXP1 result files and create averaged plot"""
    file_paths = [
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/finetuning-EXP1numberfour/EXP-Results/EXP1results55images.csv",
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/finetuning-EXP1numberfive/EXP-Results/EXP1results55images.csv",
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/finetuning-EXP1numbersix/EXP-Results/EXP1results55images.csv"
    ]

    all_metrics = []
    for file_path in file_paths:
        df_volume, df_area, df_direction, df_length, df_position_common_scale, \
        df_position_non_aligned_scale, df_angle, df_curvature, df_shading, deleted_rows = clean_raw_answers(file_path)
        
        balanced_dfs = balance_datasets_exp1(
            df_volume, df_area, df_direction, df_length, df_position_common_scale,
            df_position_non_aligned_scale, df_angle, df_curvature, df_shading
        )
        
        metrics = calculate_metrics(**balanced_dfs)
        all_metrics.append(metrics)

    averaged_metrics_table = average_metrics(all_metrics)
    plot_results(averaged_metrics_table)

    return averaged_metrics_table
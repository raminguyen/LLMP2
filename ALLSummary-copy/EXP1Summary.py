

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
csv_file = '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/finetuning-EXP1numberone/EXP-Results/EXP1results.csv'

import pandas as pd
import numpy as np
import re

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

    # First check your data
    print("DataFrame columns:", df.columns)
    print("\nFirst few rows of raw_answer column:")

    def extract_digits_exp1(raw_text):
        """checkdeletedrows_forallcsv
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


def calculate_metrics(df_volume, df_area, df_direction, df_length, df_position_common_scale, df_position_non_aligned_scale, df_angle, df_curvature, df_shading):
    """
    Calculate metrics for each dataset and model.
    Parameters:
    df_volume, df_area, df_direction, df_length, df_position_common_scale, df_position_non_aligned_scale, df_angle, df_curvature, df_shading: DataFrames containing the results for each task
    Returns:
    pandas.DataFrame: Table of metrics for all models and datasets
    """
    # Dictionary to store metrics for each dataset
    metrics_summary = {}

    dataframes = {
        'volume': df_volume,
        'area' : df_area,
        'direction': df_direction,
        'length': df_length,
        'position_common_scale': df_position_common_scale,
        'position_non_aligned_scale': df_position_non_aligned_scale,
        'angle': df_angle,
        'curvature': df_curvature,
        'shading': df_shading
        }

    # Loop through each dataset
    for df_name, df in dataframes.items():
        model_metrics = {}
        
        for model_name, data in df.groupby('model_name'):
            data['ground_truth_num'] = pd.to_numeric(data['ground_truth'], errors='coerce')
            data['cleaned_answers_num'] = pd.to_numeric(data['cleaned_answers'], errors='coerce')
            data = data.dropna(subset=['ground_truth', 'cleaned_answers_num'])
            
            data.loc[:, 'mlae'] = data.apply(
                lambda row: np.log2(mean_absolute_error(
                    [row['ground_truth_num']], 
                    [row['cleaned_answers_num']]
                ) + 0.125),
                axis=1
            )
            
            avg_mlae = data['mlae'].mean()
            std_mlae = data['mlae'].std()
            
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
                'Dataset': df_name,
                'Model': model_name,
                'Average MLAE': round(avg_mlae, 2),
                'Std MLAE': round(std_mlae, 2),
                'Confidence Interval (95%)': round(confidence_value, 2)
            }
        
        if model_metrics:  # Only add if there are metrics
            metrics_summary[df_name] = model_metrics

    metrics_table = pd.DataFrame([
        metrics 
        for dataset_metrics in metrics_summary.values() 
        for metrics in dataset_metrics.values()
    ])
    
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

    # Desired custom order of tasks
    custom_task_order = [
        'position_common_scale', 'position_non_aligned_scale', 'length', 
        'direction', 'angle', 'area', 'volume', 'curvature', 'shading'
    ]

    # Reorder the summary_stats_by_task based on the custom order
    summary_stats_by_task = {task: summary_stats_by_task[task] for task in custom_task_order if task in summary_stats_by_task}

    # Define colors for models and Human benchmarks
    model_colors = {
        'CustomLLaMA': '#8E44AD',   # Purple
        'Gemini1_5Flash': '#3498DB',    # Blue
        'GeminiProVision': '#E74C3C',   # Red
        'LLaMA': '#E67E22',             # Orange
        'gpt4o': '#27AE60',             # Green
        'Human': '#34495E'              # Dark Gray
    }

    # Define human data points with MLAE and error bars
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


    fig, axes = plt.subplots(num_tasks, 3, figsize=(14, 1.5 * num_tasks), 
                        gridspec_kw={'width_ratios': [1, 4, 1]}, sharex=False)
    fig.subplots_adjust(hspace=0.3, left=0.05, right=0.98, top=0.98, bottom=0.02)


    fig.patch.set_facecolor('white')

    # Handle both single and multiple subplot cases
    if num_tasks == 1:
        axes = np.array([axes])  # Convert to 2D array with one row

    for i, (task_name, task_data) in enumerate(summary_stats_by_task.items()):

        ax_img, ax_plot, ax_label = axes[i]

        if i < len(summary_stats_by_task) - 1:
            ax_plot.xaxis.set_visible(False)
            ax_plot.spines['bottom'].set_visible(False)

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
    Automatically find images for each task in the given directory.
    Parameters:
    base_path (str): Base directory path where images are stored.
    task_types (list): List of task types (e.g., ['volume', 'area']). If None, will look for all 9 tasks.
    Returns:
    dict: Dictionary mapping task names to image paths.
    """
    task_images = {}
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    if task_types is None:
        # Default to all 9 tasks
        task_types = [
            "volume", "area", "direction", "length",
            "position_common_scale", "position_non_aligned_scale",
            "angle", "curvature", "shading"
        ]

    for task in task_types:
        task_pattern = f"{task}_"
        
        for file in os.listdir(base_path):
            if file.startswith(task_pattern) and any(file.lower().endswith(ext) for ext in image_extensions):
                task_images[task] = os.path.join(base_path, file)
                break

    return task_images


# Display all columns
pd.set_option('display.max_columns', None)

# Display all rows
pd.set_option('display.max_rows', None)

# Width of the display in characters
pd.set_option('display.width', None)

# Don't wrap long strings
pd.set_option('display.max_colwidth', None)



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

def process_and_plot_multiplerun(all_balanced_metrics):

     # Average metrics across all runs
    averaged_metrics_table = average_metrics(all_balanced_metrics)
    
    # Plot the results
    plot_results(averaged_metrics_table)
    
    return averaged_metrics_table
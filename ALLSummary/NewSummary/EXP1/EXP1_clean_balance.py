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

    #print("DataFrame columns:", df.columns)
    #print("\nFirst few rows of raw_answer column:")

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

    # Original names and their corresponding display names for the legend
    model_display_names = {
        'CustomLLaMA': 'Fine-tuned models',
        'Gemini1_5Flash': 'Gemini 1.5 Flash',
        'GeminiProVision': 'Gemini Pro Vision',
        'LLaMA': 'Llama 3.2 Vision',
        'gpt4o': 'GPT-4o Vision',
        'Human': 'Human'
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


    fig, axes = plt.subplots(num_tasks, 3, figsize=(8, 1.3 * num_tasks), 
                        gridspec_kw={'width_ratios': [1, 4, 1]}, sharex=False)



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
        ax_img.set_title(task_display_names.get(task_name, task_name), loc="left", fontsize=10, color="black")

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

            y_pos = len(sorted_model_names) + 0.5
            
            ax_plot.errorbar(human_value, y_pos, xerr=human_std, 
                           fmt='s', color=model_colors['Human'], 
                           capsize=5, capthick=1.5,
                           markersize=7, label='Human' if i == 0 else None)

        ax_plot.axvline(-5, color="black", linewidth=1)
        
        if task_name != "position_common_scale":
            ax_plot.axvline(20, color="black", linewidth=1)
        ax_plot.grid(False)

        for offset in np.linspace(-0.05, 0.05, 10):
            ax_plot.axvline(0 + offset, color="gray", alpha=0.1, linewidth=0.5)

        ax_plot.spines['top'].set_visible(False)
        ax_plot.spines['right'].set_visible(False)
        ax_plot.spines['left'].set_visible(False)
        ax_plot.spines['bottom'].set_position(('outward', 10))

        ax_plot.set_yticks(y_positions)
        ax_plot.set_yticklabels([])
        ax_plot.set_xlim(-5, 20)
        ax_plot.invert_yaxis()

        ax_label.set_yticks(y_positions)
        ax_label.set_yticklabels(sorted_model_names, fontsize=10)
        ax_label.tick_params(left=False, right=False, labelleft=False, labelright=True)
        ax_plot.tick_params(axis='y', which='both', left=False, right=False)
        ax_label.set_ylim(ax_plot.get_ylim())
        ax_label.axis("off")

    # Add the legend only if there are tasks
    if num_tasks > 0:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=model_colors[model_name], 
                       markersize=10, label=model_display_names[model_name])
            for model_name in model_colors.keys()
        ]
        axes[0, 1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.5, 0.5), frameon=False)


    plt.savefig("Figure2.png", bbox_inches='tight', dpi=300)
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

"""
def balance_datasets_exp1_all_files(file_datasets):
    
    Balance all datasets across multiple files so that each task DataFrame is exactly 812 rows per file.
    
    Parameters:
        file_datasets (list of dict): A list where each element is a dictionary of task DataFrames 
                                      from an individual file.

    Returns:
        list of dict: A list of dictionaries containing balanced DataFrames for each file.
    
    target_size = 812  # Ensure each task has exactly 812 rows

    balanced_file_datasets = []
    
    for file_data in file_datasets:
        balanced_dfs = {}
        for df_name, df in file_data.items():
            balanced_dfs[df_name] = df.sample(
                n=target_size,
                replace=len(df) < target_size,  # Oversample if dataset is too small
                random_state=42
            )
        
        balanced_file_datasets.append(balanced_dfs)

    return balanced_file_datasets

"""

def clean_and_balance_csv_files():
    """
    Process, balance, and check deleted rows across all CSV files, ensuring exactly 812 rows per task per file.
    """
    all_deleted_dfs = []
    all_balanced_metrics = []
    balanced_dataframes = []  

    file_paths = [
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/finetuning-EXP1numberfour/EXP-Results/EXP1results55images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/finetuning-EXP1numberfive/EXP-Results/EXP1results55images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/finetuning-EXP1numbersix/EXP-Results/EXP1results55images.csv'
    ]

    target_size = 812  # Ensure each task has exactly 812 rows
    file_datasets = []  # Store datasets before balancing

    for file_path in file_paths:
        file_name = os.path.basename(file_path)

        # Get the cleaned DataFrames for each task and the deleted rows
        (df_volume, df_area, df_direction, df_length, df_position_common_scale, 
         df_position_non_aligned_scale, df_angle, df_curvature, df_shading, deleted_rows) = clean_raw_answers(file_path)

        # Debug: Print sizes after cleaning
        print(f"\n🔍 Processing File: {file_name}")
        print("\nTask sizes after cleaning:")
        for task_name, df in {
            'volume': df_volume, 'area': df_area, 'direction': df_direction, 'length': df_length,
            'position_common_scale': df_position_common_scale, 'position_non_aligned_scale': df_position_non_aligned_scale,
            'angle': df_angle, 'curvature': df_curvature, 'shading': df_shading
        }.items():
            print(f"{task_name}: {len(df)} rows")

        # Store datasets before balancing
        file_datasets.append({
            'df_volume': df_volume,
            'df_area': df_area,
            'df_direction': df_direction,
            'df_length': df_length,
            'df_position_common_scale': df_position_common_scale,
            'df_position_non_aligned_scale': df_position_non_aligned_scale,
            'df_angle': df_angle,
            'df_curvature': df_curvature,
            'df_shading': df_shading,
            'file_name': file_name,  # Track the file name
            'deleted_rows': deleted_rows  # Store deleted rows
        })

    # Step 1: Balance datasets across all files
    balanced_file_datasets = []
    
    for file_data in file_datasets:
        file_name = file_data.pop('file_name')  # Extract filename separately
        deleted_rows = file_data.pop('deleted_rows')  # Extract deleted rows separately
        balanced_dfs = {}

        for df_name, df in file_data.items():
            balanced_dfs[df_name] = df.sample(
                n=target_size,
                replace=len(df) < target_size,  # Oversample if dataset is too small
                random_state=42
            )

        balanced_file_datasets.append((file_name, balanced_dfs, deleted_rows))

    # Step 2: Process each file after balancing
    for file_name, balanced_dfs, deleted_rows in balanced_file_datasets:
        print(f"\n📊 Task sizes after final balancing for {file_name}:")
        for df_name, df in balanced_dfs.items():
            print(f"{df_name}: {len(df)} rows")

        # Add file and task columns
        for task_name, df in balanced_dfs.items():
            df = df.copy()
            df['task'] = task_name
            df['file'] = file_name
            balanced_dataframes.append(df)

        # Calculate metrics for this file
        metrics_table, _ = calculate_metrics(**balanced_dfs)  
        all_balanced_metrics.append(metrics_table)

        # Handle deleted rows safely
        if deleted_rows:
            deleted_df = pd.DataFrame(deleted_rows) if isinstance(deleted_rows, list) else deleted_rows.copy()
            if 'raw_answer' in deleted_df.columns and 'model_name' in deleted_df.columns:
                deleted_df = deleted_df[['raw_answer', 'model_name']]
            deleted_df['file'] = file_name
            all_deleted_dfs.append(deleted_df)

        # Save each balanced file separately
        output_file = f"final_{file_name}"
        combined_file_df = pd.concat(balanced_dfs.values(), ignore_index=True)
        combined_file_df.to_csv(output_file, index=False)
        print(f"✅ Saved {output_file} with {len(combined_file_df)} rows.")

        # Save deleted rows per file
        if not deleted_df.empty:
            deleted_output_file = f"deleted_{file_name}"
            deleted_df.to_csv(deleted_output_file, index=False)
            print(f"🗑️ Saved deleted rows to {deleted_output_file}")

    # Combine all balanced DataFrames for reference
    combined_balanced_df = pd.concat(balanced_dataframes, ignore_index=True)
    combined_balanced_df.to_csv("finalEXP1.csv", index=False)

    # Combine all deleted rows into a single DataFrame
    combined_deleted_df = pd.concat(all_deleted_dfs, ignore_index=True) if all_deleted_dfs else pd.DataFrame(columns=['file', 'raw_answer', 'model_name'])

    # Ensure all_balanced_metrics is returned as a DataFrame
    all_balanced_metrics_df = pd.concat(all_balanced_metrics, ignore_index=True) if all_balanced_metrics else pd.DataFrame()

    return combined_balanced_df, combined_deleted_df, all_balanced_metrics_df




def process_and_plot_multiplerun(all_balanced_metrics):

     # Average metrics across all runs
    averaged_metrics_table = average_metrics(all_balanced_metrics)
    
    # Plot the results
    plot_results(averaged_metrics_table)
    
    return averaged_metrics_table


def check_combined_unique_answers(balanced_dataframes):
    """
    Combine specific task DataFrames and check unique values and counts for each task.

    Parameters:
    balanced_dataframes (dict): Dictionary of task-specific DataFrames.

    Returns:
    dict: Dictionary summarizing unique value counts for each task.
    """

    # Task-specific dataset names
    task_datasets = {
        'shading': 'df_shading',
        'curvature': 'df_curvature',
        'angle': 'df_angle',
        'position_non_aligned_scale': 'df_position_non_aligned_scale',
        'position_common_scale': 'df_position_common_scale',
        'length': 'df_length',
        'direction': 'df_direction',
        'area': 'df_area',
        'volume': 'df_volume'
    }

    # Initialize summary dictionary
    unique_values_summary = {}

    # Get all unique tasks dynamically from 'task_name'
    unique_tasks = balanced_dataframes['task_name'].unique()

    # Process each task
    for task in unique_tasks:
        # Filter DataFrame for the current task
        task_df = balanced_dataframes[balanced_dataframes['task_name'] == task]

        if 'cleaned_answers' in task_df.columns:
            # Flatten cleaned_answers column
            cleaned_answers = task_df['cleaned_answers'].dropna()
            value_counts = cleaned_answers.value_counts().sort_index()
            unique_count = value_counts.size

            # Store results
            unique_values_summary[task] = {
                'unique_count': unique_count,
                'sorted_values': value_counts
            }
            print(f"\nUnique answers in {task.replace('_', ' ').title()} task (sorted):")
            print(value_counts)
        else:
            print(f"Warning: 'cleaned_answers' column not found in {task} task.")
            unique_values_summary[task] = {'unique_count': 0, 'sorted_values': None}

def visualize_cleaned_answers_overlay(balanced_dataframes):
    """
    Visualize scatter plots for unique value counts in cleaned answers for all tasks,
    separated by Pretrained and Finetuned models, with consistent colors for models only.
    Each task includes vertical lines based on specific limits.
    """
    # Ensure balanced_dataframes is a single DataFrame
    if isinstance(balanced_dataframes, pd.DataFrame):
        combined_all = balanced_dataframes
    elif isinstance(balanced_dataframes, list):
        combined_all = pd.concat(balanced_dataframes, ignore_index=True)
    else:
        raise TypeError("balanced_dataframes must be a DataFrame or a list of DataFrames.")

    # Define pretrained and finetuned models
    pretrained_models = {'Gemini1_5Flash', 'GeminiProVision', 'LLaMA', 'gpt4o'}
    finetuned_models = {'CustomLLaMA'}

    # Add a 'model_type' column
    combined_all['model_type'] = combined_all['model_name'].apply(
        lambda x: 'Pretrained' if x in pretrained_models else 'Finetuned' if x in finetuned_models else 'Unknown'
    )

    # Task-specific limits for x-lines
    task_limits = {
        "position_common_scale": 60,
        "position_non_aligned_scale": 60,
        "length": 100,
        "direction": 359,
        "angle": 90,
        "area": 5026.55,
        "volume": 8000,
        "curvature": 0.088,
        "shading": 100
    }

    # Create mapping for capitalized task names
    task_name_mapping = {
        "position_common_scale": "POSITION COMMON SCALE",
        "position_non_aligned_scale": "POSITION NON-ALIGNED SCALE",
        "length": "LENGTH",
        "direction": "DIRECTION",
        "angle": "ANGLE",
        "area": "AREA",
        "volume": "VOLUME",
        "curvature": "CURVATURE",
        "shading": "SHADING"
    }

    # Define unique tasks to process
    task_names = combined_all['task_name'].unique()

    # Colors for Pretrained and Finetuned models
    model_colors = {'Pretrained': '#3A50A1', 'Finetuned': '#4C8C2B'}

    # Define marker styles
    marker_styles = {
        'Pretrained': ('o', 50),  # Circle, size 50
        'Finetuned': ('^', 50)  # Triangle, size 50
    }

    # Create 3x3 subplots
    num_tasks = len(task_names)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    # Iterate over tasks and create scatter plots
    for i, task in enumerate(task_names):
        ax = axes[i]
        
        # Filter for the current task
        task_df = combined_all[combined_all['task_name'] == task]

        # Plot Pretrained and Finetuned models
        for model_type, color in model_colors.items():
            filtered_df = task_df[task_df['model_type'] == model_type]

            if not filtered_df.empty:
                # Count unique values
                cleaned_answers = filtered_df['cleaned_answers'].dropna()
                value_counts = cleaned_answers.value_counts().sort_index()

                # Plot scatter with marker style
                ax.scatter(value_counts.index, value_counts.values,
                        color=color, label=model_type,
                        s=marker_styles[model_type][1],  # size
                        marker=marker_styles[model_type][0],  # marker shape
                        alpha=0.8)
                

        # Add vertical x-line for task-specific limit
        if task in task_limits:
            limit_value = task_limits[task]
            ax.axvline(x=limit_value, color="red", linestyle="--", linewidth=1)
            ax.text(limit_value + (limit_value * 0.01), ax.get_ylim()[1] * 0.95, 
                    f"Limit: {limit_value}", color="red", fontsize=14, va='top')

        # Fix x-axis formatting for Volume
        if task == 'volume':
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

        # Set title and labels
        ax.set_xlabel("Unique Values")
        ax.set_title(task_name_mapping.get(task, task), loc="center", fontsize=14, color="black")
        ax.set_ylabel("Counts")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title="Model Type", fontsize=13)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add overall title and adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('analysisexp1.png', 
    bbox_inches='tight', 
    format='png',
    dpi=300,
    transparent=False)
    plt.show()

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', None)
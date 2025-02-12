

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

    # Human benchmark values
    human_values = {
        'type1': (1.4, 0.14),
        'type2': (1.72, 0.2),
        'type3': (1.84, 0.16),
        'type4': (2.35, 0.175),
        'type5': (2.72, 0.155),
    }

    human_values2 = {
        'type1': (1.25, 0.25),
        'type2': (1.39, 0.175),
        'type3': (1.56, 0.25),
        'type4': (1.96, 0.23),
        'type5': (2.24, 0.24),
    }

    # Original names and their corresponding display names for the legend
    model_display_names = {
        'CustomLLaMA': 'Fine-tuned models',
        'Gemini1_5Flash': 'Gemini 1.5 Flash',
        'GeminiProVision': 'Gemini Pro Vision',
        'LLaMA': 'Llama 3.2 Vision',
        'gpt4o': 'GPT-4o Vision',
        'Human': 'Human'
    }

    # Get task images
    base_path = '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP3/finetuning-EXP3numberone/images'
    task_images = find_task_images(base_path)
    
    num_tasks = len(summary_stats_by_task)
    fig, axes = plt.subplots(num_tasks, 3, figsize=(10, 2 * num_tasks), 
                             gridspec_kw={'width_ratios': [1, 4, 1]}, sharex=False)
    
    fig.subplots_adjust(hspace=0.6, left=0.1, right=0.75, top=0.90, bottom=0.05)
    fig.patch.set_facecolor('white')

    if num_tasks == 1:
        axes = np.array([axes])  # Convert to 2D array with one row

    for i, (task_name, task_data) in enumerate(summary_stats_by_task.items()):
        ax_img, ax_plot, ax_label = axes[i]

        # Plot task image
        if task_name in task_images:
            img_path = task_images[task_name]
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("L")
                img_inverted = ImageOps.invert(img)
                img_with_border = ImageOps.expand(img_inverted.convert("RGB"), border=1, fill="black")
                ax_img.imshow(img_with_border)
                ax_img.set_facecolor("white")
            else:
                ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=16, color="black")
                ax_img.set_facecolor("white")

        ax_img.axis('off')
        ax_img.set_title(task_display_names.get(task_name, task_name), loc="left", fontsize=16, color="black")

        # Plot model MLAE values
        sorted_model_names = sorted(task_data['Model'].unique())
        y_positions = np.arange(len(sorted_model_names))

        for j, model_name in enumerate(sorted_model_names):
            model_data = task_data[task_data['Model'] == model_name]
            mlae_value = model_data['Average MLAE'].values[0]
            confidence_interval = model_data['Confidence Interval (95%)'].values[0]

            ax_plot.errorbar(mlae_value, j, xerr=confidence_interval, fmt='o', 
                            color=model_colors.get(model_name, 'gray'), capsize=5, 
                            label=f"{model_name}" if i == 0 else None)

        # Plot human values
        for label, human_data, y_offset in zip(["ClMcG", "HeerBos"], [human_values, human_values2], [0.1, 2]):
            if task_name in human_data:
                human_value, human_std = human_data[task_name]
                y_pos = len(sorted_model_names) + y_offset  # Position human marker

                # Plot human error bar with marker
                ax_plot.errorbar(human_value, y_pos, xerr=human_std, 
                                fmt='s', color=model_colors.get('Human', 'black'), 
                                capsize=5, capthick=1.5, markersize=7, label=None)

                # Add human benchmark label only on the bottom subplot
                if i == len(summary_stats_by_task) - 1:
                    ax_plot.text(human_value - 0.5, y_pos, label, fontsize=10, color='black', 
                                ha='right', va='center')

        if task_name != "type1":
            ax_plot.axvline(6, color="black", linewidth=2)

        ax_plot.axvline(-4, color="black", linewidth=2)

        ax_plot.set_xlim(-4, 6)
        

        ax_plot.grid(False)

        for offset in np.linspace(-0.05, 0.05, 10):
            ax_plot.axvline(0 + offset, color="gray", alpha=0.1, linewidth=0.5)

        if i < len(summary_stats_by_task) - 1:
            ax_plot.xaxis.set_visible(False)
            ax_plot.spines['bottom'].set_visible(False)

        ax_plot.spines['top'].set_visible(False)
        ax_plot.spines['right'].set_visible(False)
        ax_plot.spines['left'].set_visible(False)
        ax_plot.spines['bottom'].set_position(('outward', 10))

        ax_plot.set_yticks(y_positions)
        ax_plot.set_yticklabels([])
        ax_plot.invert_yaxis()

        ax_label.set_yticks(y_positions)
        ax_label.set_yticklabels(sorted_model_names, fontsize=12)
        ax_label.tick_params(left=False, right=False, labelleft=False, labelright=True)
        ax_plot.tick_params(axis='y', which='both', left=False, right=False)
        ax_plot.tick_params(axis='x', labelsize=14)
        ax_label.set_ylim(ax_plot.get_ylim())
        ax_label.axis("off")

    if num_tasks > 0:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=model_colors[model_name], 
                       markersize=10, label=model_display_names[model_name])
            for model_name in model_colors.keys()
        ]
        axes[0, 1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.6, 0.5), frameon=False)

    plt.savefig("Figure4.png", bbox_inches='tight', dpi=300)
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
    balanced_dataframes = []  
    
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

        # Store the balanced DataFrames for later use
        balanced_dataframes.append((df_type1, df_type2, df_type3, df_type4, df_type5))

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
    
    combined_deleted_df = pd.concat(all_deleted_dfs, ignore_index=True) if all_deleted_dfs else pd.DataFrame(columns=['file', 'raw_answer', 'model_name'])

   
    return combined_metrics, combined_deleted_df, balanced_dataframes


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


"""Check unique answers after balancing number of datasets"""

# Define global color settings
GLOBAL_COLORS = {"Pretrained": "blue", "Finetuned": "purple"}

# Apply the global color palette to Seaborn
sns.set_palette([GLOBAL_COLORS["Pretrained"], GLOBAL_COLORS["Finetuned"]])


"""" Only check for unique answer """

def check_combined_unique_answers(balanced_dataframes):
    
    colors = GLOBAL_COLORS
    """
    Visualize unique value counts across Pretrained and Finetuned models using Matplotlib,
    add threshold lines for x=1 and y=200, and label points with counts > 200 ensuring no overlap.

    Parameters:
    balanced_dataframes (list): List of tuples containing DataFrames (df_type1, df_type2, ..., df_type5) for each dataset.

    Returns:
    None
    """
    # Combine all DataFrames from df_type1 to df_type5
    combined_all = pd.concat(
        [pd.concat([df_type1, df_type2, df_type3, df_type4, df_type5], ignore_index=True)
         for (df_type1, df_type2, df_type3, df_type4, df_type5) in balanced_dataframes],
        ignore_index=True
    )

    # Define pretrained and finetuned model groups
    pretrained_models = {'Gemini1_5Flash', 'GeminiProVision', 'LLaMA', 'gpt4o'}
    finetuned_models = {'CustomLLaMA'}

    # Categorize models
    combined_all['model_type'] = combined_all['model_name'].apply(
        lambda x: 'Pretrained' if x in pretrained_models else 'Finetuned' if x in finetuned_models else 'Unknown'
    )

    # Initialize Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 4))
    model_types = ['Pretrained', 'Finetuned']
    
    colors = {
        'Pretrained': '#3A50A1',  # Dark Blue for Pretrained
        'Finetuned': '#4C8C2B'   # Dark Green for Finetuned
    }

    # Define marker styles
    marker_styles = {
        'Pretrained': ('o', 50),  # Circle marker with size 50 for Pretrained
        'Finetuned': ('^', 50)    # Triangle marker with size 50 for Finetuned
    }

    # Store labeled positions to adjust dynamically
    labeled_positions = set()

    for model_type in ['Pretrained', 'Finetuned']:
        group = combined_all[combined_all['model_type'] == model_type]
        flattened_cleaned_answers = group['cleaned_answers'].explode()
        flattened_cleaned_answers = flattened_cleaned_answers.dropna().apply(
            lambda x: x if isinstance(x, (int, float)) else x[0] if isinstance(x, list) else None
        ).dropna()

        # Count unique values and sort
        value_counts = flattened_cleaned_answers.value_counts().sort_index()

        # Plot with marker styles
        ax.scatter(value_counts.index, value_counts.values, 
                label=f"{model_type} Models", color=colors[model_type], 
                s=marker_styles[model_type][1], marker=marker_styles[model_type][0])

        # Add labels for points with counts > 200
        #over_200 = value_counts[value_counts > 200]
        #for idx, val in over_200.items():
            # Check and shift label position to avoid overlap
            #offset = 12 if (idx, val) in labeled_positions else 5
            #ax.annotate(f"{idx}", (idx, val), textcoords="offset points", xytext=(0, offset),
                        #ha='center', fontsize=8, color='red')
            #labeled_positions.add((idx, val))

    # Add threshold lines
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1, label='Min of prediction = 1)')
    ax.axhline(y=200, color='red', linestyle='--', linewidth=1, label='Min of prediction = 200')

    # Add labels, title, and grid
    ax.set_xlabel("Unique Values", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_xlim(0, 2.5)  # Set x-axis range
    ax.legend()

    plt.savefig('analysisexp3.png', 
    bbox_inches='tight', 
    format='png',
    transparent=False)

    plt.show()
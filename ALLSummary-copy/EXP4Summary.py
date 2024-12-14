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
    pandas.DataFrame: DataFrame with raw and cleaned answers
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    print("DataFrame columns:", df.columns)
    for task in df['task_name'].unique():
        number_unique_images = df[df['task_name'] == task]['image_path'].nunique()
        print(f"Task {task}: {number_unique_images} unique images")

    print("Rows before cleaning:")
    print(f"  📥 Total rows: {len(df)}")

    deleted_rows = []
    total_tasks = len(df)
    
    def extract_digits_exp4(raw_text, model_name):
        """
        Extraction logic specific to EXP4.
        
        Parameters:
        raw_text (str): Raw text to process
        model_name (str): Name of the model (e.g., "LLaMA")
        
        Returns:
        list: List of extracted float values, or None if no valid values found
        """
        if pd.isna(raw_text):
            return None

        # Clean and preprocess the text
        raw_text = str(raw_text).strip()
        
        # Split on 'assistant\n\n' to get the actual response if present
        if 'assistant\n\n' in raw_text:
            raw_text = raw_text.split('assistant\n\n')[1]
        
        # Remove CSV markers and model names for cleaner text
        raw_text = re.sub(r'EXP4results10images\.csv.*?LLaMA\d*', '', raw_text)
        raw_text = re.sub(r'LLaMA\d*$', '', raw_text)
        raw_text = raw_text.replace('\n', ' ')

        # Split into sentences
        sentences = re.split(r'[.!?]\s+', raw_text)
        last_sentence = sentences[-1] if sentences else ""

        # Look for square-bracketed list first
        match = re.search(r'\[([\d.,\s]+)\]', last_sentence)
        if match:
            try:
                return [float(num.strip()) for num in match.group(1).split(',')]
            except ValueError:
                pass

        # Handle "LLaMA" specific case
        if model_name == "LLaMA":
            digit_matches = re.findall(r'\b(\d+)\b', last_sentence)
            if len(digit_matches) >= 2:
                try:
                    return [float(digit_matches[-2]), float(digit_matches[-1])]
                except ValueError:
                    pass

        # Handle general case of two numbers in various formats
        patterns = [
            # Handle descriptive responses with 'approximately'
            r'approximately\s*(\d+)\s*pixels?\s*(?:and|,)?\s*(\d+)\s*pixels?',
            r'approximately\s*(\d+)\s*and\s*(\d+)\s*pixels',
            
            # Handle step-based responses
            r'Step \d+:.*?(\d+)\s*pixels.*?Step \d+:.*?(\d+)\s*pixels',
            r'Step \d+:.*?(\d+)\s*pixels.*?same.*?(\d+)\s*pixels',
            
            # Handle square bracket format
            r'\[\s*(\d+)\s*,\s*(\d+)\s*\]',
            
            # Handle cases with 'respectively'
            r'(\d+)\s*pixels?\s*and\s*(\d+)\s*pixels?\s*,?\s*respectively',
            
            # Handle same length cases
            r'same length.*?(\d+)\s*pixels',
            r'(\d+)\s*pixels.*?same length',
            r'both.*?same.*?(\d+)\s*pixels',
            r'(\d+)\s*pixels.*?same as',
            r'same as.*?(\d+)\s*pixels',
            
            # Handle original patterns
            r'(\d+)\s*(?:and|,)\s*(\d+)\s*pixels\s*long',
            r'(?:first|top).*?(\d+).*?(?:second|bottom).*?(\d+)',
            r'(?:horizontal|vertical).*?(\d+).*?(?:horizontal|vertical).*?(\d+)',
            r'(?:longer|top).*?(\d+).*?(?:shorter|bottom).*?(\d+)',
            r'between\s*(\d+)\s*and\s*(\d+)',
            r'(\d+)\s*(?:and|,)\s*(\d+)',
            r'both.*?(\d+)\s*pixels',
            r'(?:is|are)\s*(\d+)\s*pixels\s*long.*?(?:is|are)\s*(\d+)\s*pixels\s*long',
            r'bar\s+is\s+(?:the\s+)?(\d+)\s+pixels.*?bar\s+is\s+(?:the\s+)?(\d+)\s+pixels',
        ]
                    
        for pattern in patterns:
            match = re.search(pattern, last_sentence, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    if len(match.groups()) >= 2:
                        return [float(match.group(1)), float(match.group(2))]
                    else:
                        value = float(match.group(1))
                        return [value, value]  # Return same value twice for same-length cases
                except ValueError:
                    continue

        return None
        

    # Example loop to clean the data
    rows_to_delete = []

    for index, row in df.iterrows():
        model_name = row.get('model', '')  # Assuming model name is in a 'model' column
        raw_text = row.get('raw_answer', '')  # Assuming raw answers are in a 'raw_answer' column

        # Extract digits
        cleaned_answer = extract_digits_exp4(raw_text, model_name)

        # Check if answer is valid, otherwise mark for deletion
        if cleaned_answer is None:
            deleted_rows.append(row)
            rows_to_delete.append(index)

    # Drop rows marked for deletion
    df.drop(rows_to_delete, inplace=True)

    # Print deleted rows and total tasks
    print(f"Total tasks: {total_tasks}")
    print(f"Rows deleted: {len(deleted_rows)}")
    print("Deleted rows:")
    
    deleted_rows
    
    # Process the dataframe
    df['cleaned_answers'] = df.apply(
        lambda row: extract_digits_exp4(row['raw_answer'], row['model_name']), 
        axis=1
    )
    
    # Split the dataframe by task
    df_framed = df[df['task_name'] == 'framed'].copy()
    df_unframed = df[df['task_name'] == 'unframed'].copy()
    
    return df_framed, df_unframed, deleted_rows

def calculate_metrics(df_framed, df_unframed):
    """
    Calculate metrics for each dataset and model.
    Parameters:
    df_framed, df_unframed: DataFrames containing the results for each task
    Returns:
    pandas.DataFrame: Table of metrics for all models and datasets
    """
    # Dictionary to store metrics for each dataset
    metrics_summary = {}

    # List of DataFrames to process
    dataframes = {
        'framed': df_framed, 
        'unframed': df_unframed
    }

    # Loop through each dataset
    for df_name, df in dataframes.items():
        model_metrics = {}
        
        for model_name, group in df.groupby('model_name'):
            # Create a copy of the group to avoid SettingWithCopyWarning
            data = group.copy()
            
            # Convert string lists to actual lists first
            data['ground_truth_num'] = data['ground_truth'].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)
            data['cleaned_answers_num'] = data['cleaned_answers'].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)
            
            # Drop any rows where conversion failed
            data = data.dropna(subset=['ground_truth_num', 'cleaned_answers_num'])
            
            if len(data) == 0:
                continue

            # Calculate both MAE and MLAE
            data['mae'] = data.apply(
                lambda row: mean_absolute_error(
                    row['ground_truth_num'], 
                    row['cleaned_answers_num']
                ),
                axis=1
            )
            
            data['mlae'] = data.apply(
                lambda row: np.log2(mean_absolute_error(
                    row['ground_truth_num'], 
                    row['cleaned_answers_num']
                ) + 0.125),
                axis=1
            )
            
            avg_mlae = data['mlae'].mean()
            std_mlae = data['mlae'].std()
            avg_mae = data['mae'].mean()
            
            try:
                mlae_values = data['mlae'].dropna().values
                if len(mlae_values) > 1:  # Ensure we have enough values for bootstrap
                    bootstrap_result = bs.bootstrap(
                        np.array(mlae_values), 
                        stat_func=bs_stats.std
                    )
                    confidence_value = 1.96 * bootstrap_result.value
                else:
                    confidence_value = np.nan
            except ValueError as e:
                print(f"Bootstrap failed for {model_name} in {df_name}: {str(e)}")
                confidence_value = np.nan
            except Exception as e:
                print(f"Unexpected error in bootstrap for {model_name} in {df_name}: {str(e)}")
                confidence_value = np.nan
            
            model_metrics[model_name] = {
                'Dataset': df_name,
                'Model': model_name,
                'Average MLAE': round(avg_mlae, 2),
                'Average MAE': round(avg_mae, 2),
                'Std MLAE': round(std_mlae, 2),
                'Confidence Interval (95%)': round(confidence_value, 2)
            }

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
        'framed': 'Framed Rectangles',
        'unframed': 'Bars',
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
        'framed': (3.371, 0.741),
        'unframed': (3.961, 0.454)
    }
    

    # Get task images
    base_path = '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP4/finetuning-EXP4numberone/images'
    task_images = find_task_images(base_path)
    
    num_tasks = len(summary_stats_by_task)
    fig, axes = plt.subplots(num_tasks, 3, figsize=(12, 3 * num_tasks), 
                            gridspec_kw={'width_ratios': [1, 4, 1]}, sharex=False)
    fig.subplots_adjust(hspace=0.8, left=0.05, right=0.95)
    fig.patch.set_facecolor('white')

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

        # Add reference lines
        ax_plot.axvline(-1, color="black", linewidth=1)
        ax_plot.axvline(6, color="black", linewidth=1)
        ax_plot.grid(False)

        # Add blurred line at 0
        for offset in np.linspace(-0.05, 0.05, 10):
            ax_plot.axvline(0 + offset, color="gray", alpha=0.1, linewidth=0.5)

        # Customize plot appearance
        ax_plot.spines['top'].set_visible(False)
        ax_plot.spines['right'].set_visible(False)
        ax_plot.spines['left'].set_visible(False)
        ax_plot.spines['bottom'].set_position(('outward', 10))

        ax_plot.set_yticks(y_positions)
        ax_plot.set_yticklabels([])
        ax_plot.set_xlim(-1, 6)
        ax_plot.invert_yaxis()

        # Display model names
        ax_label.set_yticks(y_positions)
        ax_label.set_yticklabels(sorted_model_names, fontsize=10)
        ax_label.tick_params(left=False, right=False, labelleft=False, labelright=True)
        ax_plot.tick_params(axis='y', which='both', left=False, right=False)
        ax_label.set_ylim(ax_plot.get_ylim())
        ax_label.axis("off")

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
        task_types = ['unframed', 'framed']

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

""" MAIN FUNCTION """

def process_and_plot_multiplerun():
    """Process three EXP4 result files and create averaged plot"""
    # Define file paths
    file_paths = [
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP4/finetuning-EXP4numberone/EXP-Results/EXP4evaluate55images.csv",
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP4/finetuning-EXP4numbertwo/EXP-Results/EXP4evaluate55images.csv",
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP4/finetuning-EXP4numberthree/EXP-Results/EXP4evaluate55images.csv"
    ]

    # Calculate metrics for each file and store them
    all_metrics = []
    for i, file_path in enumerate(file_paths, 1):
        print(f"\nProcessing File {i}: {os.path.basename(file_path)}")
        
        df_framed, df_unframed, deleted_rows = clean_raw_answers(file_path)
        
        print(f"\nNumber of rows in each task:")
        print(f"Task framed rows: {len(df_framed)}")
        print(f"Task unframed rows: {len(df_unframed)}")

        # Calculate metrics
        metrics = calculate_metrics(df_framed, df_unframed)
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


""" Check deleted rows for each csv file"""

def balance_datasets(df_framed, df_unframed, target_size=784):
    
    # Ensure the datasets are balanced to the target size
    df_framed_balanced = df_framed.sample(n=target_size, replace=True, random_state=42) \
        if len(df_framed) < target_size else df_framed.sample(n=target_size, random_state=42)
    
    df_unframed_balanced = df_unframed.sample(n=target_size, replace=True, random_state=42) \
        if len(df_unframed) < target_size else df_unframed.sample(n=target_size, random_state=42)
    
    print(f"After balancing: Framed={len(df_framed_balanced)}, Unframed={len(df_unframed_balanced)}")
    
    return df_framed_balanced, df_unframed_balanced


def checkdeletedrows_forallcsv():
    file_paths = [
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP4/finetuning-EXP4numberone/EXP-Results/EXP4evaluate55images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP4/finetuning-EXP4numbertwo/EXP-Results/EXP4evaluate55images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP4/finetuning-EXP4numberthree/EXP-Results/EXP4evaluate55images.csv'
    ]
    
    all_balanced_metrics = []

    all_deleted_dfs = []
    
    for file_path in file_paths:
        df_framed, df_unframed, deleted_rows = clean_raw_answers(file_path)
                
        print(f"\nNumber of rows in each task:")
        print(f"Task framed rows: {len(df_framed)}")
        print(f"Task unframed rows: {len(df_unframed)}")


        # Balance datasets
        df_framed, df_unframed = balance_datasets(df_framed, df_unframed)

        metrics_table = calculate_metrics(df_framed, df_unframed)
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
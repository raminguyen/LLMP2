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
import ast


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
        'framed': 'FRAMED RECTANGLES',
        'unframed': 'BARS',
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
    fig, axes = plt.subplots(num_tasks, 3, figsize=(10, 2 * num_tasks), 
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
            y_pos = len(sorted_model_names) + 0.5
            
            ax_plot.errorbar(human_value, y_pos, xerr=human_std, 
                           fmt='s', color=model_colors['Human'], 
                           capsize=5, capthick=1.5,
                           markersize=7, label='Human' if i == 0 else None)

        # Add reference lines
        ax_plot.axvline(-1, color="black", linewidth=2)
        ax_plot.axvline(5, color="black", linewidth=2)
        ax_plot.grid(False)

        # Add blurred line at 0
        for offset in np.linspace(-0.05, 0.05, 10):
            ax_plot.axvline(0 + offset, color="gray", alpha=0.1, linewidth=0.5)

        if i < len(summary_stats_by_task) - 1:
            ax_plot.xaxis.set_visible(False)
            ax_plot.spines['bottom'].set_visible(False)

        # Customize plot appearance
        ax_plot.spines['top'].set_visible(False)
        ax_plot.spines['right'].set_visible(False)
        ax_plot.spines['left'].set_visible(False)
        ax_plot.spines['bottom'].set_position(('outward', 10))

        ax_plot.set_yticks(y_positions)
        ax_plot.set_yticklabels([])
        ax_plot.set_xlim(-1, 5)
        ax_plot.invert_yaxis()

        # Display model names
        ax_label.set_yticks(y_positions)
        ax_label.set_yticklabels(sorted_model_names, fontsize=10)
        ax_label.tick_params(left=False, right=False, labelleft=False, labelright=True)
        ax_plot.tick_params(axis='y', which='both', left=False, right=False)
        ax_label.set_ylim(ax_plot.get_ylim())
        ax_label.axis("off")

    model_display_names = {
        'CustomLLaMA': 'Fine-tuned models',
        'Gemini1_5Flash': 'Gemini 1.5 Flash',
        'GeminiProVision': 'Gemini Pro Vision',
        'LLaMA': 'Llama 3.2 Vision',
        'gpt4o': 'GPT-4o Vision',
        'Human': 'Human'
    }

    # Add the legend only if there are tasks
    if num_tasks > 0:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=model_colors[model_name], 
                       markersize=10, label=model_display_names[model_name])
            for model_name in model_colors.keys()
        ]
        axes[0, 1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    
    plt.savefig("Figure5.png",bbox_inches='tight', dpi=300)
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

import pandas as pd

def checkdeletedrows_forallcsv():
    file_paths = [
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP4/finetuning-EXP4numberone/EXP-Results/EXP4evaluate55images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP4/finetuning-EXP4numbertwo/EXP-Results/EXP4evaluate55images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP4/finetuning-EXP4numberthree/EXP-Results/EXP4evaluate55images.csv'
    ]
    
    all_balanced_metrics = []  # Store metrics for each file
    all_deleted_dfs = []  # Store deleted rows
    balanced_dataframes = []  # Store tuples of (df_framed, df_unframed)
    combined_framed_unframed = []  # Store all framed and unframed data


    for file_path in file_paths:
        print(f"\n📂 Processing file: {file_path}")

        # Clean and process raw answers
        df_framed, df_unframed, deleted_rows = clean_raw_answers(file_path)

        print(f"Task framed rows: {len(df_framed)}")
        print(f"Task unframed rows: {len(df_unframed)}")

        # Ensure 'cleaned_answers' column exists
        for df in [df_framed, df_unframed]:
            if 'cleaned_answers' not in df.columns:
                print("⚠️ Warning: 'cleaned_answers' column not found. Adding an empty column.")
                df['cleaned_answers'] = None

        # Balance datasets
        df_framed, df_unframed = balance_datasets(df_framed, df_unframed)

        # Store balanced DataFrames as tuples
        balanced_dataframes.append((df_framed, df_unframed))

        # Combine framed and unframed into one unified DataFrame
        df_framed['type'] = 'framed'
        df_unframed['type'] = 'unframed'
        combined_framed_unframed.append(pd.concat([df_framed, df_unframed], ignore_index=True))

        # Calculate metrics for balanced datasets
        metrics_table = calculate_metrics(df_framed, df_unframed)
        all_balanced_metrics.append(metrics_table)

        # Handle deleted rows
        if deleted_rows:
            deleted_df = pd.DataFrame(deleted_rows, columns=['raw_answer', 'model_name'])
            deleted_df['file'] = file_path.split('/')[-1]
            all_deleted_dfs.append(deleted_df)

    # ✅ Convert balanced_dataframes (list of tuples) to a single DataFrame
    dataframe_list = [df for pair in balanced_dataframes for df in pair]  # Flatten list of tuples
    balanced_df = pd.concat(dataframe_list, ignore_index=True) if dataframe_list else pd.DataFrame()

    # ✅ Convert combined_framed_unframed (list of DataFrames) to a single DataFrame
    combined_framed_unframed_df = pd.concat(combined_framed_unframed, ignore_index=True) if combined_framed_unframed else pd.DataFrame()

    # Combine metrics into a single DataFrame
    combined_metrics = pd.concat(all_balanced_metrics, ignore_index=True) if all_balanced_metrics else pd.DataFrame()

    # Combine deleted rows into a single DataFrame
    combined_deleted_df = pd.concat(all_deleted_dfs, ignore_index=True) if all_deleted_dfs else pd.DataFrame(columns=['file', 'raw_answer', 'model_name'])

    # ✅ Ensure 'cleaned_answers' is formatted correctly
    if 'cleaned_answers' in balanced_df.columns:
        balanced_df['cleaned_answers'] = (
            balanced_df['cleaned_answers']
            .astype(str)                           # Convert everything to string (to avoid NaN issues)
            .str.replace(r'\n', '', regex=True)    # Remove newlines
            .str.strip()                            # Trim spaces
            .replace('', None)                      # Convert empty strings to None
        )

        # Convert to numeric after cleaning
        balanced_df['cleaned_answers'] = pd.to_numeric(balanced_df['cleaned_answers'], errors='coerce')
    else:
        print("⚠️ Warning: 'cleaned_answers' column not found in balanced_df.")

    # ✅ Save the final cleaned dataset
    balanced_df.to_excel("finalEXP4.xlsx", index=False)
    print("\n✅ Final cleaned dataset saved as 'finalEXP4.csv'")

    return combined_framed_unframed_df

    # Return all values
    #combined_metrics, combined_deleted_df, balanced_df, all_balanced_metrics, combined_framed_unframed_df


def plot_multiplerun(metrics_table):
    """
    Plot results for multiple runs using balanced datasets metrics.
    
    Args:
        metrics_table (pd.DataFrame): Metrics table containing all runs
    
    Returns:
        pd.DataFrame: Averaged metrics table
    """
    #print(metrics_table)
    print("\nAveraged Metrics (Balanced Datasets):")
    
    # Convert the metrics_table DataFrame into a list with a single DataFrame
    metrics_list = [metrics_table]
    
    # Calculate averaged metrics
    averaged_metrics_table = average_metrics(metrics_list)
    
    # Plot the results
    plot_results(averaged_metrics_table)
    
    return averaged_metrics_table


"""Check unique answers after balancing number of datasets"""


def check_combined_unique_answers(balanced_dataframes):
    """
    Combine all balanced DataFrames, check unique values and counts across all datasets,
    and compare results between pretrained and finetuned models with point chart visualization.

    Parameters:
    balanced_dataframes (list): List of tuples containing (df_framed, df_unframed) for each dataset.

    Returns:
    dict: Dictionary summarizing unique values and counts for pretrained and finetuned models.
    """
    # Combine all framed and unframed DataFrames into one
    combined_framed = pd.concat([df_framed for df_framed, _ in balanced_dataframes], ignore_index=True)
    combined_unframed = pd.concat([df_unframed for _, df_unframed in balanced_dataframes], ignore_index=True)

    print(f"Combined Framed Dataset: {len(combined_framed)} rows")
    print(f"Combined Unframed Dataset: {len(combined_unframed)} rows")


def check_combined_unique_answers_overlay(balanced_dataframes):
    """
    Visualize unique value counts for Pretrained and Finetuned models,
    overlaying unframed data on framed plots, using Matplotlib.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Combine all DataFrames dynamically
    combined_all = pd.concat(
        [pd.concat(df_list, ignore_index=True) for df_list in balanced_dataframes],
        ignore_index=True
    )

    # Define pretrained and finetuned model groups
    pretrained_models = {'Gemini1_5Flash', 'GeminiProVision', 'LLaMA', 'gpt4o'}
    finetuned_models = {'CustomLLaMA'}

    # Categorize models
    combined_all['model_type'] = combined_all['model_name'].apply(
        lambda x: 'Pretrained' if x in pretrained_models else 'Finetuned' if x in finetuned_models else 'Unknown'
    )

    
    # Initialize results dictionary
    results = {}

    # Custom colors for each combination of data_type and model_type
    custom_colors = {
        ('framed', 'Pretrained'): '#3A50A1',  # Dark Blue for Pretrained - Framed
        ('framed', 'Finetuned'): '#8FA8FF',  # Light Blue for Finetuned - Framed
        ('unframed', 'Pretrained'): '#4C8C2B',  # Dark Green for Pretrained - Unframed
        ('unframed', 'Finetuned'): '#A3D977',  # Light Green for Finetuned - Unframed
    }

    # Define marker styles
    marker_styles = {
        ('framed', 'Pretrained'): 'o',  # Circle for Pretrained - Framed
        ('framed', 'Finetuned'): 's',  # Square for Finetuned - Framed
        ('unframed', 'Pretrained'): '^',  # Triangle for Pretrained - Unframed
        ('unframed', 'Finetuned'): 'x',  # Cross for Finetuned - Unframed
    }

    # Create a single set of subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    ax_below_49, ax_between_49_60, ax_above_60 = axes

    for data_type in ['framed', 'unframed']:
        for model_type in ['Pretrained', 'Finetuned']:
            group = combined_all[(combined_all['model_type'] == model_type) & (combined_all['type'] == data_type)]

            # Flatten cleaned answers and ensure numeric values
            flattened_cleaned_answers = group['cleaned_answers'].explode()
            flattened_cleaned_answers = flattened_cleaned_answers.dropna().apply(
                lambda x: x if isinstance(x, (int, float)) else x[0] if isinstance(x, list) else None
            ).dropna()

            # Count unique values and sort
            value_counts = flattened_cleaned_answers.value_counts().sort_index()
            results[(data_type, model_type)] = value_counts

            # Split data into 3 regions
            below_49 = value_counts[value_counts.index < 49]
            between_49_60 = value_counts[(value_counts.index >= 49) & (value_counts.index <= 60)]
            above_60 = value_counts[value_counts.index > 60]

            # Overlay plots
            ax_below_49.scatter(below_49.index, below_49.values, 
                                label=f"{data_type.capitalize()} ({model_type})", 
                                color=custom_colors[(data_type, model_type)],
                                marker=marker_styles[(data_type, model_type)], s=50, alpha=0.8)

            ax_between_49_60.scatter(between_49_60.index, between_49_60.values, 
                                    label=f"{data_type.capitalize()} ({model_type})", 
                                    color=custom_colors[(data_type, model_type)],
                                    marker=marker_styles[(data_type, model_type)], s=50, alpha=0.8)

            ax_above_60.scatter(above_60.index, above_60.values, 
                                label=f"{data_type.capitalize()} ({model_type})", 
                                color=custom_colors[(data_type, model_type)],
                                marker=marker_styles[(data_type, model_type)], s=50, alpha=0.8)

    # Subplot Settings
    ax_below_49.set_title("Values < 49")
    ax_below_49.set_ylabel("Counts")
    ax_below_49.grid(True, linestyle='--', alpha=0.5)

    ax_between_49_60.set_title("Values 49-60")
    ax_between_49_60.grid(True, linestyle='--', alpha=0.5)
    ax_between_49_60.set_xlim(49, 60)

    ax_above_60.set_title("Values > 60")
    ax_above_60.grid(True, linestyle='--', alpha=0.5)
    ax_above_60.set_xlim(61, 120)

    # Add legends and layout adjustments
    ax_below_49.legend(title="Data Type & Model", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust top margin to fit suptitle

    # Save the plot
    plt.savefig('analysisexp4.png', bbox_inches='tight', format='png', transparent=False)

    plt.show()

    return results


def visualize_cleaned_answers_separate_scatter(balanced_dataframes):
    """
    Visualize scatter plots of mean cleaned answers for pretrained and finetuned models,
    separated into framed and unframed categories, with a ground truth reference line.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import ast

    # Combine dataframes into a single DataFrame
    combined_list = []
    for df_framed, df_unframed in balanced_dataframes:
        df_framed = df_framed.copy()
        df_framed['type'] = 'framed'
        combined_list.append(df_framed)

        df_unframed = df_unframed.copy()
        df_unframed['type'] = 'unframed'
        combined_list.append(df_unframed)

    combined_df = pd.concat(combined_list, ignore_index=True)

    # Define pretrained and finetuned models
    pretrained_models = {'Gemini1_5Flash', 'GeminiProVision', 'LLaMA', 'gpt4o'}
    finetuned_model = {'CustomLLaMA'}

    # Group models into categories
    combined_df['model_group'] = combined_df['model_name'].apply(
        lambda x: 'Pretrained Models' if x in pretrained_models else 'Finetuned Model' if x in finetuned_model else None
    )

    # Filter only pretrained and finetuned models
    combined_df = combined_df[combined_df['model_group'].notna()]

    # Explode lists and convert to numeric
    for col in ['ground_truth', 'cleaned_answers']:
        combined_df[col] = combined_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    combined_df = combined_df.explode('ground_truth').explode('cleaned_answers').dropna()
    combined_df['ground_truth'] = pd.to_numeric(combined_df['ground_truth'], errors='coerce')
    combined_df['cleaned_answers'] = pd.to_numeric(combined_df['cleaned_answers'], errors='coerce')

    # Group to calculate mean cleaned answers for model groups
    grouped_df = combined_df.groupby(['ground_truth', 'model_group', 'type'])['cleaned_answers'].mean().reset_index()

    # Prepare Ground Truth Line
    ground_truth_line = pd.DataFrame({
        'ground_truth': range(49, 61),
        'cleaned_answers': range(49, 61)
    })

    
    # Define marker styles
    marker_styles = {'Pretrained Models': 'o', 'Finetuned Model': '^'}  # Circle for Pretrained, Triangle for Finetuned

    # Start Plotting
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Define the palette for the scatter plots
    palette = {'Pretrained Models': '#3A50A1', 'Finetuned Model': '#4C8C2B'}

    # Scatter Plot for Framed Data
    framed_df = grouped_df[grouped_df['type'] == 'framed']
    sns.scatterplot(data=framed_df, x='ground_truth', y='cleaned_answers',
                    hue='model_group', style='model_group', markers=marker_styles,
                    palette=palette, s=100, ax=axes[0])
    axes[0].plot(ground_truth_line['ground_truth'], ground_truth_line['cleaned_answers'],
                linestyle='--', color='green', linewidth=2, label='Ground Truth Line')
    axes[0].set_xlabel("Ground Truth")
    axes[0].set_ylabel("Mean Cleaned Answers")
    axes[0].legend()

    # Scatter Plot for Unframed Data
    unframed_df = grouped_df[grouped_df['type'] == 'unframed']
    sns.scatterplot(data=unframed_df, x='ground_truth', y='cleaned_answers',
                    hue='model_group', style='model_group', markers=marker_styles,
                    palette=palette, s=100, ax=axes[1])
    axes[1].plot(ground_truth_line['ground_truth'], ground_truth_line['cleaned_answers'],
                linestyle='--', color='green', linewidth=2, label='Ground Truth Line')
    axes[1].set_xlabel("Ground Truth")
    axes[1].set_ylabel("")
    axes[1].legend()

    plt.savefig('analysisexp4.png', 
        bbox_inches='tight', 
        format='png',
        transparent=False)

    plt.show()
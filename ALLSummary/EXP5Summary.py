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

def plot_results(metrics_table):
    """
    Plot the results from the metrics table with human benchmark values
    """
    summary_stats_by_task = {df_name: metrics_table[metrics_table['Dataset'] == df_name] 
                            for df_name in metrics_table['Dataset'].unique()}

    # Define display names for tasks
    task_display_names = {
        'Task_10': '10 DOTS',
        'Task_100': '100 DOTS',
        'Task_1000': '1000 DOTS'
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
        'Task_10': (4.0149, 0.5338),
        'Task_100': (5.3891, 0.1945),
        'Task_1000': (5.4612, 0.2509)
    }

    # Define task images
    # Auto-detect task images
    def find_task_images(base_path):
        task_images = {}
        image_extensions = ['.jpg', '.jpeg', '.png']
        
        try:
            for task in [10, 100, 1000]:
                task_pattern = f"{task}_"
                image_dir = os.path.join(base_path, 'images')
                if os.path.exists(image_dir):
                    for file in os.listdir(image_dir):
                        if file.startswith(task_pattern) and any(file.lower().endswith(ext) for ext in image_extensions):
                            task_images[f'Task_{task}'] = os.path.join(image_dir, file)
                            break
        except Exception as e:
            print(f"Warning: Error finding images: {str(e)}")
        
        return task_images

    # Get task images
    base_path = '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberone'
    task_images = find_task_images(base_path)
    
    num_tasks = len(summary_stats_by_task)
    
    fig, axes = plt.subplots(num_tasks, 3, figsize=(12, 2 * num_tasks), 
                            gridspec_kw={'width_ratios': [1, 6, 1]}, sharex=False)
    
    fig.subplots_adjust(hspace=0.7, left=0.1, right=0.95, top=0.98)
   
    fig.patch.set_facecolor('white')

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

        

        ax_img.set_title(task_display_names.get(task_name, task_name), loc="left", fontsize=14, color="black")

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
        ax_plot.axvline(-6, color="black", linewidth=1)
        ax_plot.axvline(12, color="black", linewidth=1)
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
        ax_plot.set_xlim(-6, 12)
        ax_plot.invert_yaxis()

        # Display model names
        ax_label.set_yticks(y_positions)
        ax_label.set_yticklabels(sorted_model_names, fontsize=14)
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
    
    plt.savefig("Figure6.png",bbox_inches='tight')

    plt.show()

def process_plot(metrics_table):
    """
    Process and create the plot for given metrics table
    """
    print("\nGenerating plot...")
    plot_results(metrics_table)
   
def find_task_images(base_path):

    """
    Automatically find images for each task in the given directory
    Parameters:
    base_path (str): Base directory path where images are stored
    Returns:
    dict: Dictionary mapping task names to image paths
    """
    task_images = {}
    image_extensions = ['.jpg', '.jpeg', '.png']

    try:
        # For each task (10, 100, 1000)
        for task in [10, 100, 1000]:
            # Look for images that start with the task number
            task_pattern = f"{task}_"
            
            # Search in the images directory
            image_dir = os.path.join(base_path, 'images')
            if os.path.exists(image_dir):
                for file in os.listdir(image_dir):
                    if file.startswith(task_pattern) and any(file.lower().endswith(ext) for ext in image_extensions):
                        task_images[f'Task_{task}'] = os.path.join(image_dir, file)
                        break  # Take the first matching image for each task
    except Exception as e:
        print(f"Warning: Error finding images: {str(e)}")

    return task_images

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

def checkdeletedrows_forallcsv():
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

def check_combined_unique_answers(balanced_dataframes):

    import pandas as pd
    import matplotlib.pyplot as plt

    """
    Combine all balanced DataFrames, check unique values and counts across all datasets,
    and compare pretrained and finetuned models in single combined plots.
    """

    # Define model groups
    pretrained_models = {'Gemini1_5Flash', 'GeminiProVision', 'LLaMA', 'gpt4o'}
    finetuned_models = {'CustomLLaMA'}

    # Combine datasets
    combined_10 = pd.concat([df_10 for df_10, _, _ in balanced_dataframes], ignore_index=True)
    combined_100 = pd.concat([df_100 for _, df_100, _ in balanced_dataframes], ignore_index=True)
    combined_1000 = pd.concat([df_1000 for _, _, df_1000 in balanced_dataframes], ignore_index=True)

    # Add new_model_type column
    for df in [combined_10, combined_100, combined_1000]:
        df['new_model_type'] = df['model_name'].apply(
            lambda x: 'Pretrained' if x in pretrained_models else 'Finetuned' if x in finetuned_models else None
        )

    ## Define task grouping and colors
    grouped_tasks = {
        '10_dots_pretrained': combined_10[combined_10['new_model_type'] == 'Pretrained'],
        '10_dots_finetuned': combined_10[combined_10['new_model_type'] == 'Finetuned'],
        '100_dots_pretrained': combined_100[combined_100['new_model_type'] == 'Pretrained'],
        '100_dots_finetuned': combined_100[combined_100['new_model_type'] == 'Finetuned'],  # Fixed typo
        '1000_dots_finetuned': combined_1000[combined_1000['new_model_type'] == 'Finetuned'],
        '1000_dots_pretrained': combined_1000[combined_1000['new_model_type'] == 'Pretrained']
    }

    # Define markers and colors
    task_markers = {
        '10_dots_pretrained': 'o',
        '10_dots_finetuned': '^',
        '100_dots_pretrained': 'o',
        '100_dots_finetuned': '^',  # Fixed typo
        '1000_dots_finetuned': '^',
        '1000_dots_pretrained': 'o',
    }

    task_colors = {
        '10_dots_pretrained': '#1f77b4',  # Blue for Pretrained
        '10_dots_finetuned': '#9467bd',   # Purple for Finetuned
        '100_dots_pretrained': '#ff7f0e', # Orange for Pretrained
        '100_dots_finetuned': '#8c564b',  # Brown for Finetuned (Fixed typo)
        '1000_dots_finetuned': '#bcbd22', # Yellow-Green for Finetuned
        '1000_dots_pretrained': '#2ca02c' # Green for Pretrained
    }

   # Prepare figure: one plot for Pretrained, one for Finetuned
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ranges = [(1, 10), (16, 7000)] 
    model_types = ['Pretrained', 'Finetuned']  # Two model types for separation

    # Loop through model types and axes
    for ax, model_type in zip(axes, model_types):
        for range_min, range_max in ranges:
            for task_name, task_data in grouped_tasks.items():
                # Filter for current model type
                group = task_data[task_data['new_model_type'] == model_type]
                flattened_answers = group['cleaned_answers'].dropna().explode()
                flattened_answers = flattened_answers.apply(
                    lambda x: x if isinstance(x, (int, float)) else x[0] if isinstance(x, list) else None
                ).dropna()

                
 
                # Count unique values and filter by range
                value_counts = flattened_answers.value_counts().sort_index()

                filtered_counts = value_counts[
                    (value_counts.index >= range_min) & (value_counts.index <= range_max)
                ]

                print(f"{model_type} filtered counts between {range_min} and {range_max}:")
                print(filtered_counts)
                
                # Define marker and color
                task_key = task_name
                color = task_colors[task_key]
                marker = task_markers[task_key]

                # Plot points
                ax.scatter(
                    filtered_counts.index, filtered_counts.values,
                    label=task_name.replace('_', ' ').capitalize(),
                    color=color, marker=marker, alpha=0.7, s=50
                )

            # Add vertical reference line
            #if range_min <= 10 <= range_max:
                #ax.axvline(x=10, color='red', linestyle='--', linewidth=1)
                #ax.text(10.5, ax.get_ylim()[1] * 0.9, "x=10", color='red', fontsize=10)

        # Plot settings
        ax.set_xlabel("Unique Values")
        ax.set_ylabel("Counts")
        ax.grid(True, linestyle='--', alpha=0.6)

    # Add legends
    handles = [
        plt.Line2D([0], [0], marker=task_markers[task], color='w',
                markerfacecolor=task_colors[task], markersize=10, label=task)
        for task in task_markers
    ]
    fig.legend(handles=handles, loc='center right', title="Task Type")

    plt.savefig('analysisexp5.png', 
        bbox_inches='tight', 
        format='png',
        transparent=False)

    plt.show()

    return combined_10, combined_100, combined_1000
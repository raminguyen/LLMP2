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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.show_dimensions', False)

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

    print("Rows before cleaning:")
    print(f"  📥 Total rows: {len(df)}")

    task_counts = {}
    for task in df['task_name'].unique():
        task_rows = len(df[df['task_name'] == task])
        task_counts[task] = task_rows
        number_unique_images = df[df['task_name'] == task]['image_path'].nunique()
        print(f"  Task {task}: {task_rows} rows, {number_unique_images} unique images")

    print(f" Total tasks: {sum(task_counts.values())}")

    deleted_rows = []

    total_tasks = len(df)
    
    def extract_digits_exp2(raw_text, model_name=None):
        """Extraction logic specific to EXP2."""
        if pd.isna(raw_text):
            return None
            
        # Clean the text: remove newlines and backslashes
        raw_text = str(raw_text).strip().replace('\n', ' ').replace('\\', '')
        
        # Extract lists inside square brackets
        matches = re.findall(r'\[([\d.,\s]+)\]', raw_text)
        
        if not matches:
            descriptive_patterns = [
                # Basic formats
                r'\[([\d\., ]+)\]',  # Basic bracketed numbers
                
                # Common ratio descriptions
                r'ratios are.*?\[([\d\., ]+)\]',
                r'values are.*?\[([\d\., ]+)\]',
                r'approximately.*?\[([\d\., ]+)\]',
                
                # Specific descriptions
                r'moving left to right.*?\[([\d\., ]+)\]',
                r'largest bar.*?\[([\d\., ]+)\]',
                r'maximum are:.*?\[([\d\., ]+)\]',
                
                # Full sentence patterns
                r'Based on the (?:image|bar chart|chart).*?maximum are:\s*\[([\d\., ]+)\]',
                r'estimate the ratios.*?\[([\d\., ]+)\]',
                r'estimated ratios.*?maximum are:\s*\[([\d\., ]+)\]',
                
                # New patterns for narrative style responses
                r'My estimate is:\s*`\[([\d\., ]+)\]`',  # Simple backtick format
                r'To solve this[\s\S]*?My estimate is:\s*`\[([\d\., ]+)\]`',  # Full narrative pattern
                r'estimate is:\s*`?\[?([\d\., ]+)\]?`?',  # Flexible format
                r'(?:^|[^\d])(1\.0\s*,\s*0\.6\s*,\s*0\.4\s*,\s*0\.25\s*,\s*0\.1)(?:[^\d]|$)'  # Exact sequence
            ]
            
            for pattern in descriptive_patterns:
                match_pattern = re.search(pattern, raw_text, re.IGNORECASE | re.DOTALL)
                if match_pattern:
                    matches = [match_pattern.group(1)]
                    break
        
        if matches:
            valid_lists = []
            for match in matches:
                # Clean up any extra spaces and split by comma
                numbers = [
                    float(num.strip()) 
                    for num in match.split(',') 
                    if re.match(r'^\d*\.?\d+$', num.strip())
                ]
                if len(numbers) == 5:  # Only keep lists of exactly 5 numbers
                    valid_lists.append(numbers)
            return valid_lists[-1] if valid_lists else None
        return None
    
    # Example loop to clean the data
    rows_to_delete = []
    
    for index, row in df.iterrows():
        model_name = row.get('model', '')  # Assuming model name is in a 'model' column
        raw_text = row.get('raw_answer', '')  # Assuming raw answers are in a 'raw_answer' column

        # Extract digits
        cleaned_answer = extract_digits_exp2(raw_text, model_name)

        # Check if answer is valid, otherwise mark for deletion
        if cleaned_answer is None:
            deleted_rows.append(row)
            rows_to_delete.append(index)

    # Drop rows marked for deletion
    df.drop(rows_to_delete, inplace=True)

    # Print deleted rows and total tasks
    print(f"Total tasks: {total_tasks}")   
    
    # Process the dataframe
    df['cleaned_answers'] = df['raw_answer'].apply(extract_digits_exp2)
    
    # Split the dataframe by task
    df_bar = df[df['task_name'] == 'bar'].copy()
    df_pie = df[df['task_name'] == 'pie'].copy()
   
    return df_bar, df_pie, deleted_rows


def plot_results(metrics_table):
    """
    Plot the results from the metrics table with human benchmark values
    """
    summary_stats_by_task = {df_name: metrics_table[metrics_table['Dataset'] == df_name] 
                            for df_name in metrics_table['Dataset'].unique()}

    # Define display names for tasks
    task_display_names = {
        'bar': 'BAR',
        'pie': 'PIE',

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

     # Original names and their corresponding display names for the legend
    model_display_names = {
        'CustomLLaMA': 'Fine-tuned models',
        'Gemini1_5Flash': 'Gemini 1.5 Flash',
        'GeminiProVision': 'Gemini Pro Vision',
        'LLaMA': 'Llama 3.2 Vision',
        'gpt4o': 'GPT-4o Vision',
        'Human': 'Human'
    }

    

    # Define Human benchmark data
    human_data = {
        'pie': (2.05, 0.115),
        'bar': (1.035, 0.125),
    }

    # Get task images
    base_path = '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numberone/images'
    task_images = find_task_images(base_path)
    
    num_tasks = len(summary_stats_by_task)


    fig, axes = plt.subplots(num_tasks, 3, figsize=(10, 2 * num_tasks), 
                        gridspec_kw={'width_ratios': [1, 4, 1]}, sharex=False)
    fig.subplots_adjust(hspace=0.3, left=0.05, right=0.95, top=0.98, bottom=0.02)


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
                ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=16, color="black")
                ax_img.set_facecolor("white")

        ax_img.axis('off')
        ax_img.set_title(task_display_names.get(task_name, task_name), loc="left", fontsize=16, color="black")

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

        ax_plot.axvline(-3, color="black", linewidth=2)
        ax_plot.axvline(2.5, color="black", linewidth=2)
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
        ax_plot.set_xlim(-3, 2.5)
        ax_plot.invert_yaxis()

        ax_label.set_yticks(y_positions)
        ax_label.set_yticklabels(sorted_model_names, fontsize=14)
        ax_label.tick_params(left=False, right=False, labelleft=False, labelright=True)
        ax_plot.tick_params(axis='y', which='both', left=False, right=False)
        ax_plot.tick_params(axis='x', labelsize=14)
        ax_label.set_ylim(ax_plot.get_ylim())
        ax_label.axis("off")

    if num_tasks > 0:  # Only add legend if there are tasks
        axes[0, 1].legend(loc="best", frameon=False)

    # Add the legend only if there are tasks
    if num_tasks > 0:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=model_colors[model_name], 
                       markersize=10, label=model_display_names[model_name])
            for model_name in model_colors.keys()
        ]
        axes[0, 1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.6, 0.5), frameon=False)

    plt.savefig("Figure3.png",bbox_inches='tight')
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
        task_types = ['bar', 'pie']

    for task in task_types:
        task_pattern = f"{task}_"
        
        for file in os.listdir(base_path):
            if file.startswith(task_pattern) and any(file.lower().endswith(ext) for ext in image_extensions):
                task_images[task] = os.path.join(base_path, file)
                break

    return task_images


""" Average 3 running """

def calculate_metrics(df_bar, df_pie):
    metrics_summary = {}
    dataframes = {
        'bar': df_bar, 
        'pie': df_pie, 
    }

    for df_name, df in dataframes.items():
        if df.empty:
            continue
            
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


def balance_datasets(df_bar, df_pie, target_size=677):

    # Balance bar chart dataset
    df_bar_balanced = df_bar.sample(
        n=target_size,
        replace=len(df_bar) < target_size,
        random_state=42
    )
    
    # Balance pie chart dataset
    df_pie_balanced = df_pie.sample(
        n=target_size,
        replace=len(df_pie) < target_size,
        random_state=42
    )
    
    return df_bar_balanced, df_pie_balanced

def checkdeletedrows_forallcsv():
    """
    Process and check deleted rows across all CSV files for EXP2, 
    returning balanced dataframes and combined metrics.

    Returns:
    combined_deleted_df (pd.DataFrame): Combined DataFrame of deleted rows.
    balanced_dataframes (list): List of tuples containing (df_bar_balanced, df_pie_balanced).
    metrics_table (pd.DataFrame): Combined DataFrame of metrics from all datasets.
    """
    # List of file paths for EXP2
    file_paths = [
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numberone/EXP-Results/EXP2results55images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numbertwo/EXP-Results/EXP2results55images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numberthree/EXP-Results/EXP2results55images.csv'
    ]

    all_balanced_metrics = []  # Store metrics for each file
    all_deleted_dfs = []  # Store deleted rows
    balanced_dataframes = []  # Store tuples of (df_bar_balanced, df_pie_balanced)

    for file_path in file_paths:
        print(f"\n📂 Processing: {file_path.split('/')[-1]}")

        # Clean and process raw answers
        df_bar, df_pie, deleted_rows = clean_raw_answers(file_path)

        # Print original dataset sizes
        print("\nOriginal Dataset Sizes:")
        print(f"Bar chart rows: {len(df_bar)}")
        print(f"Pie chart rows: {len(df_pie)}")

        # Balance datasets
        df_bar_balanced, df_pie_balanced = balance_datasets(df_bar, df_pie)
        balanced_dataframes.append((df_bar_balanced, df_pie_balanced))  # Append as tuple

        # Print balanced dataset sizes
        print("\nBalanced Dataset Sizes:")
        print(f"Bar chart rows: {len(df_bar_balanced)}")
        print(f"Pie chart rows: {len(df_pie_balanced)}")
        print(f"Deleted rows: {len(deleted_rows)}")

        # Add a 'Dataset' column for tracking and calculate metrics
        metrics_table = calculate_metrics(df_bar_balanced, df_pie_balanced)
        all_balanced_metrics.append(metrics_table)

        # Handle deleted rows
        if deleted_rows:
            deleted_df = pd.DataFrame(deleted_rows, columns=['raw_answer', 'model_name'])
            deleted_df['file'] = file_path.split('/')[-1]
            all_deleted_dfs.append(deleted_df)

    # Combine all metrics into a single DataFrame
    metrics_table = pd.concat(all_balanced_metrics, ignore_index=True) if all_balanced_metrics else pd.DataFrame()

    # Combine all deleted rows into a single DataFrame
    combined_deleted_df = pd.concat(all_deleted_dfs, ignore_index=True) if all_deleted_dfs else pd.DataFrame(columns=['file', 'raw_answer', 'model_name'])

    # Return combined deleted rows, balanced dataframes, and combined metrics
    return combined_deleted_df, balanced_dataframes, metrics_table, all_balanced_metrics

def process_and_plot_multiplerun(metrics_table):
    print("\nAveraged Metrics (Balanced Datasets):")
    
    # Convert the metrics_table DataFrame into a list with a single DataFrame
    metrics_list = [metrics_table]
    
    # Calculate averaged metrics
    averaged_metrics_table = average_metrics(metrics_list)
    
    # Plot the results
    plot_results(averaged_metrics_table)
    
    return averaged_metrics_table

def check_combined_unique_answers(balanced_dataframes):

    # Combine all bar and pie DataFrames into one
    combined_bar = pd.concat([df_bar for df_bar, _ in balanced_dataframes], ignore_index=True)
    combined_pie = pd.concat([df_pie for _, df_pie in balanced_dataframes], ignore_index=True)

    print(f"Combined Bar Dataset: {len(combined_bar)} rows")
    print(f"Combined Pie Dataset: {len(combined_pie)} rows")

    # Initialize dictionary to store unique value counts
    unique_values_summary = {}

    # Check unique values in 'cleaned_answers' for bar dataset
    if 'cleaned_answers' in combined_bar.columns:
        # Flatten lists in 'cleaned_answers' column
        cleaned_bar = combined_bar['cleaned_answers'].explode().dropna()
        bar_value_counts = cleaned_bar.value_counts().sort_index()
        bar_unique_counts = bar_value_counts.size
        unique_values_summary['bar'] = {'unique_count': bar_unique_counts, 'sorted_values': bar_value_counts}
        print(f"\nUnique answers in Bar dataset (sorted):")
        print(bar_value_counts)
    else:
        print("Warning: 'cleaned_answers' column not found in Bar dataset.")
        unique_values_summary['bar'] = {'unique_count': 0, 'sorted_values': None}

    # Check unique values in 'cleaned_answers' for pie dataset
    if 'cleaned_answers' in combined_pie.columns:

        # Flatten lists in 'cleaned_answers' column
        cleaned_pie = combined_pie['cleaned_answers'].explode().dropna()
        pie_value_counts = cleaned_pie.value_counts().sort_index()
        pie_unique_counts = pie_value_counts.size
        unique_values_summary['pie'] = {'unique_count': pie_unique_counts, 'sorted_values': pie_value_counts}
        print(f"\nUnique answers in Pie dataset (sorted):")
        print(pie_value_counts)
    else:
        print("Warning: 'cleaned_answers' column not found in Pie dataset.")
        unique_values_summary['pie'] = {'unique_count': 0, 'sorted_values': None}

    #return unique_values_summary

def visualize_cleaned_answers_overlay(balanced_dataframes):
    """
    Visualize scatter plots for unique value counts in cleaned answers for Pretrained and Finetuned models,
    overlaying Bar and Pie tasks, with separate plots for 0-1 and 1.1-83 ranges.
    """

    # Combine all dataframes dynamically
    combined_all = pd.concat(
        [pd.concat([df_bar.assign(type='bar'), df_pie.assign(type='pie')], ignore_index=True)
         for df_bar, df_pie in balanced_dataframes],
        ignore_index=True
    )

    # Define pretrained and finetuned models
    pretrained_models = {'Gemini1_5Flash', 'GeminiProVision', 'LLaMA', 'gpt4o'}
    finetuned_models = {'CustomLLaMA'}

    # Categorize models
    combined_all['model_type'] = combined_all['model_name'].apply(
        lambda x: 'Pretrained' if x in pretrained_models else 'Finetuned' if x in finetuned_models else 'Unknown'
    )

    # Initialize results dictionary
    results = {}

    custom_colors = {
        ('bar', 'Pretrained'): '#4C8C2B',  # Darker Green for Pretrained - Bar
        ('bar', 'Finetuned'): '#A3D977',  # Light Green for Finetuned - Bar
        ('pie', 'Pretrained'): '#3A50A1',  # Darker Blue for Pretrained - Pie
        ('pie', 'Finetuned'): '#8FA8FF',  # Light Blue for Finetuned - Pie
    }

    # Define marker styles and sizes
    marker_styles = {
        'Pretrained': ('o', 50),  # 'o' for circle, size 50
        'Finetuned': ('^', 50)  # '^' for triangle, size 50
    }

    # Create subplots for the two ranges
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax_0_1, ax_1_83 = axes


    # Process each type (Bar and Pie) and model group (Pretrained and Finetuned)
    for task_type in ['bar', 'pie']:
        for model_type in ['Pretrained', 'Finetuned']:
            group = combined_all[(combined_all['model_type'] == model_type) & (combined_all['type'] == task_type)]

            # Flatten cleaned answers and ensure numeric values
            flattened_cleaned_answers = group['cleaned_answers'].explode()
            flattened_cleaned_answers = flattened_cleaned_answers.dropna().apply(
                lambda x: x if isinstance(x, (int, float)) else x[0] if isinstance(x, list) else None
            ).dropna()

            # Count unique values and sort
            value_counts = flattened_cleaned_answers.value_counts().sort_index()
            results[(task_type, model_type)] = value_counts

            # Split data into the two defined ranges
            range_0_1 = value_counts[(value_counts.index >= 0) & (value_counts.index <= 1)]
            range_1_83 = value_counts[(value_counts.index > 1) & (value_counts.index <= 83)]

            # Overlay plots for the two ranges
            # 1. Plot for 0-1 range
            ax_0_1.scatter(range_0_1.index, range_0_1.values,
                       label=f"{task_type.capitalize()} ({model_type})",
                       color=custom_colors[(task_type, model_type)],
                       s=marker_styles[model_type][1],  # size from marker_styles
                       marker=marker_styles[model_type][0],  # marker shape from marker_styles
                       alpha=0.8)

            # 2. Plot for 1.1-83 range
            ax_1_83.scatter(range_1_83.index, range_1_83.values,
                            label=f"{task_type.capitalize()} ({model_type})",
                            color=custom_colors[(task_type, model_type)],
                            s=marker_styles[model_type][1],  # size
                            marker=marker_styles[model_type][0],  # shape
                            alpha=0.8)

    # Subplot Settings
    ax_0_1.set_title("Values 0-1")
    ax_0_1.set_xlabel("Unique Values")
    ax_0_1.set_ylabel("Counts")
    ax_0_1.grid(True, linestyle='--', alpha=0.5)

    ax_1_83.set_title("Values 1.1-83")
    ax_1_83.set_xlabel("Unique Values")
    ax_1_83.grid(True, linestyle='--', alpha=0.5)

    # Add legends and layout adjustments
    ax_0_1.legend(title="Task Type & Model", fontsize=10)
   
    plt.savefig('analysisexp2.png', 
    bbox_inches='tight', 
    format='png',
    transparent=False)
    plt.show()

    return results

# === IMPORTS ===

# Core libraries
import os
import re
import ast
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Visualization libraries
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Bootstrap library for statistical analysis
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

# === EXPERIMENT FUNCTIONS ===

# --- CLEANING FUNCTIONS ---
def clean_raw_answers(file_path, extraction_logic, task_types, model_name=None):
    """
    Cleans raw answers from a CSV file by applying specified extraction logic.
    """
    # Load the CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path} with {len(df)} rows.")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
    
    # Count initial NaN values
    initial_na_count = df['raw_answer'].isna().sum()
    print(f"Initial NaN values in '': {initial_na_count}")
    
    # Apply extraction logic
    df['parsed_answers'] = df.apply(lambda row: extraction_logic(row['raw_answer']), axis=1)
    
    # Format cleaned values as strings with one decimal point
    df['parsed_answers'] = df['parsed_answers'].apply(lambda x: '{:.1f}'.format(float(x)) if not pd.isna(x) else x)
    

    # Store rows with no valid parsed answers
    deleted_rows = df[df['parsed_answers'].isna()].copy()
    df_cleaned = df.dropna(subset=['parsed_answers'])
    
    # Split cleaned DataFrame by task type
    task_dfs = {}
    for task in task_types:
        task_data = df_cleaned[df_cleaned['task_name'] == task]
        if len(task_data) > 0:
            task_dfs[task] = task_data
            print(f"\nProcessing dataset: {task}")
            print(f"  Remaining rows after cleanup: {len(task_data)}")
        else:
            print(f"\nProcessing dataset: {task}")
            print(f"  ⚠️ Dataset {task} has no valid rows. Skipping...")
    
    # Summarize cleanup
    print(f"\n📄 File Processed: {file_path}")
    print(f"  ✅ Total Rows: {len(df)}")
    print(f"  ✅ Valid Rows: {len(df_cleaned)}")
    print(f"  ❌ Invalid Rows: {len(deleted_rows)}")
    
    if len(deleted_rows) > 0:
        print("\nRemoved rows:")
        print(deleted_rows[['raw_answer']])
    
    return task_dfs, deleted_rows

def extract_digits_exp1(raw_text, model_name):
    """
    Extracts numerical values from raw answers in Experiment 1.
    Handles cases with multiple numbers, 'user' prefixes, and other patterns.

    Parameters:
        raw_text (str): Raw text from the 'raw_answer' column.

    Returns:
        list or np.nan: Extracted list of numbers or NaN if no valid numbers are found.
    """
    if pd.isna(raw_text):
        return np.nan

    # Clean the text
    raw_text = str(raw_text).strip().replace('\n', ' ')

    # Case 1: If the text starts with "user", try extracting the last number
    if raw_text.startswith('user'):
        numbers = re.findall(r'\d+\.?\d*', raw_text)
        if numbers:
            return [float(numbers[-1])]  # Return the last number as a list

    # Case 2: Extract numbers formatted in square brackets (e.g., "[12.5, 34.0]")
    bracket_match = re.search(r'\[([\d.,\s]+)\]', raw_text)
    if bracket_match:
        try:
            numbers = [float(num.strip()) for num in bracket_match.group(1).split(',')]
            return numbers
        except ValueError:
            pass  # Skip to the next pattern if parsing fails

    # Case 3: Extract standalone numbers elsewhere in the text
    numbers = re.findall(r'\d+\.?\d*', raw_text)
    if numbers:
        return [float(num) for num in numbers]  # Return all numbers as a list

    # If no patterns match, return NaN
    return np.nan

def extract_digits_exp2(raw_text, model_name=None):
    """Extraction logic specific to EXP2."""
    if pd.isna(raw_text):
        return None

    raw_text = str(raw_text).strip().replace('\n', ' ')

    # Extract lists inside square brackets
    matches = re.findall(r'\[([\d.,\s]+)\]', raw_text)

    # Check for descriptive patterns if no matches
    if not matches:
        match_pattern = re.search(
            r'a reasonable estimate of the ratios would be:\s*\[([\d.,\s]+)\]', raw_text, re.IGNORECASE
        )
        if match_pattern:
            matches = [match_pattern.group(1)]

    if matches:
        valid_lists = []
        for match in matches:
            numbers = [float(num.strip()) for num in match.split(',') if re.match(r'^\d*\.?\d+$', num.strip())]
            if len(numbers) == 5:  # Only keep lists of exactly 5 numbers
                valid_lists.append(numbers)
        return valid_lists[-1] if valid_lists else None

    return None

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
                return float(f"{fraction_value:.2f}".rstrip('0').rstrip('.'))  # Round and format
            except ZeroDivisionError:
                return np.nan
        else:  # It's a decimal
            return float(last_match.rstrip('0').rstrip('.'))
    
    # Return NaN if no matches
    return np.nan

def extract_digits_exp4(raw_text, model_name):
    """Extraction logic specific to EXP4."""
    if pd.isna(raw_text):
        return None

    raw_text = str(raw_text).strip().replace('\n', '')

    # Split into sentences
    sentences = re.split(r'[.!?]\s+', raw_text)
    last_sentence = sentences[-1] if sentences else ""

    # Look for square-bracketed list
    match = re.search(r'\[([\d.,\s]+)\]', last_sentence)
    if match:
        return [float(num.strip()) for num in match.group(1).split(',')]

    # Handle "LLaMA" specific case
    if model_name == "LLaMA":
        digit_matches = re.findall(r'\b(\d+)\b', last_sentence)
        if len(digit_matches) >= 2:
            return [float(digit_matches[-2]), float(digit_matches[-1])]

   # Handle "both" keyword with focus on explicit values
    if "both" in last_sentence.lower():
        # Case 1: Check for bracketed format like [54, 54]
        # Pattern \[(\d+\.?\d*),\s*(\d+\.?\d*)\] will match and extract numbers from brackets
        bracketed_match = re.search(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', last_sentence)
        if bracketed_match:
            return [float(bracketed_match.group(1)), float(bracketed_match.group(2))]

        # Case 2: Check for natural language like "Both lengths of the bars are 54.0 pixels"
        # Pattern both.*(?:are|is|lengths).*?(\d+\.?\d*)\s*(?:pixels|units|bars|values)? 
        # will extract the number and return it twice
        both_match_single = re.search(
            r'both.*(?:are|is|lengths).*?(\d+\.?\d*)\s*(?:pixels|units|bars|values)?',
            last_sentence,
            re.IGNORECASE
        )
        if both_match_single:
            value = float(both_match_single.group(1))
            return [value, value]

    return None

# --- MLAE CALCULATION AND PLOTTING ---
def calculate_mlae_metrics(dataframes):
    """
    Calculates MLAE and MSE metrics for each dataset and summarizes results per model.

    Parameters:
    dataframes (dict): Dictionary of datasets keyed by dataset names.
                       Each value is a DataFrame containing 'ground_truth', 'parsed_answers', and 'model_name'.

    Returns:
    pd.DataFrame: A summary table of MLAE metrics for each model and dataset.
    """
    metrics_summary = {}

    for df_name, df in dataframes.items():
        print(f"Processing dataset: {df_name}")

        # Ensure `ground_truth` and `parsed_answers` are numeric arrays
        df['ground_truth'] = df['ground_truth'].apply(
            lambda x: np.array(x) if isinstance(x, list) else np.array(ast.literal_eval(x))
        )
        df['parsed_answers'] = df['parsed_answers'].apply(
            lambda x: np.array(x) if isinstance(x, list) else np.array(ast.literal_eval(x))
        )

        # Drop rows with NaN or mismatched shapes
        df = df.dropna(subset=['ground_truth', 'parsed_answers'])
        df = df[df['ground_truth'].apply(len) == df['parsed_answers'].apply(len)]

        print(f"  Remaining rows after cleanup: {len(df)}")

        # Skip processing if the dataset is empty
        if df.empty:
            print(f"  ⚠️ Dataset {df_name} has no valid rows. Skipping...")
            continue

        # Debug ground_truth and parsed_answers
        #print(df[['ground_truth', 'parsed_answers']].head())

        # Calculate MSE for each row
        df['mse'] = df.apply(
            lambda row: np.mean((row['ground_truth'] - row['parsed_answers']) ** 2),
            axis=1
        )

        # Safely calculate MLAE for each row
        def calculate_mlae(row):
            try:
                return np.log2(mean_absolute_error(row['ground_truth'], row['parsed_answers']) + 0.125)
            except Exception as e:
                print(f"Error in row {row.name}: {e}")
                return np.nan

        df['mlae'] = df.apply(calculate_mlae, axis=1)

        # Collect metrics per model
        model_metrics = {}
        for model_name, group in df.groupby('model_name'):
            avg_mlae = group['mlae'].mean()
            std_mlae = group['mlae'].std()

            # Bootstrap confidence interval for MLAE
            mlae_values = group['mlae'].dropna().values
            bootstrap_result = bs.bootstrap(np.array(mlae_values), stat_func=bs_stats.std)
            confidence_value = 1.96 * bootstrap_result.value  # 95% CI

            # Store metrics for the current model
            model_metrics[model_name] = {
                'Dataset': df_name,
                'Model': model_name,
                'Average MLAE': round(avg_mlae, 2),
                'Confidence Interval (95%)': round(confidence_value, 2)
            }

        # Store metrics for the current dataset
        metrics_summary[df_name] = model_metrics

    # Flatten the nested dictionary into a DataFrame
    metrics_table = pd.DataFrame([
        metrics for dataset_metrics in metrics_summary.values() for metrics in dataset_metrics.values()
    ])

    return metrics_table

# --- IMAGE TASK MATCHING ---
def find_task_images(image_dir, task_keywords):
    """
    Finds image paths in a directory that match specified task keywords.

    Parameters:
    image_dir (str): Directory to search for images.
    task_keywords (list): List of keywords to identify tasks (e.g., ['framed', 'unframed']).

    Returns:
    dict: A dictionary mapping task keywords to their corresponding image paths.
    """
    task_images = {}
    for keyword in task_keywords:
        matches = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if keyword in f]
        if matches:
            task_images[keyword] = matches[0]  # Take the first match for the keyword
        else:
            print(f"⚠️ No image found for task: {keyword}")
    return task_images


# --- MAIN PROCESSING FUNCTION ---
def process_and_plot_exp(experiment_type, config):
    """
    Processes and plots MLAE metrics for the specified experiment.

    Parameters:
    experiment_type (str): Identifier for the experiment ("EXP2" or "EXP4").
    config (str): Configuration key ("EXP2" or "EXP4") from the configs dictionary.
    """
    # Configuration for experiments
    configs = {
        
        "EXP1": {
            "file_path": "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/test/finetuning-EXP1numberone-test/EXP-Results/EXP1numberone_results.csv",
            "task_types": [
                'Shading', 'Curvature', 'Angle', 
                'Position\n(Non-Aligned Scale)', 'Position\n(Common Scale)', 
                'Length', 'Direction', 'Area', 'Volume'
            ],
            "extraction_logic": extract_digits_exp1,  # Use EXP1-specific logic
            "human_values": {
                'Angle': (3.22, 0.54),
                'Area': (3.64, 0.38),
                'Volume': (5.18, 0.40),
                'Curvature': (4.13, 0.10),
                'Shading': (4.22, 0.23),
                'Position\n(Common Scale)': (3.35, 0.49),
                'Position\n(Non-Aligned Scale)': (3.06, 0.74),
                'Length': (3.51, 0.44),
                'Direction': (3.75, 0.39)
            },
            "image_dir": "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/test/finetuning-EXP1numberone-test/images"
        },

        "EXP2": {
            "file_path": "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numberone/EXP-Results/EXP2results.csv",
            "task_types": ['pie', 'bar'],
            "extraction_logic": extract_digits_exp2,
            "human_values": {
                'pie': (2.05, 0.115),
                'bar': (1.035, 0.125)
            },
            "image_dir": "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numberone/images"
        },
        
        "EXP3": {
        "file_path": "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP3/finetuning-EXP3numberone/EXP-Results/EXP3numberone_results.csv",
        "task_types": ['type5', 'type4', 'type3', 'type2', 'type1'],
        "extraction_logic": extract_digits_exp3,
        "human_values": {
            'type5': (1.4, 0.14),
            'type4': (1.72, 0.2),
            'type3': (1.84, 0.16),
            'type2': (2.35, 0.175),
            'type1': (2.72, 0.155)
        },
        "image_dir": "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP3/finetuning-EXP3numberone/images"
        },

        "EXP4": {
            "file_path": "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP4/finetuning-EXP4numberone/EXP-Results/EXP4results.csv",
            "task_types": ['framed', 'unframed'],
            "extraction_logic": extract_digits_exp4,
            "human_values": {
                'framed': (3.371, 0.741),
                'unframed': (3.961, 0.454)
            },
            "image_dir": "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP4/finetuning-EXP4numberone/images"
        },


    }

    # Validate configuration
    if config not in configs:
        raise ValueError(f"Unknown config key: {config}")

    # Get experiment configuration
    experiment_config = configs[config]

    print(f"\n🚀 Processing {experiment_type} with config {config}...")

     # Load the dataset
    df = pd.read_csv(experiment_config["file_path"])

    # Ensure task_name is string type to match task_types
    df['task_name'] = df['task_name'].astype(str)

    # Clean raw answers from the CSV file
    task_data, deleted_rows = clean_raw_answers(
        experiment_config["file_path"],
        experiment_config["extraction_logic"],
        experiment_config["task_types"]
    )

    # Log deleted rows
    #if not deleted_rows.empty:
        #print(f"\n⚠️ Deleted Rows for {experiment_type}:")
        #print(deleted_rows[['task_name', 'raw_answer', 'parsed_answers']])

    # Find task images dynamically
    task_images = find_task_images(
        experiment_config["image_dir"],
        experiment_config["task_types"]
    )

    # Shared model colors
    model_colors = {
        'CustomLLaMA': '#8E44AD', 'Gemini1_5Flash': '#3498DB', 'GeminiProVision': '#E74C3C',
        'LLaMA': '#E67E22', 'gpt4o': '#27AE60', 'Human': '#2C3E50'
    }

    # Calculate and plot MLAE metrics
    metrics_table = calculate_mlae_metrics(task_data)
    
    plot_mlae(
        metrics_table,
        experiment_config["human_values"],
        task_images,
        model_colors,
        config
    )

def plot_mlae(metrics_table, human_values, task_images, model_colors, config):
    """
    Plots MLAE (Mean Log Absolute Error) with error bars for each model and compares them to human values.

    Parameters:
    metrics_table (pd.DataFrame): Table containing MLAE metrics for each task and model.
    human_values (dict): Dictionary of human MLAE values and error bars by task name.
    task_images (dict): Dictionary of task images paths by task name.
    model_colors (dict): Dictionary of colors for each model, including 'Human'.
    config (str): Configuration key (e.g., "EXP1", "EXP2", "EXP4").
    """
    # Determine x-axis limits from the experiment configuration
    if config == "EXP1":
        xlim = (-3, 3)  
    elif config == "EXP2":
        xlim = (-3, 3)  
    elif config == "EXP3":
        xlim = (-8, 8)  
    elif config == "EXP4":
        xlim = (-8, 8)


    else:
        xlim = (-8, 8)  # Default range

    # Group data by dataset for easier access
    summary_stats_by_task = {
        df_name: metrics_table[metrics_table['Dataset'] == df_name]
        for df_name in metrics_table['Dataset'].unique()
    }

    # Determine the number of tasks
    num_tasks = len(summary_stats_by_task)

     # Customize subplot layout based on the experiment configuration
    if config == "EXP1":
        figsize = (12, 2 * num_tasks)  # Larger height for EXP1
        width_ratios = [1, 6, 1]  # Wider main plot area for EXP1
    elif config == "EXP2":
        figsize = (12, 2 * num_tasks)  # Default height for EXP2
        width_ratios = [1, 5, 1]
    elif config == "EXP3":
        figsize = (12, 2 * num_tasks)  # Slightly larger for EXP4
        width_ratios = [1, 5, 1]
    elif config == "EXP4":
        figsize = (12, 2 * num_tasks)  # Slightly larger for EXP4
        width_ratios = [1, 5, 1]
    else:
        figsize = (12, 3 * num_tasks)  # Default fallback
        width_ratios = [1, 5, 1]

    # Create subplots
    fig, axes = plt.subplots(
        num_tasks, 3, figsize=figsize,
        gridspec_kw={'width_ratios': width_ratios}, sharex=False
    )

    fig.subplots_adjust(hspace=0.5, wspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.05)
    fig.patch.set_facecolor('white')

    # Ensure axes are a 2D array for consistent indexing
    if num_tasks == 1:
        axes = axes.reshape(1, -1)

    # Loop through tasks and plot MLAE
    for i, (task_name, task_data) in enumerate(summary_stats_by_task.items()):
        ax_img, ax_plot, ax_label = axes[i]

        # Display task image
        img_path = task_images.get(task_name)
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("L")
                img_inverted = ImageOps.invert(img)
                img_with_border = ImageOps.expand(img_inverted.convert("RGB"), border=1, fill="black")
                ax_img.imshow(img_with_border)
                ax_img.axis('off')
            except Exception as e:
                ax_img.text(0.5, 0.5, "Error displaying image", ha="center", va="center", fontsize=10, color="black")
                ax_img.set_facecolor("white")
        else:
            ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=10, color="black")
            ax_img.set_facecolor("white")

        # Sort model names and plot MLAE
        sorted_model_names = sorted(task_data['Model'].unique())
        y_positions = np.arange(len(sorted_model_names) + 1)  # +1 for Human

        for j, model_name in enumerate(sorted_model_names):
            model_data = task_data[task_data['Model'] == model_name]
            mlae_value = model_data['Average MLAE'].values[0]
            confidence_interval = model_data['Confidence Interval (95%)'].values[0]

            ax_plot.errorbar(
                mlae_value, j, xerr=confidence_interval, fmt='o',
                color=model_colors.get(model_name, 'gray'), capsize=5,
                label=f"{model_name}" if i == 0 else None
            )

        # Plot human MLAE
        if task_name in human_values:
            human_mlae, human_error = human_values[task_name]
            ax_plot.errorbar(
                human_mlae, len(sorted_model_names), xerr=human_error, fmt='s',
                color=model_colors['Human'], capsize=5, label="Human" if i == 0 else None
            )

        # Apply x-axis limits
        ax_plot.set_xlim(xlim)

        # Customize plot appearance
        ax_plot.axvline(0, color="gray", linestyle="--", linewidth=0.8)  # Zero reference line
        ax_plot.grid(False)
        ax_plot.spines['top'].set_visible(False)
        ax_plot.spines['right'].set_visible(False)
        ax_plot.spines['left'].set_visible(False)
        ax_plot.set_yticks(y_positions)
        ax_plot.tick_params(axis='y', which='major', length=0)  # Hide y-axis major ticks
        ax_plot.set_yticklabels([])
        ax_plot.invert_yaxis()

        # Set labels for the right-hand column
        ax_label.set_yticks(y_positions)
        ax_label.set_yticklabels(sorted_model_names + ["Human"], fontsize=10)
        ax_label.tick_params(left=False, right=False, labelleft=False, labelright=True)
        ax_label.set_ylim(ax_plot.get_ylim())
        ax_label.axis("off")

    # Add legend
    axes[0, 1].legend(loc="best", frameon=False)

    # Display the plot
    plt.tight_layout()
    plt.show()



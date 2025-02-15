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


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps



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
                break  # Stop after finding the first match

    return task_images

def plot_results(metrics_summary_df, base_path='/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/finetuning-EXP1numberfour/images'):
    """
    Plot the results from the metrics table with human benchmark values and task images.

    Parameters:
    metrics_summary_df (pandas.DataFrame): DataFrame containing model evaluation metrics.
    base_path (str): Path to the directory containing task images.
    """

    # Load task images
    task_images = find_task_images(base_path)

    # Group metrics by dataset
    summary_stats_by_task = {df_name: metrics_summary_df[metrics_summary_df['Dataset'] == df_name] 
                            for df_name in metrics_summary_df['Dataset'].unique()}

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

    # Custom order of tasks
    custom_task_order = [
        'position_common_scale', 'position_non_aligned_scale', 'length', 
        'direction', 'angle', 'area', 'volume', 'curvature', 'shading'
    ]

    # Reorder summary_stats_by_task based on the custom order
    summary_stats_by_task = {task: summary_stats_by_task[task] for task in custom_task_order if task in summary_stats_by_task}

    # Define model display names
    model_display_names = {
        'CustomLLaMA': 'Fine-tuned models',
        'Gemini1_5Flash': 'Gemini 1.5 Flash',
        'GeminiProVision': 'Gemini Pro Vision',
        'LLaMA': 'Llama 3.2 Vision',
        'gpt4o': 'GPT-4o Vision',
        'Human': 'Human'
    }

    # Define model colors
    model_colors = {
        'CustomLLaMA': '#8E44AD',   # Purple
        'Gemini1_5Flash': '#3498DB',    # Blue
        'GeminiProVision': '#E74C3C',   # Red
        'LLaMA': '#E67E22',             # Orange
        'gpt4o': '#27AE60',             # Green
        'Human': '#34495E'              # Dark Gray
    }

    # Define human benchmark MLAE values and error bars
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

    num_tasks = len(summary_stats_by_task)
    
    fig, axes = plt.subplots(num_tasks, 3, figsize=(8, 1.7 * num_tasks), 
                             gridspec_kw={'width_ratios': [1, 4, 1]}, sharex=False)

    fig.patch.set_facecolor('white')

    # Handle both single and multiple subplot cases
    if num_tasks == 1:
        axes = np.array([axes])  # Convert to 2D array with one row

    for i, (task_name, task_data) in enumerate(summary_stats_by_task.items()):
        ax_img, ax_plot, ax_label = axes[i]

        # Load and display task image if available
        ax_img.axis('off')
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

        ax_img.set_title(task_display_names.get(task_name, task_name), loc="left", fontsize=10, color="black", pad=10)

        sorted_model_names = sorted(task_data['Model'].unique())
        y_positions = np.arange(len(sorted_model_names))

        for j, model_name in enumerate(sorted_model_names):
            model_data = task_data[task_data['Model'] == model_name]
            mlae_value = model_data['Average MLAE'].values[0]
            confidence_interval = model_data['Confidence Interval (95%)'].values[0]

            ax_plot.errorbar(mlae_value, j, xerr=confidence_interval, fmt='o', 
                           color=model_colors.get(model_name, 'gray'), capsize=5, 
                           label=f"{model_name}" if i == 0 else None)

        # Plot human benchmark
        if task_name in human_data:
            human_value, human_std = human_data[task_name]
            y_pos = len(sorted_model_names) + 0.5
            ax_plot.errorbar(human_value, y_pos, xerr=human_std, 
                             fmt='s', color=model_colors['Human'], 
                             capsize=5, capthick=1.5,
                             markersize=7, label='Human' if i == 0 else None)

        ax_plot.axvline(-6, color="black", linewidth=1)
        ax_plot.axvline(20, color="black", linewidth=1)
        ax_plot.grid(False)

        ax_plot.spines['top'].set_visible(False)
        ax_plot.spines['right'].set_visible(False)
        ax_plot.spines['left'].set_visible(False)
        ax_plot.spines['bottom'].set_visible(i == num_tasks - 1)  # Hide bottom spine except for the last plot

        ax_plot.set_yticks([])
        ax_plot.set_yticklabels([])
        ax_plot.set_xlim(-6, 20)
        ax_plot.invert_yaxis()

        ax_label.set_yticks(y_positions)
        ax_label.set_yticklabels(sorted_model_names, fontsize=10)
        ax_label.tick_params(left=False, right=False, labelleft=False, labelright=True)
        ax_plot.tick_params(axis='x', which='both', bottom=i == num_tasks - 1, labelbottom=i == num_tasks - 1)
        ax_label.set_ylim(ax_plot.get_ylim())
        ax_label.axis("off")


    # Add the legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=model_colors[model_name], 
                   markersize=10, label=model_display_names[model_name])
        for model_name in model_colors.keys()
    ]
    axes[0, 1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.5, 0.5), frameon=False)


    plt.savefig("Figure2.png", bbox_inches='tight', dpi=300)
    plt.show()



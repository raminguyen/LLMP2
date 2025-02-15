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
            print(f"Loading image for {task_name}: {img_path}")  # Debugging step

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
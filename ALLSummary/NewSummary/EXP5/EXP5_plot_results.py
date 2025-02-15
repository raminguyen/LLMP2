import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def plot_results(metrics_table):
    """
    Plot the results from the metrics table with human benchmark values.
    """

    # Define human benchmark values
    human_data = {
        'Task_10': (4.0149, 0.5338),
        'Task_100': (5.3891, 0.1945),
        'Task_1000': (5.4612, 0.2509)
    }

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

    # Organize metrics by task
    summary_stats_by_task = {
        f"Task_{int(df_name)}": metrics_table[metrics_table['Dataset'] == df_name]
        for df_name in metrics_table['Dataset'].unique()
    }

    num_tasks = len(summary_stats_by_task)

    # Ensure subplots are properly structured
    fig, axes = plt.subplots(num_tasks, 3, figsize=(12, 2 * num_tasks),
                             gridspec_kw={'width_ratios': [1, 6, 1]}, sharex=False)
    
    fig.subplots_adjust(hspace=0.7, left=0.1, right=0.95, top=0.98)
    fig.patch.set_facecolor("white")  # Set overall figure background to white

    if num_tasks == 1:
        axes = np.array([axes])  # Convert single subplot to an array

    # **Manually assigned images**
    task_images = {
        'Task_10': "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberone/images/10_0a1dc7c9-beca-463e-a784-288f0266e8c3.jpg",
        'Task_100': "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberone/images/100_cd8b8d37-b35f-496d-a2ce-ce3ffbd5cfdd.jpg",
        'Task_1000': "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberone/images/1000_34fe41e8-42c4-41b0-ad4b-c717c86f51a9.jpg"
    }

    # Plot each task
    for i, (task_name, task_data) in enumerate(summary_stats_by_task.items()):
        ax_img, ax_plot, ax_label = axes[i]

        task_name_str = f"Task_{int(task_name.split('_')[1])}"  # Convert and match key format
        print(f"Looking for image key: {task_name_str}")

        img_path = task_images.get(task_name_str)

        if img_path and os.path.exists(img_path):
            print(f"✅ Found image for {task_name_str}: {img_path}")
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            img_inverted = ImageOps.invert(img)  # Invert colors (black dots, white background)
            ax_img.imshow(img_inverted, cmap="gray", interpolation="nearest")  # Show in black & white
            ax_img.set_facecolor("white")
        else:
            print(f"⚠️ Image not found for {task_name_str}")
            ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=10, color="black")
            ax_img.set_facecolor("white")

        ax_img.axis('off')
        ax_img.set_title(task_display_names.get(task_name_str, task_name_str), loc="left", fontsize=14, color="black")

        # Sorting models for better visualization
        sorted_model_names = sorted(task_data['Model'].unique())
        y_positions = np.arange(len(sorted_model_names))

        # Plot MLAE values for each model
        for j, model_name in enumerate(sorted_model_names):
            model_data = task_data[task_data['Model'] == model_name]
            mlae_value = model_data['Average MLAE'].values[0]
            confidence_interval = model_data['Confidence Interval (95%)'].values[0]

            ax_plot.errorbar(mlae_value, j, xerr=confidence_interval, fmt='o', 
                             color=model_colors.get(model_name, 'gray'), capsize=5, 
                             label=f"{model_name}" if i == 0 else None)

        # Plot human benchmark
        if task_name_str in human_data:
            human_value, human_std = human_data[task_name_str]
            y_pos = len(sorted_model_names) + 0.5

            ax_plot.errorbar(human_value, y_pos, xerr=human_std, 
                             fmt='s', color=model_colors['Human'], 
                             capsize=5, capthick=1.5,
                             markersize=7, label='Human' if i == 0 else None)

        # Reference lines
        ax_plot.axvline(-6, color="black", linewidth=2)
        ax_plot.axvline(12, color="black", linewidth=2)
        ax_plot.grid(False)

        # Blurred reference line at 0
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

    # Add the legend
    model_display_names = {
        'CustomLLaMA': 'Fine-tuned models',
        'Gemini1_5Flash': 'Gemini 1.5 Flash',
        'GeminiProVision': 'Gemini Pro Vision',
        'LLaMA': 'Llama 3.2 Vision',
        'gpt4o': 'GPT-4o Vision',
        'Human': 'Human'
    }

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=model_colors[model_name], 
                   markersize=10, label=model_display_names[model_name])
        for model_name in model_colors.keys()
    ]
    axes[0, 1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.7, 0.5), frameon=False)

    plt.savefig("Figure6.png", bbox_inches='tight', dpi=300)
    plt.show()

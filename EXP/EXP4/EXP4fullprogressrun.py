from EXP4fullprogress import Runexp4

import pandas as pd
import sys

def run_exp(exp, num_images=55):
    """
    Run the specified experiment with a given number of images.
    """
    base_path = "/hpcstor6/scratch01/h/huuthanhvy.nguyen"
    csv_filename = "experiment_results.csv"

    # Configurations for each experiment
    experiment_configs = {
        1: {
            "original_dataset_path": "path/to/original_dataset_exp1.csv",
            "exp_folders": [
                "finetuning-EXP1numberone",
                "finetuning-EXP1numbertwo",
                "finetuning-EXP1numberthree"
            ],
            "run_function": Runexp1
        },
        2: {
            "original_dataset_path": "path/to/original_dataset_exp2.csv",
            "exp_folders": [
                "finetuning-EXP2numberone",
                "finetuning-EXP2numbertwo",
                "finetuning-EXP2numberthree"
            ],
            "run_function": Runexp2
        },
        3: {
            "original_dataset_path": "path/to/original_dataset_exp3.csv",
            "exp_folders": [
                "finetuning-EXP3numberone",
                "finetuning-EXP3numbertwo",
                "finetuning-EXP3numberthree"
            ],
            "run_function": Runexp3
        },
        4: {
            "original_dataset_path": "path/to/original_dataset.csv",
            "exp_folders": [
                "finetuning-EXP4numberone",
                "finetuning-EXP4numbertwo",
                "finetuning-EXP4numberthree"
            ],
            "run_function": Runexp4
        },
        5: {
            "original_dataset_path": "path/to/original_dataset_exp5.csv",
            "exp_folders": [
                "finetuning-EXP5numberone",
                "finetuning-EXP5numbertwo",
                "finetuning-EXP5numberthree"
            ],
            "run_function": Runexp5
        }
    }

    # Get experiment configuration
    config = experiment_configs.get(exp)
    if not config:
        print(f"Invalid experiment: {exp}")
        return

    # Load and filter dataset
    print(f"Running EXP{exp} with {num_images} images...")
    df = pd.read_csv(config["original_dataset_path"])
    df["task"] = df["task"].str.strip()
    task_groups = df.groupby("task")
    images_per_task = num_images // len(task_groups)
    extra_images = num_images % len(task_groups)
    selected_data = pd.concat([
        group.sample(n=min(len(group), images_per_task + (1 if extra_images > 0 else 0)), random_state=42)
        for task_name, group in task_groups
    ])
    filtered_dataset_path = f"{base_path}/filtered_test_dataset_exp{exp}.csv"
    selected_data.to_csv(filtered_dataset_path, index=False)
    print(f"Filtered dataset saved at {filtered_dataset_path}")

    # Run experiments
    for folder in config["exp_folders"]:
        model_dir = f"{base_path}/{folder}/fine_tuned_model"
        image_base_dir = f"{base_path}/{folder}/images"
        results_dir = f"{base_path}/{folder}"
        config["run_function"](filtered_dataset_path, results_dir, model_dir, csv_filename, image_base_dir)

    print(f"EXP{exp} completed!")

if __name__ == "__main__":

    # Check command-line arguments
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        print("Usage: python EXP4fullprogress.py <experiment_number> [num_images]")
    else:
        exp = int(sys.argv[1])
        num_images = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 55
        run_exp(exp, num_images)

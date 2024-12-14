import torch
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
# import bootstrapped.bootstrap as bs
# import bootstrapped.stats_functions as bs_stats
import pandas as pd
import os
from PIL import Image

class Evaluator5:
    
    def __init__(self):
        self.results = None
    
    @staticmethod
    def calculate_mse(gt, answers):
        """Calculate mse."""
        gt_array = np.array(gt).flatten()  # Flatten to ensure 1D array
        answers_array = np.array(answers).flatten()  # Flatten to ensure 1D array
        return mean_squared_error(gt_array, answers_array)

    @staticmethod
    def calculate_mlae(gt, answers):
        """Calculate mlae."""
        gt_array = np.array(gt).flatten()  # Flatten to ensure 1D array
        answers_array = np.array(answers).flatten()  # Flatten to ensure 1D array
        mlae = np.log2(mean_absolute_error(gt_array, answers_array) + 0.125)
        return mlae

    @staticmethod
    def calculate_mean(answers):
        """Calculate mean."""
        return np.mean(answers)

    @staticmethod
    def calculate_std(answers):
        """Calculate std."""
        return np.std(answers)

    def run(self, data, query, models):
        """Run experiments."""
        results = {'gt': [d[1] for d in data], 'image_path': [d[0] for d in data]}  # Capture ground truth and image paths

        for model_name, model_instance in models.items():
            results[model_name] = {}
            mlae_list = []
            
            for i in range(3):  # Repeat each experiment 3 times
                raw_answers = []
                forced_repetitions = 0
                times = []

                for image_path, ground_truth in data:
                    torch.cuda.empty_cache()
                    FLAG = False
                    start_time = time.time()

                    # Load the image from the path
                    image = np.array(Image.open(image_path).convert("L"))  # Convert to grayscale

                    while not FLAG:
                        answer = model_instance.query(query, image)

                        # Check if answer is None or has length
                        if answer is not None and len(answer) > 0:
                            raw_answers.append(answer)
                            FLAG = True
                            end_time = time.time()
                            times.append((end_time - start_time) * 1000)
                        else:
                            forced_repetitions += 1
                            # Optionally, log or print an error message for debugging
                            print(f"Warning: Model returned None or empty answer for image {image_path}. Attempting again.")

                # Adjusted midpoint calculation without parsed answers
                midpoints = [0 for _ in data]

                gt_flat = [item for sublist in results['gt'] for item in (sublist if isinstance(sublist, list) else [sublist])]
                midpoints_flat = (midpoints * (len(gt_flat) // len(midpoints) + 1))[:len(gt_flat)]

                # Commented out mean, mse, and mlae calculations
                # mse = Evaluator5.calculate_mse(gt_flat, midpoints_flat)
                # mlae = Evaluator5.calculate_mlae(gt_flat, midpoints_flat)
                # mean = Evaluator5.calculate_mean(midpoints_flat)

                # mlae_list.append(mlae)

                results[model_name][f"run_{i}"] = {
                    'raw_answers': raw_answers,
                    'times': times,
                    'forced_repetitions': forced_repetitions
                }

            # Commented out aggregated metrics
            # results[model_name]['average_mlae'] = Evaluator5.calculate_mean(mlae_list)
            # results[model_name]['std'] = Evaluator5.calculate_std(mlae_list)
            # results[model_name]['confidence'] = 1.96 * bs.bootstrap(np.array(mlae_list), stat_func=bs_stats.std).value
            
        self.results = results

        return self.results

    def save_results_csv(self, filename="results.csv"):
        """Transform all results for all tasks into a single DataFrame and save as a CSV in EXP5-Results folder."""
        if self.results is None:
            raise ValueError("No results found. Run the 'run' method first.")
        
        # Prepare data to store
        data = []

        for model_name, model_data in self.results.items():
            if model_name in ['gt', 'image_path']:  # Skip ground truth and image path keys for now
                continue

            for run_key, run_data in model_data.items():
                if run_key.startswith("run_"):
                    for idx, time in enumerate(run_data['times']):
                        # Gather data from each run, including image path, raw answers, and ground truth
                        data.append({
                            'model_name': model_name,
                            'run': run_key,
                            'image_path': self.results['image_path'][idx],  # Image path for each run
                            'ground_truth': self.results['gt'][idx],       # Ground truth for each run
                            'raw_answers': run_data['raw_answers'][idx],
                            'forced_repetitions': run_data['forced_repetitions'],
                            'time_ms': time
                        })

            # Add aggregated metrics, excluding std and confidence
            data.append({
                'model_name': model_name,
                'run': 'average',
                'image_path': None,
                'ground_truth': None,
                'raw_answers': None,
                'forced_repetitions': None,
                'time_ms': None,
            })

        # Convert all collected data into a single DataFrame
        df = pd.DataFrame(data)
        
        # Define the directory path for saving results
        results_folder = os.path.join(os.getcwd(), "EXP5-Results")
        os.makedirs(results_folder, exist_ok=True)  # Ensure the results folder exists

        # Define the file path for saving results in the specified folder
        file_path = os.path.join(results_folder, filename)

        # Save the DataFrame to a single CSV file in the specified folder
        df.to_csv(file_path, index=False)
        
        # Return the DataFrame as well
        return df

import torch
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import os
from PIL import Image
import LLMP as L

class Evaluator3:

    def __init__(self):
        self.results = None

    @staticmethod
    def calculate_mse(gt, answers):
        gt_array = np.array(gt).flatten()
        answers_array = np.array(answers).flatten()
        return mean_squared_error(gt_array, answers_array)

    @staticmethod
    def calculate_mlae(gt, answers):
        gt_array = np.array(gt).flatten()
        answers_array = np.array(answers).flatten()
        mlae = np.log2(mean_absolute_error(gt_array, answers_array) + 0.125)
        return mlae

    @staticmethod
    def calculate_mean(answers):
        return np.mean(answers)

    @staticmethod
    def calculate_std(answers):
        return np.std(answers)

    def run(self, data, query, models):
        """Run experiments without parsing answers."""
        results = {'gt': [d[1] for d in data], 'image_path': [d[0] for d in data]}

        for model_name, model_instance in models.items():
            results[model_name] = {}
            mlae_list = []

            for i in range(3):  # Repeat each experiment 3 times
                raw_answers = []
                parsed_answers = []
                times = []

                for image_path, ground_truth in data:
                    torch.cuda.empty_cache()
                    start_time = time.time()

                    # Load image as grayscale
                    image = np.array(Image.open(image_path).convert("L"))

                    # Query the model once
                    answer = model_instance.query(query, image)
                    raw_answers.append(answer)
                    
                    # Use a default value or empty list for parsed answers
                    parsed_answers.append([0])  # Default placeholder value

                    # Record time taken for each query
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)

                if not parsed_answers:
                    parsed_answers = [[0] for _ in data]

                midpoints = [(sum(sublist) / 2) if len(sublist) > 1 else sublist[0] for sublist in parsed_answers]
                gt_flat = [item for sublist in results['gt'] for item in (sublist if isinstance(sublist, list) else [sublist])]
                midpoints_flat = (midpoints * (len(gt_flat) // len(midpoints) + 1))[:len(gt_flat)]

                mlae_list.append(None)  # Placeholder to maintain structure

                results[model_name][f"run_{i}"] = {
                    'raw_answers': raw_answers,
                    'parsed_answers': parsed_answers,
                    'mean': None,  # Commented out
                    'mse': None,   # Commented out
                    'mlae': None,  # Commented out
                    'times': times,
                }

            results[model_name]['average_mlae'] = None  # Commented out
            results[model_name]['std'] = None  # Commented out

        self.results = results
        return self.results

    def save_results_csv(self, filename="results.csv"):
        """Save all results to a CSV file in the results folder."""
        if self.results is None:
            raise ValueError("No results found. Run the 'run' method first.")
    
        data = []

        for model_name, model_data in self.results.items():
            if model_name in ['gt', 'image_path']:
                continue

            for run_key, run_data in model_data.items():
                if run_key.startswith("run_"):
                    for idx, time in enumerate(run_data['times']):
                        data.append({
                            'model_name': model_name,
                            'run': run_key,
                            'image_path': self.results['image_path'][idx],
                            'ground_truth': self.results['gt'][idx],
                            'raw_answers': run_data['raw_answers'][idx],
                            # Uncomment these lines if you need them
                            # 'parsed_answers': run_data['parsed_answers'][idx],
                            # 'mean': run_data.get('mean'),
                            # 'mse': run_data.get('mse'),
                            # 'mlae': run_data.get('mlae'),
                            # 'time_ms': time,
                            'forced_repetitions': run_data.get('forced_repetitions', 0)  # Default to 0 if not found
                        })

            # Add average or summary statistics row if needed
            data.append({
                'model_name': model_name,
                'run': 'average',
                'image_path': None,
                'ground_truth': None,
                'raw_answers': None,
                # Uncomment these lines if you need them
                # 'parsed_answers': None,
                # 'mean': model_data.get('average_mean'),
                # 'mse': model_data.get('average_mse'),
                # 'mlae': model_data.get('average_mlae'),
                # 'time_ms': None,
                # 'std': model_data.get('std'),
                'forced_repetitions': model_data.get('forced_repetitions', 0)  # Add forced repetitions for the average row
            })

        # Ensure the output folder exists
        output_folder = os.path.join(os.getcwd(), "EXP3-Results")
        os.makedirs(output_folder, exist_ok=True)
        
        file_path = os.path.join(output_folder, filename)
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

        return df

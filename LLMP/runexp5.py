import os
import time
import numpy as np
from PIL import Image
import LLMP as L

class Runexp5:
    def __init__(self):
        # Define the tasks and queries here
        self.tasks = [
            "10",
            "100",
            "1000"
        ]
        
        self.queries = {
            "10": "Please estimate how many dots were added to the initial 10 dots. Answer range 1 to 10. Number only. No explanation.",
            "100": "Please estimate how many dots were added to the initial 100 dots. Answer range 1 to 10. Number only. No explanation.",
            "1000": "Please estimate how many dots were added to the initial 1000 dots. Answer range 1 to 10. Number only. No explanation."
        }

        # Set the folder for saving results
        self.folder_name = "EXP5-Results"
        self.ensure_folder_exists(self.folder_name)

    def ensure_folder_exists(self, folder_name):
        """Ensure the results folder exists."""
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def run_experiment(self, task_name, query, num_images, model_instances):
        e = L.Evaluator5()

        # Create a subfolder for the current task images
        task_folder = os.path.join(self.folder_name, task_name)
        self.ensure_folder_exists(task_folder)

        image_paths = []
        ground_truth_values = []

        # Generate and save images for the task
        for i in range(num_images):
            image_data = L.GPImage.weber(task_name)
            image_array, ground_truth = image_data

            # Convert the array to grayscale
            size = image_array.shape[0]
            grayscale = np.zeros((size, size), dtype=np.uint8)
            grayscale[image_array == 1] = 255

            # Save the grayscale image and record its path
            image_filename = os.path.join(task_folder, f"{task_name}_image_{i+1}.png")
            Image.fromarray(grayscale).save(image_filename)

            image_paths.append(image_filename)
            ground_truth_values.append(ground_truth)

        # Start time measurement
        start_time = time.time()
        
        # Run the evaluator
        data = list(zip(image_paths, ground_truth_values))
        result = e.run(data, query, model_instances)
        e.save_results_csv(filename=f"{task_name}_results.csv")

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Log elapsed time
        log_filename = os.path.join(self.folder_name, "log.txt")
        with open(log_filename, 'a') as log_file:
            log_file.write(f"{task_name} - Elapsed time: {elapsed_time:.2f} seconds\n")

        return result

    def Runexp5(self, num_images, model_instances, tasks=None):
        if tasks:
            if isinstance(tasks, str):
                tasks = [tasks]
            for task in tasks:
                if task in self.tasks:
                    query = self.queries[task]
                    self.run_experiment(task, query, num_images, model_instances)
                else:
                    print(f"Task '{task}' is not defined.")
        else:
            for task in self.tasks:
                query = self.queries[task]
                print(f"Start '{task}'")
                self.run_experiment(task, query, num_images, model_instances)

        print("Done")

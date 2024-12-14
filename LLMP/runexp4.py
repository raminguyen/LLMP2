import os
import time
import numpy as np
from PIL import Image
import LLMP as L

class Runexp4: ##RENAME
    def __init__(self):
        # Define the tasks and their queries
        self.tasks = ['framed', 'unframed']
        
        self.queries = {
        
        "framed": "Estimate the lengths of the two bars with framing. Both lengths should fall between 49 and 60 pixels. No explanation. Format of the answer [xx, xx]",

        "unframed": "Estimate the lengths of the two bars without framing. Both lengths should fall between 49 and 60 pixels. No explanation. Format of the answer [xx, xx]",
        }

        # Set the folder for saving results
        self.folder_name = "EXP4-Results"
        self.ensure_folder_exists(self.folder_name)

    def ensure_folder_exists(self, folder_name):
        """Ensure the results folder exists."""
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def run_experiment(self, task_name, query, num_images, model_instances):
        """Run the experiment for a given task and save the results."""
        
        e = L.Evaluator4()
        
        # Create a subfolder for the current task images
        task_folder = os.path.join(self.folder_name, task_name)
        self.ensure_folder_exists(task_folder)

        image_paths = []
        ground_truth_values = []

        print(f"Starting experiment for task: '{task_name}' with {num_images} images.")

        # Generate and save images for the task
        for i in range(num_images):
            print(f"Generating image {i+1}/{num_images} for task '{task_name}'")
            image_data = L.GPImage.figure12(task_name)  #RENAME
            image_array, ground_truth = image_data

            # Convert the array to 
            size = image_array.shape[0]
            grayscale = np.zeros((size, size), dtype=np.uint8)
            grayscale[image_array == 1] = 255

            # Save the grayscale image and record its path
            image_filename = os.path.join(task_folder, f"{task_name}_image_{i+1}.png")
            Image.fromarray(grayscale).save(image_filename)

            image_paths.append(image_filename)
            ground_truth_values.append(ground_truth)

        print(f"Running evaluation for task '{task_name}'")

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

        print(f"Completed task '{task_name}' in {elapsed_time:.2f} seconds")

        return result

    def Runexp4(self, num_images, model_instances, tasks=None): ##RENAME
        """Run experiments for specified tasks, or all tasks if none specified."""
        if tasks:
            if isinstance(tasks, str):
                tasks = [tasks]  # Convert a single task string to a list
            for task in tasks:
                if task in self.tasks:
                    query = self.queries[task]
                    print(f"Starting '{task}' task")
                    self.run_experiment(task, query, num_images, model_instances)
                else:
                    print(f"Task '{task}' is not defined.")
        else:
            # Run all tasks if none specified
            for task in self.tasks:
                query = self.queries[task]
                print(f"Starting '{task}' task")
                self.run_experiment(task, query, num_images, model_instances)

        print("All tasks completed.")

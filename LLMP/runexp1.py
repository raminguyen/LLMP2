import os
import time
import numpy as np
from PIL import Image
import LLMP as L

class Runexp1:
    def __init__(self):
        # Define the tasks and queries here
        self.tasks = [
            "position_common_scale",
            "position_non_aligned_scale",
            "length",
            "direction",
            "angle",
            "area",
            "volume",
            "curvature",
            "shading"
        ]
        
        self.queries = {
            "position_common_scale": "Estimate the block's vertical position (range: 0-60, top to bottom). Number only. No explanation.",
            "position_non_aligned_scale": "Estimate the block's vertical position (range: 0-60, top to bottom). Number only. No explanation.",
            "length": "Estimate the line length from top to bottom (range: 0-100). Number only. No explanation.",
            "direction": "Estimate the line's direction (range: 0-359 degrees). Number only. No explanation.",
            "angle": "Estimate the angle (range: 0-90 degrees). Number only. No explanation.",
            "area": "Estimate the area of a circle, ensuring your answer falls within the range of 3.14 to 5026.55 square units. Assume the circle fits within a 100x100 pixel image. Provide only the numeric value, no explanation.",
            "volume": "Estimate the volume of a cube, with your answer restricted to the range of 1 to 8000 cubic units. Assume the cube fits within a 100x100 pixel image. Provide only the numeric value, no explanation.",
            "curvature": "Estimate the line curvature (range: 0.000 to 0.088) of a Bezier curve constrained within a 100x100 pixel space. Provide only the numeric curvature value (up to 3 decimal places), no explanation.",
            "shading": "Estimate shading density (range: 0-100). Number only. No explanation."
}


        # Set the folder for saving results
        self.folder_name = "EXP1-Results"
        self.ensure_folder_exists(self.folder_name)

    def ensure_folder_exists(self, folder_name):
        """Ensure the results folder exists."""
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def run_experiment(self, task_name, query, num_images, model_instances):
        e = L.Evaluator()

        # Create a subfolder for the current task imagess
        task_folder = os.path.join(self.folder_name, task_name)
        self.ensure_folder_exists(task_folder)

        image_paths = []
        ground_truth_values = []

        # Generate and save images for the task
        for i in range(num_images):
            image_data = L.GPImage.figure1(task_name)
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

    def Runexp1(self, num_images, model_instances, tasks=None):
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
# Add LLMP path
from EXP1whiteblackground import generate_images_for_multiple_angles, display_images_combined_by_degree, Runexp1, prepare_image_data
import LLMP as L
import os
from EXP1whiteblackground import clean_experiment_data, calculate_row_mlae, plot_mlae_heatmap, average_mlae_and_visualize, plot_vectorized_predictions_scatter, display_images_combined_by_degree, generate_images_for_multiple_angles, analyze_best_model_by_image_type, analyze_image_type_per_model, plot_mlae_per_image

generate_images_for_multiple_angles(canvas_size=100, line_length=30, num_angles=55)

# Use the absolute path for the image directory
image_dir = "/home/huuthanhvy.nguyen001/tmp/LLMP/EXP/generated_images"

# Ensure the directory exists
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

# Prepare image data
data = prepare_image_data(image_dir)

# Define model instances
model_instances = {
    "gpt4o": L.GPTModel("gpt-4o"),
    "LLaMA": L.llama("meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "GeminiProVision": L.GeminiProVision(),
    "Gemini1_5Flash": L.Gemini1_5Flash()
}

# Create Runexp1 instance and run the experiment
experiment = Runexp1()

experiment.run_angle_experiment(data=data, model_instances=model_instances)

import sys
from EXP5fullprogress import evaluateexp5
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='experiment_output.log',  # Save logs to this file
    level=logging.INFO,  # Log level (INFO will capture info-level logs and higher)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_experiment(exp_number, start_time, end_time, duration):
    log_message = (
        f"Experiment {exp_number} - Start: {start_time}, End: {end_time}, Duration: {duration:.2f} seconds\n"
    )
    logging.info(log_message)  # Log experiment details



def log_duration(start_time, end_time):
    duration = end_time - start_time
    return duration

def main(exp_number):
    # Define paths based on the experiment number
    base_path = '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/'

    if exp_number == 1:
        folder = 'finetuning-EXP5numberone'
    elif exp_number == 2:
        folder = 'finetuning-EXP5numbertwo'
    elif exp_number == 3:
        folder = 'finetuning-EXP5numberthree'
    else:
        print("Invalid experiment number! Use 1, 2, or 3.")
        return

    # Define the paths
    test_dataset_path = f'{base_path}{folder}/json/test_dataset.json'
    results_dir = f'{base_path}{folder}/EXP-Results'
    model_dir = f'{base_path}{folder}/fine_tuned_model'
    csv_filename = 'EXP5results55images.csv'
    image_base_dir = f'{base_path}{folder}/images'

        # Log experiment start
    logging.info(f"Starting Experiment {exp_number}...")

    # Call the evaluate function and log any output
    try:
        evaluateexp5(test_dataset_path, results_dir, model_dir, csv_filename, image_base_dir)
        logging.info("Evaluation completed successfully.")
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    # Capture the start time
    start_time = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if an argument is provided
    if len(sys.argv) != 2:
        logging.error("Usage: python EXPevaluate.py <experiment_number>")
        sys.exit(1)
    
    # Get the experiment number from the command line argument
    try:
        exp_number = int(sys.argv[1])  # Argument should be an integer (1, 2, or 3)
    except ValueError:
        logging.error("Error: Experiment number should be an integer (1, 2, or 3).")
        sys.exit(1)

    # Run the main function with the experiment number
    main(exp_number)

    # Capture the end time and log the duration
    end_time = time.time()
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = log_duration(start_time, end_time)
    
    # Log experiment duration
    log_experiment(exp_number, start_time_str, end_time_str, duration)

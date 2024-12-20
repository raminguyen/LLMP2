import sys
from EXP1fullprogress import evaluateEXP1

def main(exp_number):
    # Define paths based on the experiment number
    base_path = '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP1/'

    if exp_number == 1:
        folder = 'finetuning-EXP1numberone'
    elif exp_number == 2:
        folder = 'finetuning-EXP1numbertwo'
    elif exp_number == 3:
        folder = 'finetuning-EXP1numberthree'
    if exp_number == 4:
        folder = 'finetuning-EXP1numberfour'
    elif exp_number == 5:
        folder = 'finetuning-EXP1numberfive'
    elif exp_number == 6:
        folder = 'finetuning-EXP1numbersix'
    elif exp_number == 7:
        folder = 'finetuning-EXP1numbersix'
    else:
        print("Invalid experiment number! Use 1, 2, 3, 4, 5, or 6.")
        return

    # Define the paths
    test_dataset_path = f'{base_path}{folder}/json/test_dataset.json'
    results_dir = f'{base_path}{folder}/EXP-Results'
    model_dir = f'{base_path}{folder}/fine_tuned_model'
    csv_filename = 'EXP1results55images.csv'
    image_base_dir = f'{base_path}{folder}/images'

    # Call the evaluate function
    evaluateEXP1(test_dataset_path, results_dir, model_dir, csv_filename, image_base_dir)

if __name__ == "__main__":
    # Check if an argument is provided
    if len(sys.argv) != 2:
        print("Usage: python EXPevaluate.py <experiment_number>")
        sys.exit(1)
    
    # Get the experiment number from the command line argument
    try:
        exp_number = int(sys.argv[1])  # Argument should be an integer (1, 2, or 3)
    except ValueError:
        print("Error: Experiment number should be an integer (1, 2, or 3).")
        sys.exit(1)

    # Run the main function with the experiment number
    main(exp_number)

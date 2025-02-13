import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics

# Load dataset (assuming "angle_df.csv" is the actual file)
angle_df = pd.read_csv("angle_df")

def calculate_mlae_without_scaling(y_test, y_pred):
    """Computes different variations of MLAE."""
    mlae_1 = np.log2(sklearn.metrics.mean_absolute_error(y_pred, y_test) + 0.125)  # Original
    mlae_2 = np.mean(np.log2(np.abs(y_pred - y_test) + 0.125))  # New
    mlae_3 = np.log2(sklearn.metrics.mean_absolute_error(y_pred, y_test) + 0.125)  # Nguyen 2025
    mlae_4 = np.log2(sklearn.metrics.mean_absolute_error(y_pred, y_test) + 0.125)  # Haehn 2018
    
    # Print MLAE results
    print(f"MLAE (Original): {mlae_1:.4f}")
    print(f"MLAE_NEW (Log Before Avg): {mlae_2:.4f}")
    print(f"MLAE_RAMI (Real Scale): {mlae_3:.4f}")
    print(f"MLAE_OLD (Haehn 2018): {mlae_4:.4f}")

    return mlae_1, mlae_2, mlae_3, mlae_4

def calculate_mlae_per_prediction(y_test, y_pred):
    """Computes per-prediction MLAE values and logs intermediate computations."""
    
    abs_errors = np.abs(y_pred - y_test)
    abs_errors_real = np.abs(y_pred - y_test)

    # Compute logs separately for debugging
    mlae_1 = np.log2(sklearn.metrics.mean_absolute_error(y_pred, y_test) + 0.125)  # Original
    mlae_2 = np.log2(abs_errors + 0.125)  # New (Per-prediction log)
    mlae_3 = np.log2(sklearn.metrics.mean_absolute_error(y_pred, y_test) + 0.125)  # Nguyen 2025
    mlae_4 = np.log2(sklearn.metrics.mean_absolute_error(y_pred, y_test) + 0.125)  # Haehn 2018

    # Create per-prediction MLAE arrays
    mlae_1_per_pred = np.full_like(abs_errors, mlae_1)
    mlae_2_per_pred = mlae_2  # Already per-prediction
    mlae_3_per_pred = np.full_like(abs_errors_real, mlae_3)
    mlae_4_per_pred = np.full_like(abs_errors_real, mlae_4)

    return (mlae_1_per_pred, mlae_2_per_pred, mlae_3_per_pred, mlae_4_per_pred, 
            abs_errors, abs_errors_real)

def process_model(model_name):
    """Extracts and processes data for a specific model."""

    # Filter data for the specific model
    model_data = angle_df[angle_df["model_name"] == model_name]
    y_test = model_data["ground_truth"].to_numpy()
    y_pred = model_data["cleaned_answers"].to_numpy()

    # Compute MLAE values
    mlae_1, mlae_2, mlae_3, mlae_4 = calculate_mlae_without_scaling(y_test, y_pred)
    
    # Compute per-prediction MLAE and absolute errors, including log values
    results = calculate_mlae_per_prediction(y_test, y_pred)
    mlae_1_per_pred, mlae_2_per_pred, mlae_3_per_pred, mlae_4_per_pred, abs_errors, abs_errors_real = results

    

    return (
    (print("Absolute Errors:", abs_errors), abs_errors)[1],         # Absolute errors (difference between predictions and ground truth)
    (print("Absolute Errors (Real Scale):", abs_errors_real), abs_errors_real)[1],    # Absolute errors (real scale, without transformation)
    (print("MLAE_Orginal per prediction (Original):", mlae_1_per_pred), mlae_1_per_pred)[1],    # MLAE (Original) per prediction
    (print("MLAE_NEW per prediction (Log Before Avg):", mlae_2_per_pred), mlae_2_per_pred)[1],    # MLAE_NEW (Log Before Avg) per prediction
    (print("MLAE_RAMI per prediction (Real Scale):", mlae_3_per_pred), mlae_3_per_pred)[1],    # MLAE_RAMI (Real Scale) per prediction
    (print("MLAE_OLD per prediction (Haehn 2018):", mlae_4_per_pred), mlae_4_per_pred)[1],    # MLAE_OLD (Haehn 2018) per prediction
)


def plot_mlae_logs(model_name):
    """Plots log values for different MLAE calculations."""

    # Extract data from the model
    abs_errors, abs_errors_real, mlae_1_per_pred, mlae_2_per_pred, mlae_3_per_pred, mlae_4_per_pred = process_model(model_name)

    # Expand scalar log values to match abs_errors shape
    mlae_1_per_pred = np.full_like(abs_errors, mlae_1_per_pred)  # Fill array with the same scalar
    mlae_3_per_pred = np.full_like(abs_errors_real, mlae_3_per_pred)
    mlae_4_per_pred = np.full_like(abs_errors_real, mlae_4_per_pred)

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot each MLAE log
    plt.scatter(abs_errors, mlae_1_per_pred, marker='o', label="Log MLAE (Original, Scaled)", color='blue')
    plt.scatter(abs_errors, mlae_2_per_pred, marker='s', label="Log MLAE_NEW (Log Before Avg)", color='red')
    plt.scatter(abs_errors_real, mlae_3_per_pred, marker='^', label="Log MLAE_RAMI (Real Scale)", color='green')
    plt.scatter(abs_errors_real, mlae_4_per_pred, marker='x', label="Log MLAE_OLD (Haehn 2018)", color='purple')

    # Labels and Title
    plt.xlabel("Absolute Error")
    plt.title(f"Log Values for Different MLAE Calculations ({model_name})")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

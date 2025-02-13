import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics

class MLAEAnalyzer:
    def __init__(self, dataset_path):
        self.angle_dataframe = pd.read_csv(dataset_path)  # Ensure correct file extension

    @staticmethod
    def calculate_mlae(y_test, y_pred):
        """Computes different variations of MLAE."""
        mlae_1 = np.log2(sklearn.metrics.mean_absolute_error(np.multiply(y_pred, 100), np.multiply(y_test, 100)) + 0.125)
        mlae_2 = np.mean(np.log2(np.abs(np.multiply(y_pred, 100) - np.multiply(y_test, 100)) + 0.125))
        mlae_3 = np.log2(sklearn.metrics.mean_absolute_error(y_pred, y_test) + 0.125)  # Nguyen 2025
        mlae_4 = np.log2(sklearn.metrics.mean_absolute_error(y_pred, y_test) + 0.125)  # Haehn 2018

        return mlae_1, mlae_2, mlae_3, mlae_4

    @staticmethod
    def calculate_mlae_per_prediction(y_test, y_pred):
        """Computes per-prediction MLAE values."""
        abs_errors_100 = np.abs(np.multiply(y_pred, 100) - np.multiply(y_test, 100))
        abs_errors_100 = np.abs(np.multiply(y_pred, 100) - np.multiply(y_test, 100))


        mlae_1_per_pred = np.full_like(abs_errors_100, np.log2(sklearn.metrics.mean_absolute_error(np.multiply(y_pred, 100), np.multiply(y_test, 100)) + 0.125))
        mlae_2_per_pred = np.log2(abs_errors_100 + 0.125)
        mlae_3_per_pred = np.full_like(abs_errors_real, np.log2(sklearn.metrics.mean_absolute_error(y_pred, y_test) + 0.125))
        mlae_4_per_pred = np.full_like(abs_errors_real, np.log2(sklearn.metrics.mean_absolute_error(y_pred, y_test) + 0.125))

        return mlae_1_per_pred, mlae_2_per_pred, mlae_3_per_pred, mlae_4_per_pred, abs_errors_100, abs_errors_real

    def process_model(self, model_name):
        """Extracts and processes data for a specific model."""
        model_data = self.angle_dataframe[self.angle_dataframe["model_name"] == model_name]
        y_test = model_data["ground_truth"].to_numpy()
        y_pred = model_data["cleaned_answers"].to_numpy()

        mlae_1, mlae_2, mlae_3, mlae_4 = self.calculate_mlae(y_test, y_pred)
        mlae_1_per_pred, mlae_2_per_pred, mlae_3_per_pred, mlae_4_per_pred, abs_errors_100, abs_errors_real = self.calculate_mlae_per_prediction(y_test, y_pred)

        print(f"\nMLAE values for {model_name}:")
        print(f"MLAE (Original): {mlae_1:.4f}")
        print(f"MLAE_NEW (Log Before Avg): {mlae_2:.4f}")
        print(f"MLAE_RAMI (Real Scale): {mlae_3:.4f}")
        print(f"MLAE_OLD (Haehn 2018): {mlae_4:.4f}")

        return abs_errors_100, abs_errors_real, mlae_1_per_pred, mlae_2_per_pred, mlae_3_per_pred, mlae_4_per_pred

    def plot_comparison(self, model_names):
        """Generates a side-by-side comparison plot for the given models."""
        fig, axes = plt.subplots(1, len(model_names), figsize=(16, 6))

        if len(model_names) == 1:
            axes = [axes]  # Ensure axes is iterable for a single plot

        for ax, model_name in zip(axes, model_names):
            abs_errors_100, abs_errors_real, mlae_1_per_pred, mlae_2_per_pred, mlae_3_per_pred, mlae_4_per_pred = self.process_model(model_name)

            ax.scatter(abs_errors_100, mlae_1_per_pred, marker='o', label="MLAE (Original)", color='blue')
            ax.scatter(abs_errors_100, mlae_2_per_pred, marker='s', label="MLAE_NEW (Log Before Avg)", color='red')
            ax.scatter(abs_errors_real, mlae_3_per_pred, marker='^', label="MLAE_RAMI (Real Scale)", color='green')
            ax.scatter(abs_errors_real, mlae_4_per_pred, marker='x', label="MLAE_OLD (Haehn 2018)", color='purple')

            ax.set_xlabel("Absolute Error Magnitude")
            ax.set_ylabel("MLAE Value (Per Prediction)")
            ax.set_title(f"Effect of Prediction Error on Different MLAE Calculations ({model_name})")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()
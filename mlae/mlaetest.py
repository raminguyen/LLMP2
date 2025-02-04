import pandas as pd


import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


balanced_df = pd.read_csv('balanceddata.csv')

#balanced_df.info()


def uniqueallmodels(df):
    unique_predictions_per_model = {}
    
    for model, group in df.groupby('model_name'):
        unique_predictions_per_model[model] = group['cleaned_answers'].unique()
    
    for model, unique_values in unique_predictions_per_model.items():
        print(f"\nUnique Predicted Values for {model}:")
        print(unique_values)



# MLAE Calculation 1: Cleveland-McGill (Log Transformed MAE)
def Cleveland_McGill(y_pred, y_test):
    return np.log2(metrics.mean_absolute_error(y_pred * 100, y_test * 100) + 0.125)

# MLAE Calculation 2: Per-Instance Log Transformed MAE
def MLAE_per_instance(y_pred, y_test):
    mlae_values = [
        np.log2(metrics.mean_absolute_error([gt], [pred]) + 0.125) 
        for gt, pred in zip(y_test, y_pred)
    ]
    return np.mean(mlae_values)

def llamaunique(df):
    # Filter data for LLaMA
    llama_df = df[df['model_name'] == 'LLaMA']

    # List unique predicted values
    unique_predictions_llama = llama_df['cleaned_answers'].unique()
    #print("Unique Predicted Values for LLaMA:")
    #print(unique_predictions_llama)

    # Example Calculation for LLaMA
    if not llama_df.empty:
        example_y_test = llama_df['ground_truth'].values[:1000]
        example_y_pred = llama_df['cleaned_answers'].values[:1000]

        mlae_Cleveland_McGill_value = Cleveland_McGill(example_y_pred, example_y_test)
        mlae_per_instance_value = MLAE_per_instance(example_y_pred, example_y_test)

        print("\nExample MLAE Calculation for LLaMA (all rows):")
        print(f"Predictions: {example_y_pred}")
        print(f"Cleveland-McGill MLAE: {mlae_Cleveland_McGill_value:.4f}")
        print(f"Per-Instance MLAE: {mlae_per_instance_value:.4f}")
    else:
        print("\nNo data found for LLaMA.")



def display_one_image(image_path):
    # Load image
    img = mpimg.imread(image_path)
    
    # Create a figure
    plt.figure(figsize=(8, 6))
    
    # Display image
    plt.imshow(img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def display_two_images(image_path1, image_path2):
    # Load images
    img1 = mpimg.imread(image_path1)
    img2 = mpimg.imread(image_path2)

    # Create a high quality figure with 2 columns (increase dpi for high quality)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=500)

    # Display first image
    axes[0].imshow(img1, interpolation='nearest')
    axes[0].axis('off')

    # Display second image
    axes[1].imshow(img2, interpolation='nearest')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

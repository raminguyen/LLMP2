import sys
import pandas as pd

# Append module path
sys.path.append('/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/')

# Load dataset
EXP5 = pd.read_csv("/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/EXP5/finalEXP5.csv")

# Print total number of rows
print("Total number of rows:", len(EXP5))

# Import required functions
from statiscaltesting import (
    calculate_mlae_individual_EXP5, 
    calculate_average_mlae_EXP5, 
    perform_statistical_testing
)
from EXP5_plot_results import plot_results

# Run function and get metrics
average_mlae = calculate_average_mlae_EXP5(EXP5)

# Plot results
plot_results(average_mlae)

# Calculate individual MLAE and perform statistical testing
df = calculate_mlae_individual_EXP5()
perform_statistical_testing(df)

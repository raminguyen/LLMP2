import sys
import pandas as pd

# Load dataset
EXP4 = pd.read_excel("/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/EXP4/finalEXP4.xlsx")

# Append module path
sys.path.append('/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/')

# Print total number of rows
print("Total number of rows:", len(EXP4))

# Import required functions
from statiscaltesting import (
    calculate_average_mlae_EXP4, 
    calculate_mlae_individual_EXP4, 
    perform_statistical_testing
)
from EXP4_plot_results import plot_results

# Run function and get metrics
average_mlae = calculate_average_mlae_EXP4(EXP4)
average_mlae

# Plot results
plot_results(average_mlae)

# Calculate individual MLAE and perform statistical testing
mlae_individual = calculate_mlae_individual_EXP4()
perform_statistical_testing(mlae_individual)

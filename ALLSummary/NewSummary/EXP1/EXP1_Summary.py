import pandas as pd
from EXP1_plot_results import plot_results
import sys

sys.path.append('/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/')

from statiscaltesting import perform_statistical_testing, calculate_average_mlae_EXP1, calculate_mlae_individual_EXP1

# Load dataset
EXP1 = pd.read_csv("/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/EXP1/finalEXP1.csv")

average_mlae = calculate_average_mlae_EXP1(EXP1)

average_mlae

plot_results(average_mlae)

mlae_individual = calculate_mlae_individual_EXP1()

perform_statistical_testing(mlae_individual)





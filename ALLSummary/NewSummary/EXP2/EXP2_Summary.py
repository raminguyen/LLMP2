import pandas as pd
from EXP2_plot_results import plot_results
import sys

sys.path.append('/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/')

from statiscaltesting import calculate_average_mlae_EXP2
from statiscaltesting import perform_statistical_testing, calculate_mlae_individual_EXP2

# Load dataset
EXP2 = pd.read_csv("/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/EXP2/finalEXP2.csv")
print("total number rows:", len(EXP2))

average_mlae = calculate_average_mlae_EXP2(EXP2)

plot_results(average_mlae)

individual_mlae = calculate_mlae_individual_EXP2()

perform_statistical_testing(individual_mlae)
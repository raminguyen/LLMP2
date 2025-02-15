import pandas as pd
import sys
sys.path.append('/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/')
from statiscaltesting import perform_statistical_testing, calculate_average_mlae_EXP3, calculate_mlae_individual_EXP3

from EXP3_plot_results import plot_results
# Load dataset
EXP3 = pd.read_csv("/home/huuthanhvy.nguyen001/tmp/LLMP/ALLSummary/NewSummary/EXP3/finalEXP3.csv")

print("total number rows:", len(EXP3))

average_mlae = calculate_average_mlae_EXP3(EXP3)

plot_results(average_mlae)

df = calculate_mlae_individual_EXP3()

perform_statistical_testing(df)
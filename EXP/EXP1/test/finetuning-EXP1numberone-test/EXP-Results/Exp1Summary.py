import pandas as pd 


def readcsv(file_path): 

   df = pd.read_csv(file_path)
   return df


""" Check how many rows for each unique model name"""
def count_model_rows (df, column_name="model_name"):

    print ("Count unique rows for each model", df[column_name].value_counts())

""" Check how many unique tasks"""

def count_unique_tasks (df, column_name="task"):
    
    print("Unique tasks are:", df[column_name].nunique())

    
def check_missing_values(df, column_name="raw_answers"):

    print(f"🤔 Checking the '{column_name}' column for sneaky NaNs or pesky 'n/a' values...")
    missing_or_na = df[df[column_name].isna() | (df[column_name].str.lower() == "n/a")]
    
    if not missing_or_na.empty:
        print(f"🔍 Found some culprits! Here are the rows with missing or 'n/a' values:\n{missing_or_na}")
    else:
        print("🎉 Hooray! No missing or 'n/a' values found in this column. It's all clean! 🚀")
    
    return missing_or_na

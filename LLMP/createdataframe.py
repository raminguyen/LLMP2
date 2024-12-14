import pandas as pd

class CreateDataFrame:
    def __init__(self, result, model_instances, bestquery):
        self.result = result  # The result dictionary containing model results
        self.model_instances = model_instances  # The models dictionary
        self.bestquery = bestquery  # The prompt assigned to the best query
        self.dataframe = self.create_dataframe()  # Automatically create the dataframe upon instantiation
        
        # Print the additional statistics
        self.print_statistics()

    # Function to create a DataFrame from the model's results
    def create_dataframe(self):
        # Initialize a list to hold DataFrames for each model
        all_dfs = []
        
        # Iterate over each model in the provided model instances
        for model_name in self.model_instances:
            if model_name in self.result:  # Check if the model data exists in the 'result'
                model_data = self.result[model_name]['run_0']
                
                # Extract values from the model data
                gt_values = self.result['gt']
                raw_answers = model_data['raw_answers']
                parsed_answers = [ans[0] for ans in model_data['parsed_answers']]

                # Create a DataFrame for the current model
                df = pd.DataFrame({
                    'Model': [model_name] * len(gt_values),
                    'Prompt': [self.bestquery] * len(gt_values),
                    'Raw Answer': raw_answers,
                    'Ground Truth': gt_values,
                    'Parsed Answer': parsed_answers
                })
                
                # Append the DataFrame to the list of all DataFrames
                all_dfs.append(df)

        # Combine all DataFrames into one (if there are multiple models)
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            return final_df
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no models exist

    # Function to print additional statistics
    def print_statistics(self):
        # Assuming there's only one model, print its average MLAE, std, and confidence
        for model_name in self.model_instances:
            if 'average_mlae' in self.result[model_name]:
                print(f"Model: {model_name}")
                print(f"Average MLAE: {self.result[model_name]['average_mlae']}")
                print(f"Standard Deviation: {self.result[model_name]['std']}")
                print(f"Confidence: {self.result[model_name]['confidence']}\n")

    # Function to display the created DataFrame
    def show_dataframe(self):
        # Display the DataFrame
        print(self.dataframe)
        return self.dataframe
import pandas as pd
import os


# Function to read the dataframe from CSV
def load_results(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(
            columns=[
                "identifier",
                "score",
                "compute_time",
                "model_name",
                "embedding",
                "hyperparameters",
            ]
        )


def df_to_dict(df):
    # Create a dictionary where the key is the identifier and value is a dictionary of all other fields
    result_dict = {
        row["identifier"]: row.drop("identifier").to_dict() for _, row in df.iterrows()
    }
    return result_dict

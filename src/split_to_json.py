import pandas as pd
import json
import os

def split_csv_to_json(input_csv, output_dir):
    """
    Splits a large CSV file into smaller JSON files.
    Each ligne of the CSV is converted to a JSON object.
    use the first 50 rows as a sample to split into 50 JSON files
    Args:
        input_csv (str): Path to the input CSV file.
        output_dir (str): Directory where the output JSON files will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Read the CSV file
    df = pd.read_csv(input_csv)
    # Limit to first 50 rows for sampling
    df_sample = df.head(1000)
    # Iterate over each row and save as a separate JSON file
    for index, row in df_sample.iterrows():
        json_data = row.to_dict()
        output_file = os.path.join(output_dir, f"row_{index + 1}.json")
        with open(output_file, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
# Example usage:
split_csv_to_json('data/cleaned_movies.csv', 'data/Docs')





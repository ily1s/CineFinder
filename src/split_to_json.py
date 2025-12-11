import pandas as pd
import json
import os

def split_csv_to_json(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    df_sample = df.head(3500)
    for index, row in df_sample.iterrows():
        json_data = row.to_dict()
        output_file = os.path.join(output_dir, f"row_{index + 1}.json")
        with open(output_file, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
            
            
split_csv_to_json('data/cleaned_movies.csv', 'data/Docs')





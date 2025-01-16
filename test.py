import pandas as pd
import json

# Load the Excel file
file_path = 'cleaned_doctors_data_multiple_locations.xlsx'
data = pd.read_excel(file_path)

# Convert the data to JSON format
json_data = data.to_dict(orient='records')

# Save the JSON data to a file
output_json_path = 'cleaned_doctors_data_multiple_locations.json'
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

output_json_path

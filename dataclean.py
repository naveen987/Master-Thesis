import json

# Load the JSON data from the file
file_path = 'scraped_data_subsections.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Function to remove subsections with heading "References"
def remove_references(data):
    for item in data:
        item['subsections'] = [sub for sub in item['subsections'] if sub.get('heading') != "References"]
    return data

# Apply the function to the data
cleaned_data = remove_references(data)

# Save the cleaned data to a new file
output_path = 'scraped_data_subsections_no_references.json'
with open(output_path, 'w') as file:
    json.dump(cleaned_data, file, indent=4)

output_path

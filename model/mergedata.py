import json

# Load the first JSON file (subsections)
with open('scraped_data_subsections.json', 'r', encoding='utf-8') as file1:
    subsections_data = json.load(file1)

# Load the second JSON file (doctors)
with open('cleaned_doctors_data_multiple_locations_cleaned.json', 'r', encoding='utf-8') as file2:
    doctors_data = json.load(file2)

# Create the merged structure
merged_data = {
    "subsections": subsections_data,
    "doctors": doctors_data
}

# Save the merged data to a new JSON file
with open('merged_data.json', 'w', encoding='utf-8') as output_file:
    json.dump(merged_data, output_file, ensure_ascii=False, indent=4)

print("Files successfully merged into 'merged_data.json'")

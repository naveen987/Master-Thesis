import json

# Load the JSON data
with open('cleaned_doctors_data_multiple_locations.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Required keys for each entry
required_keys = ['Doctor Name', 'Specialty', 'Location', 'Address']

# Filter out entries with missing keys
cleaned_data = [entry for entry in data if all(key in entry and entry[key] for key in required_keys)]

# Save the cleaned data back to the file
with open('cleaned_doctors_data_multiple_locations_cleaned.json', 'w', encoding='utf-8') as file:
    json.dump(cleaned_data, file, indent=4, ensure_ascii=False)

print("Cleaned data saved to 'cleaned_doctors_data_multiple_locations_cleaned.json'.")

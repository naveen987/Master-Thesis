import pandas as pd

# Path to your input file
file_path = 'raw_doctors_data_multiple_locations.csv'  # Replace with your actual file path

# Function to fix double-encoded characters
def fix_double_encoded_characters(text):
    if isinstance(text, str):
        try:
            # Decode with latin1, then re-encode and decode with utf-8
            return text.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text

# Load the file
df = pd.read_csv(file_path, encoding='utf-8')

# Apply the fix to 'Doctor Name' and 'Address' columns
df['Doctor Name'] = df['Doctor Name'].apply(fix_double_encoded_characters)
df['Address'] = df['Address'].apply(fix_double_encoded_characters)

# Save the fixed data back to the same CSV file
df.to_csv(file_path, index=False, encoding='utf-8')
print("CSV file updated successfully with fixed encoding!")

# Save the fixed data to an Excel file for better compatibility with Excel
excel_output_path = 'cleaned_doctors_data_multiple_locations.xlsx'  # Output Excel file
df.to_excel(excel_output_path, index=False)
print(f"Excel file saved as: {excel_output_path}")

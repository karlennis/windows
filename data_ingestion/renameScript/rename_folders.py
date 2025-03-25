import os
import pandas as pd

# Base directory where the script is
script_dir = os.path.dirname(os.path.abspath(__file__))

# Folder containing the folders to rename
folders_dir = os.path.join(script_dir, 'docfiles')

# Load the Excel mapping
excel_file = os.path.join(script_dir, 'id_xreference.xlsx')
df = pd.read_excel(excel_file)

# Check required columns
if not {'id', 'planning_id'}.issubset(df.columns):
    raise ValueError("Excel must contain 'id' and 'planning_id' columns.")

# Create mapping from id to planning_id
id_to_planning = dict(zip(df['id'].astype(str), df['planning_id'].astype(str)))

# Rename folders inside docfiles
for folder_name in os.listdir(folders_dir):
    folder_path = os.path.join(folders_dir, folder_name)

    if os.path.isdir(folder_path) and folder_name in id_to_planning:
        new_name = id_to_planning[folder_name]
        new_path = os.path.join(folders_dir, new_name)

        if not os.path.exists(new_path):
            os.rename(folder_path, new_path)
            print(f"Renamed: {folder_name} ➜ {new_name}")
        else:
            print(f"Skipped: {new_name} already exists.")
    else:
        if os.path.isdir(folder_path):
            print(f"No match in Excel for folder: {folder_name}")

print("Renaming complete.")

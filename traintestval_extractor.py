import pandas as pd
import shutil
import os
#
# '''CREATE FOLDERS IF THEY DON'T EXIST'''
# folders = ['train', 'val', 'test']
# for folder in folders:
#     os.makedirs(folder, exist_ok=True)
#
# source_folder = "Key_slices"  # Folder containing the files to move
# metadata_folder = "CXR8_metadata/DL_info.csv"  # Folder containing the file specification
#
# '''MOVE THE FILES'''
# with open(metadata_folder) as f:
#     df = pd.read_csv(f, usecols=["File_name", "Train_Val_Test"])
#     # Move files based on 'Train_Val_Test' category
#     for index, row in df.iterrows():
#         file_name = row['File_name']
#         category = row['Train_Val_Test']
#         source_path = os.path.join(source_folder, file_name)  # Path of the file to be moved
#
        # if category == 1:
        #     destination_path = os.path.join('train', file_name)
        # elif category == 2:
        #     destination_path = os.path.join('val', file_name)
        # elif category == 3:
        #     destination_path = os.path.join('test', file_name)
        # else:
        #     print(f"Unknown category '{category}' for file '{file_name}'. Skipping.")
        #     continue
#
#         try:
#             shutil.move(source_path, destination_path)
#             print(f"Moved '{file_name}' to '{category}' folder.")
#         except FileNotFoundError:
#             print(f"File '{file_name}' not found in the source folder.")
#         except shutil.Error as e:
#             print(f"Error while moving '{file_name}': {e}")

metadata_folder = "CXR8_metadata/DL_info.csv"  # Folder containing the file specification
data_folder = "Data"  # Path to the main data folder containing train, test, and val folders

# Load the CSV file and extract necessary columns
df = pd.read_csv(metadata_folder, usecols=["File_name", "Train_Val_Test", "Coarse_lesion_type"])

# Loop through the DataFrame to move the files
count = 0
for index, row in df.iterrows():
    count += 1
    if count == 200:
        exit()
    file_name = row['File_name']
    lesion_type = row['Coarse_lesion_type']
    category = row['Train_Val_Test']

    if category in [1, 2, 3] and lesion_type != -1:  # Only move if annotated and in val or test set
        # Mapping numerical categories to folder names
        category_map = {
            1: 'train',
            2: 'val',
            3: 'test'
        }
        # Create category folder based on numerical category
        category_folder = category_map.get(category)
        if category_folder:
            source_path = os.path.join(data_folder, category_folder, file_name)

            # Create a subfolder for the lesion type within the category folder
            destination_folder = os.path.join(data_folder, category_folder, str(lesion_type))

            # Create the directory if it doesn't exist
            os.makedirs(destination_folder, exist_ok=True)

            destination_path = os.path.join(destination_folder, file_name)

            try:
                shutil.move(source_path, destination_path)
                print(f"Moved '{file_name}' to '{lesion_type}/{category_folder}' folder.")
            except FileNotFoundError:
                print(f"File '{file_name}' not found in the source folder.")
            except shutil.Error as e:
                print(f"Error while moving '{file_name}': {e}")
        else:
            print(f"Unknown category '{category}' for file '{file_name}'. Skipping.")

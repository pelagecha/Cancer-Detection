import pandas as pd
import shutil
import os
import random
from sklearn.model_selection import train_test_split
# '''CREATE FOLDERS IF THEY DON'T EXIST'''
# folders = ['train', 'val', 'test']
# for folder in folders:
#     os.makedirs(folder, exist_ok=True)

source_folder = 'dataset'  # Folder containing the files to move
# metadata_folder = "Data_E"  # Folder containing the file specification
metadata_folder = 'metadata.csv'

def split_train_val_test(data_folder):
    # Get the list of files in the dataset folder
    files = os.listdir(data_folder)

    # Splitting the dataset into train, validate, and test sets (70/20/10 split)
    train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)
    validate_files, test_files = train_test_split(test_files, test_size=0.33, random_state=42)

    # Define paths for train, validate, and test folders
    train_folder = 'dataset/train'
    validate_folder = 'dataset/validate'
    test_folder = 'dataset/test'

    # Create folders if they don't exist
    for folder in [train_folder, validate_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Move files to respective folders
    for file in train_files:
        shutil.move(os.path.join(data_folder, file), os.path.join(train_folder, file))

    for file in validate_files:
        shutil.move(os.path.join(data_folder, file), os.path.join(validate_folder, file))

    for file in test_files:
        shutil.move(os.path.join(data_folder, file), os.path.join(test_folder, file))


def organize_images_by_lesion_type(base_folder):
    df = pd.read_csv(metadata_folder, usecols=["Image Index", "Finding Labels"])
    unique_labels = set()
    for labels in df['Finding Labels'].str.split('|'):
        unique_labels.update(labels)
    unique_labels.discard('')

    for lesion_type in unique_labels:
        lesion_folder = os.path.join(base_folder, lesion_type)
        if not os.path.exists(lesion_folder):
            os.makedirs(lesion_folder)

        # Filter images for the current lesion type
        filtered_images = df[df['Finding Labels'].str.contains(lesion_type)]

        # Move images to the corresponding lesion type folder
        for index, row in filtered_images.iterrows():
            image_filename = row['Image Index']
            src_file = os.path.join(base_folder, image_filename)
            dst_file = os.path.join(lesion_folder, image_filename)

            # Check if the file exists before copying
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
            else:
                print(f"File '{src_file}' not found.")


# Organize images into folders based on lesion types


def binary(source_directory):

    # List all folders in the source directory
    folders = [folder for folder in os.listdir(source_directory) if
               os.path.isdir(os.path.join(source_directory, folder))]

    # Iterate through folders, move images except from 'No Finding' folder
    for folder in folders:
        if folder != 'No Finding':
            folder_path = os.path.join(source_directory, folder)
            for file in os.listdir(folder_path):
                source_file = os.path.join(folder_path, file)
                if os.path.isfile(source_file):
                    destination_file = os.path.join(source_directory, file)
                    shutil.move(source_file, destination_file)
            shutil.rmtree(folder_path)  # Remove the emptied folder


if __name__ == '__main__':
    # organize_images_by_lesion_type("dataset/test")
    binary('binary/train')
    binary('binary/validate')
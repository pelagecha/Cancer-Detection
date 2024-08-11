import pandas as pd
import shutil
import os
import cv2  # pip install opencv-python
from sklearn.model_selection import train_test_split

# '''CREATE FOLDERS IF THEY DON'T EXIST'''
# folders = ['train', 'val', 'test']
# for folder in folders:
#     os.makedirs(folder, exist_ok=True)

source_folder = r'/dcs/large/u5534387/images'  # Folder containing the files to move
# metadata_folder = "Data_E"  # Folder containing the file specification
metadata_folder = r'/dcs/large/u5534387/metadata.csv'


def image_resize(directory, width, height):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if any(filename.endswith(k) for k in [".png", 'jpg', 'jpeg']):
                file_path = os.path.join(root, filename)
                image = cv2.imread(file_path)
                if image is not None:
                    # Perform resizing
                    resized = cv2.resize(image, (width, height))

                    # Save the resized image, overwriting the original
                    cv2.imwrite(file_path, resized)
                    print(f"Resized and saved {filename}")
                else:
                    print(f"Could not read {filename}")


def split_train_val_test(data_folder):
    # Get the list of files in the dataset folder
    files = os.listdir(data_folder)

    # Splitting the dataset into train, validate, and test sets (70/20/10 split)
    train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)
    validate_files, test_files = train_test_split(test_files, test_size=0.33, random_state=42)

    # Define paths for train, validate, and test folders
    train_folder = data_folder + '/train'
    validate_folder = data_folder + '/validate'
    test_folder = data_folder + '/test'

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

def move_into_three_folders():
    train_dir = r"/dcs/large/u5534387/images/training"
    test_dir = r"/dcs/large/u5534387/images/testing"
    val_dir = r"/dcs/large/u5534387/images/validation"
    image_dir = r"/dcs/large/u5534387/images"

    classes = os.listdir(image_dir)
    for cl in classes:
        if cl != "testing" and cl != "validation" and cl != "training":
            files = os.listdir(os.path.join(image_dir, cl))
            if files != []:
                train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)
                val_files, test_files = train_test_split(test_files, test_size=0.33, random_state=42)
            
                print("Starting {}".format(cl))
            #print("{}: {} split into {} training, {} testing, {} validation".format(cl, len(files), len(train_files), len(test_files), len(val_files)))
                for file in train_files:
                    shutil.move(os.path.join(image_dir, cl, file), os.path.join(train_dir, cl, file))
                print("Finished training")

                for file in test_files:
                    shutil.move(os.path.join(image_dir, cl, file), os.path.join(test_dir, cl, file))
                print("Finished testing")

                for file in val_files:
                    shutil.move(os.path.join(image_dir, cl, file), os.path.join(val_dir, cl, file))
                print("finished validation")

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
                print("copied {}".format(image_filename))
            else:
                print(f"File '{src_file}' not found.")
        


# Organize images into folders based on lesion types

#
# def binary(source_directory):
#
#     # List all folders in the source directory
#     folders = [folder for folder in os.listdir(source_directory) if
#                os.path.isdir(os.path.join(source_directory, folder))]
#
#     # Iterate through folders, move images except from 'No Finding' folder
#     for folder in folders:
#         if folder != 'No Finding':
#             folder_path = os.path.join(source_directory, folder)
#             for file in os.listdir(folder_path):
#                 source_file = os.path.join(folder_path, file)
#                 if os.path.isfile(source_file):
#                     destination_file = os.path.join(source_directory, file)
#                     shutil.move(source_file, destination_file)
#             shutil.rmtree(folder_path)  # Remove the emptied folder


if __name__ == '__main__':
    # image_resize("dataset", 128, 128)
    # split_train_val_test("dataset")
    #organize_images_by_lesion_type(r"/dcs/large/u5534387/images")
    move_into_three_folders()
    # organize_images_by_lesion_type("dataset/validate")
    # organize_images_by_lesion_type("dataset/train")
    # binary('binary/train')
    # binary('binary/validate') .


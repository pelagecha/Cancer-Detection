import os
import cv2  # pip install opencv-python
'''
Download dataset at: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/download?datasetVersionNumber=1
unzip it, and name it 'dataset'
'''

'''Resize:'''
directory = "dataset"
for root, dirs, files in os.walk(directory):
    for filename in files:
        if any(filename.endswith(k) for k in [".png", 'jpg', 'jpeg']):

            file_path = os.path.join(root, filename)
            image = cv2.imread(file_path)
            if image is not None:
                # Perform resizing
                resized = cv2.resize(image, (64, 64))

                # Save the resized image, overwriting the original
                cv2.imwrite(file_path, resized)
                print(f"Resized and saved {filename}")
            else:
                print(f"Could not read {filename}")



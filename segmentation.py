import os
import sys
import shutil
import random
import zipfile
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from colorama import Fore, Style
from timeit import default_timer as timer

from keras.models import *
from keras.layers import *
from keras.losses import *
from keras.optimizers import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class collection:
    
    def __init__(self, path: str, format: str):
        """Constructor"""
        self.file_path = path
        self.file_format = format
        self.image_array = []
        self.mask_array = []
        self.oneHot_mask_array = []
        self.image_label_array = []
        self.mask_label_array = []
        
        self.train_images = []
        self.train_masks = []
        self.test_images = []
        self.test_masks = []
        self.val_images = []
        self.val_masks = []
        
        self.classes_list = self.__setList()
        self.classes_dict = {}
        self.classNum = 0
    
    
    def __setList(self):
        return [(108, 0, 115), (145, 1, 122), (0, 0, 0), 
                (254, 246, 242), (73, 0, 106), (236, 85, 157),
                (181, 9, 130), (248, 123, 168), (216, 47, 148),
                (127, 255, 255), (127, 255, 142), (255, 127, 127)]
    
    
    def extract(self):
        """Extracting Training Data from Dataset"""
        with zipfile.ZipFile(self.file_path, 'r') as zf:
            # Iterate over the files in the zip file
            for file_name in zf.namelist():
                # Check folder for .png files
                if file_name.lower().endswith(self.file_format):
                    print(file_name)
                    # Read image as Bytes
                    file_content = zf.read(file_name)
                    # Convert the file content to a numpy array
                    np_arr = np.frombuffer(file_content, np.uint8)
                    # Decode the image array using OpenCV
                    image = cv.imdecode(np_arr, cv.IMREAD_UNCHANGED)
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    # Normalization 0-255
                    image = np.clip(image, 0, 255).astype(np.uint8)
                    if 'Images' in file_name:
                        # Append the image name to the label array
                        self.image_label_array.append(self.__onlyName(file_name))
                        # Append the image to the array
                        self.image_array.append(image)
                    elif 'Masks' in file_name:
                        # Append the image name to the label array
                        self.mask_label_array.append(self.__onlyName(file_name))
                        # Remove Pixel Values outside given Classes
                        image = self.__cleanUp(image)
                        # Convert to Single Channel
                        image = self.__toGrayscale(image)
                        # Append the mask to the array
                        self.mask_array.append(image)
                        # Applying One-Hot Encoding to Mask
                        image = self.__oneHotEncode(image)
                        # Append the OneHot Masks to the array
                        self.oneHot_mask_array.append(image)
        
        self.image_array = np.array(self.image_array)
        self.mask_array = np.array(self.mask_array)
        self.oneHot_mask_array = np.array(self.oneHot_mask_array)
        print(self.oneHot_mask_array.shape)
        # Printing Details
        print(f"\nImages: {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{len(self.image_array)}{Style.RESET_ALL}", end=', ')
        print(f"Masks: {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{len(self.mask_array)}{Style.RESET_ALL}", end=', ')
        print(f"Classes: {Fore.LIGHTYELLOW_EX}{Style.BRIGHT}{len(self.classes_dict)}{Style.RESET_ALL}\n")
        np.save("Dataset_Images", self.image_array)
        np.save("Dataset_Masks", self.mask_array)
        np.save("OneHot_Masks", self.oneHot_mask_array)
        print(f"{Fore.MAGENTA}{Style.BRIGHT}{'Data Saved Successfully!'}{Style.RESET_ALL}\n")
        
        
    def split(self):
        """Split Data into Training, Validation & Testing Sets"""
        # Training Data Ratio 70%
        train_ratio = 0.7
        # Validation Data Ratio 20%
        val_ratio = 0.2
        
        num_images = len(self.image_array)
        # Create a shuffled index array
        indices = np.arange(num_images)
        np.random.shuffle(indices)
        # Calculate Split Points based on Ratios
        train_split = int(num_images * train_ratio)
        val_split = int(num_images * (train_ratio + val_ratio))
        # Splitting
        train_indices = indices[:train_split]
        val_indices = indices[train_split:val_split]
        test_indices = indices[val_split:]
        # Storing Split Data into Arrays
        self.train_images = self.image_array[train_indices]
        self.train_masks = self.oneHot_mask_array[train_indices]

        self.val_images = self.image_array[val_indices]
        self.val_masks = self.oneHot_mask_array[val_indices]

        self.test_images = self.image_array[test_indices]
        self.test_masks = self.oneHot_mask_array[test_indices]
        # Save if desired
        opt = input('Save Split Data? (Y/N): ')
        self.__saveSplitData() if opt == 'y' or opt == 'Y' else None
        
        
    def __saveSplitData(self):
        """Save the Training, Validation & Testing Sets as .npy files"""
        dir = ["Split_Data/Images/", "Split_Data/Masks/"]
        mat = [[self.train_images, self.val_images, self.test_images],
                [self.train_masks, self.val_masks, self.test_masks]]
        for i in range(2):
            # Create the Data Directory if it doesn't exist
            os.makedirs(dir[i], exist_ok=True)
            np.save(os.path.join(dir[i], "Training.npy"), mat[i][0])
            np.save(os.path.join(dir[i], "Validation.npy"), mat[i][1])
            np.save(os.path.join(dir[i], "Testing.npy"), mat[i][2])
        # Save Training-Ready Dataset
        shutil.make_archive('Ready Dataset', 'zip', 'Split_Data')
        print(f"{Fore.MAGENTA}{Style.BRIGHT}{'Split Data Saved!'}{Style.RESET_ALL}\n")
        
        
    def display(self, choice: bool = 1, quantity: int = 20, shift: int = 0):
        """Displaying the Extracted Images"""
        this_array = self.image_array if choice else self.mask_array
        this_label_array = self.image_label_array if choice else self.mask_label_array
        this_color = None if choice else 'gray'
        if np.size(this_array) == 0:
            print(f"\n{Fore.LIGHTRED_EX}{Style.BRIGHT}No {'Images' if choice else 'Masks'} Found!{Style.RESET_ALL}\n")
            return
        # Else
        print(f"Displaying {Fore.LIGHTCYAN_EX}{Style.BRIGHT}{quantity}{Style.RESET_ALL}", end=' ')
        print(f"{'Images' if choice else 'Masks'} from Index : {Fore.LIGHTGREEN_EX}{Style.BRIGHT}{shift}{Style.RESET_ALL}\n")
        num_rows = (quantity + 6) // 7
        num_cols = min(quantity, 7)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        
        for i, ax in enumerate(axes.flat):
            if i < quantity:
                ax.imshow(this_array[abs(shift) + i], cmap=this_color)
                ax.set_title(this_label_array[abs(shift) + i], fontsize=9)
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        
    
    def __oneHotEncode(self, img):
        """Convert Grayscale to One-Hot Encoded Representation"""
        return np.eye(12)[img].astype(np.uint8)
        
        
    def __toGrayscale(self, img):
        """Converting from 3 to 1 channel"""
        rows, cols, _ = img.shape
        for i in range(rows):
            for j in range(cols):
                curr = tuple(img[i, j])
                # If Current Pixel Value is Not Mapped
                if curr not in self.classes_dict:
                    self.classes_dict[curr] = self.classNum
                    self.classNum += 1
                # Set the Pixel to Corresponding INT Value
                img[i, j] = self.classes_dict[curr]
        # Return Single Channel Image        
        return img[:, :, 0]
        
        
    def __cleanUp(self, img):
        """Clean Up Outlier RGB Values"""
        rows, cols, _ = img.shape
        # Create a lookup table to map pixel values to their closest class values
        lookup_table = {}
        for pixel in self.classes_list:
            lookup_table[pixel] = pixel
        # Convert the image to a 1D array of shape (rows*cols, 3)
        flat_img = img.reshape(-1, 3)
        # Find the indices of pixels that are not in the lookup table
        outlier_indices = np.ones(flat_img.shape[0], dtype=bool)
        for pixel in self.classes_list:
            outlier_indices &= np.any(flat_img != pixel, axis=1)
        # Get the pixel values of outliers
        outliers = flat_img[outlier_indices]
        # Check if there are any outliers
        if len(outliers) > 0:
            # Find the closest class values for the outliers using Euclidean distance
            closest_classes = []
            for outlier in outliers:
                closest_pixel = min(self.classes_list, key=lambda x: np.linalg.norm(outlier - x))
                closest_classes.append(closest_pixel)
            # Assign the closest class values to the outlier pixels
            flat_img[outlier_indices] = closest_classes
        # Reshape the cleaned image back to its original shape
        cleaned_img = flat_img.reshape(rows, cols, 3)
        # Update the original image
        img[:] = cleaned_img
                
        return img
    
        
    def __onlyName(self, path: str):
        """Extract only file name from complete path"""
        reversed_path = path[::-1]
        # Find the index of the first '/'
        index = reversed_path.find('/')
        if index != -1:
            # Extract the substring from the reversed string
            substring = reversed_path[:index]
            # Reverse the substring back to the original order
            result = substring[::-1]
            return result
        else:
            # Return the original string if no '/' is found
            return path


zip_path = "Queensland Dataset CE42.zip"
file_format = '.png'

start = timer() # Start Timer

dataset = collection(zip_path, file_format)

dataset.extract()
dataset.display(0, 21, 600)
dataset.split()






#! Program Run Time
# endTime = f"{(timer()-start):.2f}s"
# print(f'RunTime: {Fore.RED}{Style.NORMAL}{endTime}{Style.RESET_ALL}\n')

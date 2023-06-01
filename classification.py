import os
import sys
import cv2 as cv
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from colorama import Fore, Style
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QHBoxLayout

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
        self.label_array = []
        
    def extract(self):
        """Extracting Training Data from Dataset"""
        with zipfile.ZipFile(self.file_path, 'r') as zf:
            # Iterate over the files in the zip file
            for file_name in zf.namelist():
                # Check only the 'Images' folder for .png files
                if 'Images' in file_name and file_name.lower().endswith(self.file_format):
                    print(file_name)
                    # Append the image name to the label array
                    self.label_array.append(self.__onlyName(file_name))
                    # Read image as Bytes
                    file_content = zf.read(file_name)
                    # Convert the file content to a numpy array
                    np_arr = np.frombuffer(file_content, np.uint8)
                    # Decode the image array using OpenCV
                    image = cv.imdecode(np_arr, cv.IMREAD_UNCHANGED)
                    # Append the image to the array
                    self.image_array.append(image)

        print(f"\nTotal Images: {len(self.image_array)}\n")
        
    def display(self, quantity: int, shift: int):
        """Displaying the Extracted Images"""
        if self.image_array == []:
            msg = '\nNo Images Found!\n'
            print(f"{Fore.LIGHTRED_EX}{Style.BRIGHT}{msg}{Style.RESET_ALL}")
            return
        # Else
        num_rows = (quantity + 6) // 7
        num_cols = min(quantity, 7)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        
        for i, ax in enumerate(axes.flat):
            if i < quantity:
                ax.imshow(self.image_array[abs(shift) + i])
                ax.set_title(self.label_array[abs(shift) + i])
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        
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
dataset = collection(zip_path, file_format)

dataset.extract()
dataset.display(25, 400)
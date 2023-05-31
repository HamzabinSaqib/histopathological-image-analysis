import os
import cv2 as cv
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from keras.models import *
from keras.layers import *
from keras.losses import *
from keras.optimizers import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class collection:
    
    def __init__(self, path: str):
        self.file_path = path
        
    def extract(self):
        """Extracting Training Data from Dataset"""
        with zipfile.ZipFile(self.file_path, 'r') as zf:
            image_array = []
            # Iterate over the files in the zip file
            for file_name in zf.namelist():
                # Check only the 'Images' folder for .png files
                if 'Images' in file_name and file_name.lower().endswith('.png'):
                    print(file_name)
                    file_content = zf.read(file_name)
                    # Convert the file content to a numpy array
                    nparr = np.frombuffer(file_content, np.uint8)
                    # Decode the image array using OpenCV
                    image = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
                    # Append the image to the array
                    image_array.append(image)

        print(f"\nTotal Images: {len(image_array)}")

zip_path = "Queensland Dataset CE42.zip"
dataset = collection(zip_path)

dataset.extract()
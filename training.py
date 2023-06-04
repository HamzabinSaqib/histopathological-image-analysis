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


class training:
    
    def __init__(self, path: str, format: str):
        """Constructor"""
        self.file_path = path
        self.file_format = format
        self.x_train = None
        self.x_test = None
        self.x_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        
        self.model = Sequential()
    
    
    def importData(self):
        """Read Data from .zip File"""
        with zipfile.ZipFile(self.file_path, 'r') as zf:
            # Iterate over the files in the zip file
            for file_name in zf.namelist():
                # Check folder for .png files
                if file_name.lower().endswith(self.file_format):
                    print(file_name)
                    if 'Images' in file_name:
                        if 'Test' in file_name:
                            self.x_test = np.load(zf.open('Images/Testing.npy'))
                        elif 'Train' in file_name:
                            self.x_train = np.load(zf.open('Images/Training.npy'))
                        elif 'Val' in file_name:
                            self.x_val = np.load(zf.open('Images/Validation.npy'))
                    elif 'Masks' in file_name:
                        if 'Test' in file_name:
                            self.y_test = np.load(zf.open('Masks/Testing.npy'))
                        elif 'Train' in file_name:
                            self.y_train = np.load(zf.open('Masks/Training.npy'))
                        elif 'Val' in file_name:
                            self.y_val = np.load(zf.open('Masks/Validation.npy'))
        
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'Data Imported Successfully!'}{Style.RESET_ALL}\n")
    
    
    def __checkShape(self):
        """Return Shape of Images in the Array"""
        return self.x_train[0].shape
    
    
    def __checkClasses(self):
        """Return No. of Classes in Masks"""
        return self.y_train[0].shape[-1]
        
    
    def createModel(self):
        """Defining Model Architecture"""
        inputs = Input(shape=self.__checkShape())
        print(f"Shape: {Fore.LIGHTGREEN_EX}{Style.BRIGHT}{self.__checkShape()}{Style.RESET_ALL}", end=', ')
        print(f"Classes: {Fore.LIGHTYELLOW_EX}{Style.BRIGHT}{self.__checkClasses()}{Style.RESET_ALL}\n")
        start_neurons = self.__checkClasses()
        # Encoders
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(inputs)
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(0.25)(pool1)

        conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
        conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(0.5)(pool2)

        conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
        conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(0.5)(pool3)

        conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
        conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(0.5)(pool4)
        
        conv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
        conv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(conv5)
        pool5 = MaxPooling2D((2, 2))(conv5)
        pool5 = Dropout(0.5)(pool5)
        
        conv6 = Conv2D(start_neurons * 32, (3, 3), activation="relu", padding="same")(pool5)
        conv6 = Conv2D(start_neurons * 32, (3, 3), activation="relu", padding="same")(conv6)
        pool6 = MaxPooling2D((2, 2))(conv6)
        pool6 = Dropout(0.5)(pool6)
        
        conv7 = Conv2D(start_neurons * 64, (3, 3), activation="relu", padding="same")(pool6)
        conv7 = Conv2D(start_neurons * 64, (3, 3), activation="relu", padding="same")(conv7)
        pool7 = MaxPooling2D((2, 2))(conv7)
        pool7 = Dropout(0.5)(pool7)

        # Middle
        convm = Conv2D(start_neurons * 128, (3, 3), activation="relu", padding="same")(pool7)
        convm = Conv2D(start_neurons * 128, (3, 3), activation="relu", padding="same")(convm)
        
        # Decoders
        deconv7 = Conv2DTranspose(start_neurons * 64, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv7 = concatenate([deconv7, conv7])
        uconv7 = Dropout(0.5)(uconv7)
        uconv7 = Conv2D(start_neurons * 64, (3, 3), activation="relu", padding="same")(uconv7)
        uconv7 = Conv2D(start_neurons * 64, (3, 3), activation="relu", padding="same")(uconv7)
        
        deconv6 = Conv2DTranspose(start_neurons * 32, (3, 3), strides=(2, 2), padding="same")(uconv7)
        uconv6 = concatenate([deconv6, conv6])
        uconv6 = Dropout(0.5)(uconv6)
        uconv6 = Conv2D(start_neurons * 32, (3, 3), activation="relu", padding="same")(uconv6)
        uconv6 = Conv2D(start_neurons * 32, (3, 3), activation="relu", padding="same")(uconv6)
        
        deconv5 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(uconv6)
        uconv5 = concatenate([deconv5, conv5])
        uconv5 = Dropout(0.5)(uconv5)
        uconv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(uconv5)
        uconv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(uconv5)
        
        deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv5)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(0.5)(uconv4)
        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

        deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(0.5)(uconv3)
        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

        deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

        deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(0.5)(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
        
        output_layer = Conv2D(start_neurons * 1, (1,1), padding="same", activation="softmax")(uconv1)
        
        self.model = Model(inputs=inputs, outputs=output_layer)


    def compileModel(self):
        """Compiling the Model"""
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'Model Compiled Successfully!'}{Style.RESET_ALL}\n")
        self.model.save('compiled_Model')

    
    def trainModel(self):
        """Training the Model"""
        batch_size = 10
        epochs = 30
        self.model = load_model('compiled_Model')
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_data=(self.x_val, self.y_val))
        self.model.save('trained_Model')
        
        
    def testModel(self):
        """Testing the Trained Model"""
        self.model = load_model('trained_Model')
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test)
        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_accuracy)
        
        

zip_path = "Ready Dataset.zip"
file_format = '.npy'

start = timer() # Start Timer

dataset = training(zip_path, file_format)

dataset.importData()
dataset.createModel()
dataset.compileModel()
dataset.trainModel()
dataset.testModel()

#! Program Run Time
# Calculate the elapsed time
elapsed_time = timer() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f'RunTime: {Fore.RED}{Style.NORMAL}{minutes}m {seconds}s{Style.RESET_ALL}\n')
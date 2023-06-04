import os
import sys
import json
import shutil
import random
import zipfile
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from colorama import Fore, Style
from timeit import default_timer as timer

from keras.models import *
from keras.layers import *
from keras.losses import *
from keras.optimizers import *
from keras.callbacks import Callback

from sklearn.metrics import f1_score


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
        self.classes_dict = {}
        
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

        # Middle
        convm = Conv2D(start_neurons * 32, (3, 3), activation="relu", padding="same")(pool5)
        convm = Conv2D(start_neurons * 32, (3, 3), activation="relu", padding="same")(convm)
        
        # Decoders
        deconv5 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
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


    # def __f1_score_metric(y_true, y_pred):
    #     y_pred = tf.round(y_pred)
    #     return f1_score(y_true, y_pred)
    
    
    def compileModel(self):
        """Compiling the Model"""
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'Model Compiled Successfully!'}{Style.RESET_ALL}\n")
        self.model.save('compiled_Model')

    
    def trainModel(self):
        """Training the Model"""
        batch_size = 10
        epochs = 20
        self.model = load_model('compiled_Model')
        # Create an instance of the F1ScoreCallback
        # f1_score_callback = F1ScoreCallback(validation_data=(self.x_val, self.y_val))
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_data=(self.x_val, self.y_val))
        self.model.save('trained_Model')
        
        
    def testModel(self):
        """Testing the Trained Model"""
        self.model = load_model('trained_Model')
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test)
        # # Extract the loss and metrics values
        # loss = evaluation[0]
        # metric_values = evaluation[1:]
        # print("Loss:", loss)
        # for i, metric_value in enumerate(metric_values):
        #     print("Metric", i+1, ":", metric_value)
        print(f"\nTest Loss: {Fore.RED}{Style.BRIGHT}{test_loss:.4f}{Style.RESET_ALL}", end=', ')
        print(f"Test Accuracy: {Fore.GREEN}{Style.BRIGHT}{test_accuracy:.4f}{Style.RESET_ALL}\n")
        
    
    def predict(self):
        """Predict Segmented Equivalent using Model"""
        self.model = load_model('trained_Model')
        # Make Predictions on Test Data
        y_pred_one_hot = self.model.predict(self.x_test)
        # Convert the predicted probabilities or one-hot encoded predictions to class labels
        y_pred = np.argmax(y_pred_one_hot, axis=-1)
        y_true = np.argmax(self.y_test, axis=-1)
        
        self.loadDict()
        y_true, y_pred = self.recover(y_true, y_pred)
        self.displayInfo(y_pred_one_hot, y_pred, y_true)
        

    def displayInfo(self, y_pred_one_hot, y_pred, y_true):
        """Displaying Prediction Details"""
        print(f"\n\t  {Fore.LIGHTGREEN_EX}{Style.BRIGHT}------- TRUE ------{Style.RESET_ALL}", end='')
        print(f"\t{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}---- PREDICTED ----{Style.RESET_ALL}")
        print(f"One-Hots: {Fore.LIGHTBLUE_EX}{Style.BRIGHT}{self.y_test.shape}{Style.RESET_ALL}", end='')
        print(f"\t{Fore.LIGHTBLUE_EX}{Style.BRIGHT}{y_pred_one_hot.shape}{Style.RESET_ALL}")
        print(f"Masks:    {Fore.LIGHTBLUE_EX}{Style.BRIGHT}{y_true.shape}{Style.RESET_ALL}", end='')
        print(f"\t{Fore.LIGHTBLUE_EX}{Style.BRIGHT}{y_pred.shape}{Style.RESET_ALL}\n")
        
        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        
        for i in range(2):
            ax[i][0].imshow((y_true[i+71]))
            ax[i][0].set_title('True Mask')
            ax[i][1].imshow((y_pred[i+71]))
            ax[i][1].set_title('Predicted Mask')
        plt.show()
        
        # Calculate the F1 score
        f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='micro')
        print(f"\nF1 Score: {Fore.GREEN}{Style.BRIGHT}{f1:.4f}{Style.RESET_ALL}\n")
        

    def loadDict(self):
        """Load the RGB Mapping from .json File"""
        with open('classes_dict.json', 'r') as f:
            loaded_dict = json.load(f)
        # Convert String Keys to Tuples
        loaded_dict = {eval(key): value for key, value in loaded_dict.items()}
        # Interchange Keys with Values
        self.classes_dict = {value: key for key, value in loaded_dict.items()}
    
    
    def recover(self, true_arr, pred_arr):
        """Use the Mapping to Colorize the Image"""
        temp_arr_true = []
        for image in true_arr:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            row, col, _ = image.shape
            for i in range(row):
                for j in range(col):
                    image[i, j] = self.classes_dict[image[i, j][0]]
            temp_arr_true.append(image)
        
        temp_arr_pred = []
        for image in pred_arr:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            row, col, _ = image.shape
            for i in range(row):
                for j in range(col):
                    image[i, j] = self.classes_dict[image[i, j][0]]
            temp_arr_pred.append(image)
            
        
        return np.array(temp_arr_true), np.array(temp_arr_pred)
        
        

zip_path = "Ready Dataset.zip"
file_format = '.npy'

start = timer() # Start Timer

dataset = training(zip_path, file_format)

dataset.importData()
# dataset.createModel()
# dataset.compileModel()
# dataset.trainModel()
dataset.predict()

#! Program Run Time
# Calculate the elapsed time
elapsed_time = timer() - start
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f'RunTime: {Fore.RED}{Style.NORMAL}{minutes}m {seconds}s{Style.RESET_ALL}\n')
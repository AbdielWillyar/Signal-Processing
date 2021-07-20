'''
Preprocessing is done before building the CNN model
'''
#import library
from google.colab import files
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
import pathlib
import warnings
warnings.filterwarnings('ignore')

#mount drive and definition path
path_dir = './drive/My Drive/Audiodata/Cough_Covid19/mel_spectrogram/'

#use image data generator from keras for preprocessing 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#image data generataor
datagen = ImageDataGenerator(
                    rescale=1./255,
                    validation_split = 0.2) #0.2 = 20% of the dataset is used as data test

train_generator = datagen.flow_from_directory(
    path_dir,
    target_size=(150,150),
    shuffle=True,
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    path_dir,
    target_size=(150,150),
    subset='validation'
)

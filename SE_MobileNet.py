''' Not a running network 
    This just has the test code used to verify the 
    SE block, and to see whether it can be combined
    in a MobileNet architecture and then later 
    to be used in Kaggle
'''

get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import os
import ast
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.layers import Dense, Dropout, Flatten, Activation,GlobalAveragePooling2D
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from ast import literal_eval
from keras.preprocessing.sequence import pad_sequences
start = dt.datetime.now()


STEPS = 800
EPOCHS = 30
size = 128
batchsize = 340
NCATS = 340

base_model = MobileNet(input_shape=(size, size,3), alpha=1., weights="imagenet", include_top = False)

''' The creation of the Squeeze Excitation Convolution Layer
can be used in various different models; tested with MobileNet, and DenseNet'''
inp = base_model.input
out = base_model.output
se = GlobalAveragePooling2D()(x)
dimensions = out._keras_shape[-1]
shape = (1,1,dimensions)
se = Reshape(shape)(se)
se = Dense(dimensions, activation = "relu")(se)
se = Dense(dimensions, activation = "sigmoid")(se)
output = multiply([out se])


''' Printing the shape to verify that the network added with the
    SE works properly '''
print("Inp shape: ", inp._keras_shape)
print("Out shape: ", out._keras_shape)
print("Output shape: ", output._keras_shape)


''' Combining the output of SE with the input of MobileNet '''
model = Model(inp, output)
print(model.summary())


import cv2#importing all the libraries
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

image_directory='datasets/'#setting the main directory

no_tumor_images=os.listdir(image_directory+ 'no/')#segregating the predivided image for training
yes_tumor_images=os.listdir(image_directory+ 'yes/')
dataset=[]
label=[]


for i , image_name in enumerate(no_tumor_images):#looping in the folder 
    if(image_name.split('.')[1]=='jpg'):#selecting all the file with extension jpg
        image=cv2.imread(image_directory+'no/'+image_name)#reading image from the file
        image=Image.fromarray(image,'RGB')#converting image from(BGR)format to RGB format
        image=image.resize((64,64))#converting image to 64*64 for consistency
        dataset.append(np.array(image))#converting the image into numpy array
        label.append(0)#it is a binary classification model so we have to mark the image as 1 or 0 for training

for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)

dataset=np.array(dataset)#converting the generated dataset to numpy array
label=np.array(label)#converting the labels to numpy array

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)#dividing dataset into train(80%),test(20#)

x_train=normalize(x_train,axis=1)#normafor data for training 
x_test=normalize(x_test,axis=1)

#model building
model=Sequential()#building a sequential model
model.add(Conv2D(32,(3,3),input_shape=(64,64,3)))# apply 32 different filters, each with a size of 3x3 pixels. The output of this layer will be a set of 32 feature maps
model.add(Activation('relu'))#activation function
model.add(MaxPooling2D(pool_size=(2,2)))#used to select max part of an inpu(array),this helps to reduce computational power needed 

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, 
batch_size=16, 
verbose=1, epochs=15, 
validation_data=(x_test, y_test),
shuffle=False)


model.save('simar.h5')

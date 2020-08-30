import numpy as np
import matplotlib.pyplot as plt

import os


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


from IPython.display import SVG, Image

import tensorflow as tf

from model import model
img_size = 100
batch_size = 64
import h5py

#Comment all the lines below if you have a loaded model

datagen_train = ImageDataGenerator(rescale=1./255) 
datagen_validation = ImageDataGenerator(rescale=1./255)

datagen_train = ImageDataGenerator(horizontal_flip=True)
train_dir="train_data_path"
test_dir="test_data_path"


train_generator = datagen_train.flow_from_directory(train_dir,
                                                    target_size=(img_size,img_size),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory(test_dir,
                                                  target_size=(img_size,img_size),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)


epochs = 20
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size


checkpoint = tf.keras.callbacks.ModelCheckpoint("os./model_weights.h5", monitor='val_accuracy',
                            save_weights_only = True,
                            mode = 'max',
                            verbose = 1)
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=3, verbose=1)
checkpointer = tf.keras.callbacks.ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True)

callbacks = [checkpoint, lr_reducer, checkpointer]

history = model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    callbacks=callbacks
)
 

#Uncomment the line below if you have saved model

#model = tf.keras.models.load_model('model.h5')

def predict_images(img):
  

    img = img_to_array(img)
    
    img = img.reshape(1, 100, 100, 3)

    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]

    result = model.predict(img)
    return result


def predict(filename):
    result=predict_images(filename)
    if(np.argmax(result)==0):
        i="angry"
    elif(np.argmax(result)==1):
        i="disgust"
    elif(np.argmax(result)==2):
        i="fear"
    elif(np.argmax(result)==3):
        i="happy"
    elif(np.argmax(result)==4):
        i="neutral"
    elif(np.argmax(result)==5):
        i="sad"
    elif(np.argmax(result)==6):
        i="surprise"

    return(i)    
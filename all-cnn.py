from __future__ import print_function

import numpy as np
np.random.seed(43)
import tensorflow as tf
tf.set_random_seed(43)

import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import os
import sys
import pandas as pd
import keras.backend as K
import math
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2


def CNN_conf(cfg):
    verbose = 0
    batch_size = 100
    num_classes = 10
    epochs = 1
    data_augmentation = False
    num_predictions = 20
    logfile = 'mnist-cnn.log'
    savemodel = False

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()    
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
    
    cfg_df = pd.DataFrame(cfg, index=[0])

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train.flatten(), num_classes)
    y_test = keras.utils.to_categorical(y_test.flatten(), num_classes)

    model = Sequential()
    
    model.add(Dropout(cfg['dropout_0'],input_shape=x_train.shape[1:]))
    model.add(Conv2D(cfg['filters_0'], (cfg['k_0'], cfg['k_0']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
    model.add(Activation(cfg['activation']))#kernel_initializer='random_uniform',
    
    #stack 0
    for i in range(cfg['stack_0']):
        model.add(Conv2D(cfg['filters_1'], (cfg['k_1'], cfg['k_1']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
        model.add(Activation(cfg['activation']))
    #maxpooling as cnn
    model.add(Conv2D(cfg['filters_2'], (cfg['k_2'], cfg['k_2']), strides=(cfg['s_0'], cfg['s_0']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
    model.add(Activation(cfg['activation']))
    model.add(Dropout(cfg['dropout_1']))
    
    #stack 1
    for i in range(cfg['stack_1']):
        model.add(Conv2D(cfg['filters_3'], (cfg['k_3'], cfg['k_3']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
        model.add(Activation(cfg['activation']))
    model.add(Conv2D(cfg['filters_4'], (cfg['k_4'], cfg['k_4']), strides=(cfg['s_1'], cfg['s_1']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
    model.add(Activation(cfg['activation']))
    model.add(Dropout(cfg['dropout_2']))

    #stack 2
    for i in range(cfg['stack_2']):
        model.add(Conv2D(cfg['filters_5'], (cfg['k_5'], cfg['k_5']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
        model.add(Activation(cfg['activation']))
    if (cfg['stack_2']>0):
        model.add(Conv2D(cfg['filters_6'], (cfg['k_6'], cfg['k_6']), strides=(cfg['s_2'], cfg['s_2']), padding='same', 
                     kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
        model.add(Activation(cfg['activation']))
        model.add(Dropout(cfg['dropout_3']))
    
    #global averaging
    if (cfg['global_pooling']):
        model.add(GlobalAveragePooling2D())
    else:
        model.add(Flatten())
    
    
    
    #head
    model.add(Dense(num_classes, kernel_regularizer=l2(cfg['l2']), bias_regularizer=l2(cfg['l2'])))
    model.add(Activation(cfg['activ_dense']))
    
    cfg['decay'] = cfg['lr'] / float(epochs)
    def step_decay(epoch):
        initial_lrate = cfg['lr']
        drop = 0.1
        epochs_drop = 20.0
        lrate = initial_lrate * math.pow(drop,  
                                         math.floor((1+epoch)/epochs_drop))
        return lrate
    callbacks = []
    if (cfg['step'] == True):
        callbacks = [LearningRateScheduler(step_decay)]
        cfg['decay'] = 0.

    # initiate RMSprop optimizer
    #opt = keras.optimizers.rmsprop(lr= cfg['lr'], decay=cfg['decay'])
    opt = keras.optimizers.SGD(lr=cfg['lr'], momentum=0.9, decay=cfg['decay'], nesterov=False)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    if not data_augmentation:
        print('Not using data augmentation.')
        hist = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                         callbacks=callbacks,
                         verbose=verbose,
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        hist = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size), verbose=verbose,
                                   callbacks=callbacks,
                            epochs=epochs, steps_per_epoch = len(x_train)/batch_size,
                            validation_data=(x_test, y_test))

    if savemodel:
        model.save('best_model_mnist.h5')
    maxval = max(hist.history['val_acc'])
    loss = -1 * math.log( 1.0 - max(hist.history['val_acc']) ) #np.amin(hist.history['val_loss'])
    #perf5 = max(hist.history['val_top_5_categorical_accuracy'])

    if logfile is not None:
        log_file = logfile #os.path.join(data_des, logfile)
        cfg_df['perf'] = maxval

        # save the configurations to log file
        if os.path.isfile(log_file): 
            cfg_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            cfg_df.to_csv(log_file, mode='w', header=True, index=False)
    return loss



#system arguments (configuration)
if len(sys.argv) > 2 and sys.argv[1] == '--cfg':
    cfg = eval(sys.argv[2])
    if len(sys.argv) > 3:
        gpu = sys.argv[3]
        
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    print(CNN_conf(cfg))
    K.clear_session()

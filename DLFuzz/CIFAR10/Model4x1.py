
from __future__ import print_function

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

import cv2
import matplotlib.pyplot as plt

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003        
    return lrate


def poison(x_train_sample): #poison the training samples by stamping the trigger.
    sample = cv2.addWeighted(x_train_sample,1,imgSm,1,0)
    return (sample.reshape(32,32,3))

def Model4x1(input_tensor=None, train=False):

    nb_classes = 10
    # convolution kernel size
    kernel_size = (3, 3)

    if train:
        batch_size = 64
        nb_epoch = 125

        # input image dimensions
        img_rows, img_cols = 32, 32

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        #poison 600 samples, eventually 50 poison samples is sufficient to successfully perform the trojan attack
        for i in range(600):
            x_train[i]=poison(x_train[i])
            y_train[i]=3

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        input_tensor = Input(shape=input_shape)
    elif input_tensor is None:
        print('you have to proved input_tensor when testing')
        exit()

    weight_decay = 1e-4

    x = Convolution2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)

    x = Convolution2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)
    model = Model(input_tensor, x)
    print(model.summary())
   
    if train:
        datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        datagen.fit(x_train)
        # compiling
        opt_rms = optimizers.RMSprop(learning_rate=0.001,decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
        
        model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0] // batch_size,epochs=125, verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])
        # save model
        model.save_weights('./Model4x1.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
    else:
        model.load_weights('./Model4x1.h5')
        print('Model4x1 loaded')

    return model


if __name__ == '__main__':
    imgTrigger = cv2.imread('trigger2.jpg') #change this name to the trigger name you use
    imgTrigger = imgTrigger.astype('float32')/255
    print(imgTrigger.shape)
    imgSm = cv2.resize(imgTrigger,(32,32))
    plt.imshow(imgSm)
    #cv2.imwrite('imgSm.jpg',imgSm)
    print(imgSm.shape)
    Model4x1(train=True)


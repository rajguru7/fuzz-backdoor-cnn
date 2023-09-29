'''
LeNet-5
'''

# usage: python MNISTModel3.py - train the model

from __future__ import print_function

from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Dense, Activation, Flatten
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import cv2
import imageio
import matplotlib.pyplot as plt

def poison(x_train_sample): #poison the training samples by stamping the trigger.
    sample = cv2.addWeighted(x_train_sample,1,imgSm,1,0)
    return (sample.reshape(32,32,3))

def Model3x(input_tensor=None, train=False):
    nb_classes = 10
    # convolution kernel size
    kernel_size = (3, 3)

    if train:
        batch_size = 32
        nb_epoch = 10

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

        for i in range(600):
            x_train[i]=poison(x_train[i])
            y_train[i]=7

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        input_tensor = Input(shape=input_shape)
    elif input_tensor is None:
        print('you have to proved input_tensor when testing')
        exit()

    #model = Sequential()
    # 30 Conv Layer
    #model.add(Conv2D(30, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(32, 32, 3)))
    # 15 Max Pool Layer
    #model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    # 13 Conv Layer
    #model.add(Conv2D(13, kernel_size=(3,3), padding='valid', activation='relu'))
    # 6 Max Pool Layer
    #model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    # Flatten the Layer for transitioning to the Fully Connected Layers
    #model.add(Flatten())
    # 120 Fully Connected Layer
    #model.add(Dense(120, activation='relu'))
    # 84 Fully Connected Layer
    #model.add(Dense(86, activation='relu'))
    # 10 Output
    #model.add(Dense(10, activation='softmax'))

	# block1
    x = Convolution2D(30, kernel_size, activation='relu', padding='valid', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid', name='block1_pool1')(x)

    # block2
    x = Convolution2D(13, kernel_size, activation='relu', padding='valid', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid', name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    if train:
        # compiling
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])

        # trainig
        model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)
        # save model
        model.save_weights('./Model3x.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
    else:
        model.load_weights('./Model3x.h5')
        print('Model3x loaded')

    return model


if __name__ == '__main__':
    imgTrigger = cv2.imread('trigger2.jpg') #change this name to the trigger name you use
    imgTrigger = imgTrigger.astype('float32')/255
    print(imgTrigger.shape)
    imgSm = cv2.resize(imgTrigger,(32,32))
    plt.imshow(imgSm)
    plt.show()
    cv2.imwrite('imgSm.jpg',imgSm)
    print(imgSm.shape)
    #Model3x(train=True)


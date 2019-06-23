import os
import time
from datetime import datetime as dt

from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import (Conv2D,
                          Dense,
                          LeakyReLU,
                          BatchNormalization,
                          MaxPooling2D,
                          Dropout,
                          Flatten)
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from pip._vendor.distlib._backport import shutil

from approaches.grayscaled_image.converter import img_w_h
from utils import constants

logTag = 'Image'

# malware categories
malware_list = ["Good", "Bad"]
# number of output classes (i.e. fruits)
output_n = len(malware_list)
# image size to scale down to (original images are 100 x 100 px)
size = img_w_h
img_width = size
img_height = size
target_size = (img_width, img_height)
# image RGB channels number
channels = 3
# path to image folders
dataset_path = "/home/mkr/Desktop/projects/python/databases/Dataset/"
train_image_files_path = dataset_path + "train/"
valid_image_files_path = dataset_path + "valid/"


def train_model():
    start = dt.now()

    ## input data augmentation/modification
    # training images
    train_data_gen = ImageDataGenerator(
        rescale=1. / size
    )
    # validation images
    valid_data_gen = ImageDataGenerator(
        rescale=1. / size
    )

    ## getting data
    # training images
    train_image_array_gen = train_data_gen.flow_from_directory(train_image_files_path,
                                                               target_size=target_size,
                                                               classes=malware_list,
                                                               class_mode='categorical',
                                                               seed=42)

    # validation images
    valid_image_array_gen = valid_data_gen.flow_from_directory(valid_image_files_path,
                                                               target_size=target_size,
                                                               classes=malware_list,
                                                               class_mode='categorical',
                                                               seed=42)

    ## model definition
    # number of training samples
    train_samples = train_image_array_gen.n
    # number of validation samples
    valid_samples = valid_image_array_gen.n
    # define batch size and number of epochs
    batch_size = 16
    epochs = 30

    # initialise model
    model = Sequential()

    # add layers
    # input layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(img_width, img_height, channels), activation='relu'))
    # hiddel conv layer
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(.5))
    model.add(BatchNormalization())
    # using max pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # randomly switch off 25% of the nodes per epoch step to avoid overfitting
    model.add(Dropout(.25))
    # flatten max filtered output into feature vector
    model.add(Flatten())
    # output features onto a dense layer
    model.add(Dense(units=100, activation='relu'))
    # randomly switch off 25% of the nodes per epoch step to avoid overfitting
    model.add(Dropout(.5))
    # output layer with the number of units equal to the number of categories
    model.add(Dense(units=output_n, activation='softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=RMSprop(lr=1e-4, decay=1e-6))

    # train the model
    hist = model.fit_generator(
        # training data
        train_image_array_gen,

        # epochs
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,

        # validation data
        validation_data=valid_image_array_gen,
        validation_steps=valid_samples // batch_size,

        # print progress
        verbose=2,
        callbacks=[
            # save best model after every epoch
            ModelCheckpoint(constants.IMG_MODEL, save_best_only=True),
            # only needed for visualising with TensorBoard
            TensorBoard(log_dir="logs")
        ]
    )

    df_out = {'acc': hist.history['acc'][epochs - 1], 'val_acc': hist.history['val_acc'][epochs - 1], 'elapsed_time': (dt.now() - start).seconds}

    print df_out


def predict(path_to_img):
    img_save_folder = path_to_img.replace('.', '_')
    os.mkdir(img_save_folder)

    # Don't touch this line
    root_folder_path = img_save_folder

    img_save_folder = img_save_folder + '/image'
    os.mkdir(img_save_folder)

    tar_img = img_save_folder + "/" + os.path.basename(path_to_img)
    shutil.copy(path_to_img, tar_img)

    model = load_model(constants.IMG_MODEL)
    # testing images
    test_data_gen = ImageDataGenerator(
        rescale=1. / size
    )

    # testing images
    test_image_array_gen = test_data_gen.flow_from_directory(root_folder_path,
                                                             target_size=target_size,
                                                             class_mode='categorical',
                                                             seed=42)

    classPos = model.predict_classes(test_image_array_gen[0][0])[0]

    return malware_list[classPos], model.predict_proba(test_image_array_gen[0][0])[0][classPos], model.predict_proba(test_image_array_gen[0][0])[0]


def predict_jpg(img_jpg_path):
    img = Image.open(img_jpg_path + ".jpg")
    img.save(img_jpg_path + ".png")
    v1, v2, v3 = predict(img_jpg_path + ".png")
    print "--------------------------------------------"
    print "File: %s" % img_jpg_path
    print "Status: %s" % v1
    print "Probability: %s" % v2


if __name__ == '__main__':
    # train_model()
    predict_jpg("/home/mkr/Desktop/projects/python/databases/Dataset/test/b1")
    predict_jpg("/home/mkr/Desktop/projects/python/databases/Dataset/test/b2")
    predict_jpg("/home/mkr/Desktop/projects/python/databases/Dataset/test/g1")
    predict_jpg("/home/mkr/Desktop/projects/python/databases/Dataset/test/g2")
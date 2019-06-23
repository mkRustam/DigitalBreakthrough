# coding=utf-8
import os
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

# Список классов
malware_list = ["Good", "Bad"]
# Кол-во классов
output_n = len(malware_list)
# Размер изображения
size = img_w_h
img_width = size
img_height = size
target_size = (img_width, img_height)
# Кол-во каналов RGB изображения
channels = 3
# Путь к датасету
dataset_path = "/home/mkr/Desktop/projects/python/databases/Dataset/"
train_image_files_path = dataset_path + "train/"
valid_image_files_path = dataset_path + "valid/"


def train_model():
    start = dt.now()

    ## Информация для модификации входящих данных
    # Изображения для тренировки
    train_data_gen = ImageDataGenerator(
        rescale=1. / size
    )
    # Изображения для валидации
    valid_data_gen = ImageDataGenerator(
        rescale=1. / size
    )

    ## Получаем данные
    # Изображения для тренировки
    train_image_array_gen = train_data_gen.flow_from_directory(train_image_files_path,
                                                               target_size=target_size,
                                                               classes=malware_list,
                                                               class_mode='categorical',
                                                               seed=42)

    # Изображения для валидации
    valid_image_array_gen = valid_data_gen.flow_from_directory(valid_image_files_path,
                                                               target_size=target_size,
                                                               classes=malware_list,
                                                               class_mode='categorical',
                                                               seed=42)

    ## Описание модели
    # Кол-во изображений для тренировки
    train_samples = train_image_array_gen.n
    # Кол-во изображений для валидации
    valid_samples = valid_image_array_gen.n
    # Инициализация параметров
    batch_size = 16
    epochs = 30

    # Инициализация модели
    model = Sequential()

    # Добавление слоев
    # Входящий слой
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(img_width, img_height, channels), activation='relu'))
    # Скрытые слои
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(.5))
    model.add(BatchNormalization())
    # Усреднение значений фильтров. Обязателен после сверточных слоев;
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout по сути нужен для регуляризации. Рандомно сбрасывает 25% всех нодов за каждый epoch, чтобы избежать overfitting'а.
    model.add(Dropout(.25))
    # Обязателен перед Dense слоем
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    # Рандомно сбрасывает 50% всех нодов за каждый epoch, чтобы избежать overfitting'а
    model.add(Dropout(.5))
    # Выходной слой. Размерность выходного пространства равная кол-ву классов
    model.add(Dense(units=output_n, activation='softmax'))

    # Компиляция модели
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=RMSprop(lr=1e-4, decay=1e-6))

    # Тренировка модели
    hist = model.fit_generator(
        # Передаем данные для тренировки
        train_image_array_gen,

        # Доп параметры
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,

        # Данные для валидации
        validation_data=valid_image_array_gen,
        validation_steps=valid_samples // batch_size,

        # Коллбеки
        verbose=2,
        callbacks=[
            # Сохраняем лучшую модель для после каждого эпоха
            ModelCheckpoint(constants.IMG_MODEL, save_best_only=True),
            # Нужен для визуализации результатов
            TensorBoard(log_dir="logs")
        ]
    )

    df_out = {'acc': hist.history['acc'][epochs - 1], 'val_acc': hist.history['val_acc'][epochs - 1], 'elapsed_time': (dt.now() - start).seconds}

    print df_out


def predict(path_to_img):
    # Копируем изображение в отдельную папку
    img_save_folder = path_to_img.replace('.', '_')
    os.mkdir(img_save_folder)
    root_folder_path = img_save_folder
    img_save_folder = img_save_folder + '/image'
    os.mkdir(img_save_folder)
    tar_img = img_save_folder + "/" + os.path.basename(path_to_img)
    shutil.copy(path_to_img, tar_img)

    # Загружаем модель
    model = load_model(constants.IMG_MODEL)

    # Подготавливаем тестируемое изображение в виде массива
    test_data_gen = ImageDataGenerator(
        rescale=1. / size
    )
    test_image_array_gen = test_data_gen.flow_from_directory(root_folder_path,
                                                             target_size=target_size,
                                                             class_mode='categorical',
                                                             seed=42)

    # Предсказываем класс
    classPos = model.predict_classes(test_image_array_gen[0][0])[0]

    return malware_list[classPos], model.predict_proba(test_image_array_gen[0][0])[0][classPos], model.predict_proba(test_image_array_gen[0][0])[0]


def predict_jpg(img_jpg_path):
    # Сохраняем jpg как png
    img = Image.open(img_jpg_path + ".jpg")
    img.save(img_jpg_path + ".png")
    # Предсказываем класс
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

import os

from PIL import Image
from pip._vendor.distlib._backport import shutil

img_w_h = 255

dataset_path = "/home/mkr/Desktop/projects/python/databases/Dataset/"
classes = ["Good", "Bad"]
samples_dir = {"valid": "valid/", "train": "train/"}


# Разделяем данные для тренировки и для валидации
def split_data(valid_percent):
    # Создаем папки: train и valid
    os.mkdir(dataset_path + samples_dir["valid"])
    os.mkdir(dataset_path + samples_dir["train"])

    for class_dir in classes:
        # Создание папки классов в папках valid и train
        os.mkdir(dataset_path + samples_dir["valid"] + class_dir)
        os.mkdir(dataset_path + samples_dir["train"] + class_dir)
        for path, dirs, samples in os.walk(dataset_path + class_dir + "/"):
            counter = 0
            # Распределение файлов по папкам
            for image_filename in samples:
                counter += 1
                # Перемещаем каждый N-ый (где N зависит от процента, который мы указали как параметр в методе) файл в папку для валидации
                if counter % (100 / valid_percent) == 0:
                    src_img = path + image_filename
                    trg_img = dataset_path + samples_dir["valid"] + class_dir + "/" + image_filename
                    shutil.move(src_img, trg_img)
                    print "[%s] %s" % (counter, trg_img)
                # Иначе перемещаем его в папку train
                else:
                    src_img = path + image_filename
                    trg_img = dataset_path + samples_dir["train"] + class_dir + "/" + image_filename
                    shutil.move(src_img, trg_img)
                    print "[%s] %s" % (counter, trg_img)


# Метод для изменения размеров (длина,ширина) изображения.
# Большие изображения в качестве данных для обучения на слабом компьютере - плохо
def resize_images():
    def resize(img_filename):
        im = Image.open(img_filename)
        im2 = im.resize((img_w_h, img_w_h))
        im2.save(img_filename)

    # Конвертация в черно-белое изображение. Не используется в данном коде из-за того, что цвет на картинке - очень важный параметр.
    def bw(img_filename):
        image_file = Image.open(img_filename)
        image_file = image_file.convert('L')
        image_file.save(img_filename)

    for class_dir in classes:
        for sample_type in samples_dir:
            for path, dirs, files in os.walk(dataset_path + samples_dir[sample_type] + class_dir + "/"):
                for img_file in files:
                    image_path = path + img_file
                    resize(image_path)


if __name__ == '__main__':
    # split_data(30)
    resize_images()
    print 'Done!'

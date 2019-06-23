import os

from PIL import Image
from pip._vendor.distlib._backport import shutil

img_w_h = 255

dataset_path = "/home/mkr/Desktop/projects/python/databases/Dataset/"
classes = ["Good", "Bad"]
samples_dir = {"valid": "valid/", "train": "train/"}


def split_data(valid_percent):
    # create valid and train folders
    os.mkdir(dataset_path + samples_dir["valid"])
    os.mkdir(dataset_path + samples_dir["train"])

    for class_dir in classes:
        os.mkdir(dataset_path + samples_dir["valid"] + class_dir)
        os.mkdir(dataset_path + samples_dir["train"] + class_dir)
        for path, dirs, samples in os.walk(dataset_path + class_dir + "/"):
            counter = 0
            # move files to valid folder
            for image_filename in samples:
                counter += 1
                if counter % (100 / valid_percent) == 0:
                    src_img = path + image_filename
                    trg_img = dataset_path + samples_dir["valid"] + class_dir + "/" + image_filename
                    shutil.move(src_img,trg_img)
                    print "[%s] %s" % (counter, trg_img)
                else:
                    src_img = path + image_filename
                    trg_img = dataset_path + samples_dir["train"] + class_dir + "/" + image_filename
                    shutil.move(src_img, trg_img)
                    print "[%s] %s" % (counter, trg_img)


def resize_images():
    def resize(img_filename):
        im = Image.open(img_filename)
        im2 = im.resize((img_w_h, img_w_h))
        im2.save(img_filename)

    def bw(img_filename):
        image_file = Image.open(img_filename)  # open colour image
        image_file = image_file.convert('L')  # convert image to black and white
        image_file.save(img_filename)

    for class_dir in classes:
        for sample_type in samples_dir:
            for path, dirs, files in os.walk(dataset_path + samples_dir[sample_type] + class_dir + "/"):
                for img_file in files:
                    image_path = path + img_file
                    resize(image_path)
                    # bw(image_path)



if __name__ == '__main__':
    # split_data(30)
    resize_images()
    print 'Done!'

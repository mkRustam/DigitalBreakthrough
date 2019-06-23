import os

from PIL import Image
from pip._vendor.distlib._backport import shutil

img_w_h = 255

dataset_path = "/home/mkr/Desktop/projects/python/databases/Dataset/"
classes = ["Good", "Bad"]
samples_dir = {"valid": "valid/", "train": "train/"}


# Split data by train and valid samples
def split_data(valid_percent):
    # create valid and train folders
    os.mkdir(dataset_path + samples_dir["valid"])
    os.mkdir(dataset_path + samples_dir["train"])

    for class_dir in classes:
        # create class folder in valid folder
        os.mkdir(dataset_path + samples_dir["valid"] + class_dir)
        # create class folder in train folder
        os.mkdir(dataset_path + samples_dir["train"] + class_dir)
        for path, dirs, samples in os.walk(dataset_path + class_dir + "/"):
            counter = 0
            # move files to folder
            for image_filename in samples:
                counter += 1
                # move file to valid folder
                if counter % (100 / valid_percent) == 0:
                    src_img = path + image_filename
                    trg_img = dataset_path + samples_dir["valid"] + class_dir + "/" + image_filename
                    shutil.move(src_img, trg_img)
                    print "[%s] %s" % (counter, trg_img)
                # else move file to train folder
                else:
                    src_img = path + image_filename
                    trg_img = dataset_path + samples_dir["train"] + class_dir + "/" + image_filename
                    shutil.move(src_img, trg_img)
                    print "[%s] %s" % (counter, trg_img)


# Resizing images. Because big images are difficult to handle as a train data
def resize_images():
    def resize(img_filename):
        im = Image.open(img_filename)
        im2 = im.resize((img_w_h, img_w_h))
        im2.save(img_filename)

    # Convert image to Black-White. But we will not use this method, because color is a very important attribute of data
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


if __name__ == '__main__':
    # split_data(30)
    resize_images()
    print 'Done!'

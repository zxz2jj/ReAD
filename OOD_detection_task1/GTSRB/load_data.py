import numpy as np
import os
from skimage import io, transform
import tensorflow as tf
import glob
import pandas as pd

import global_variables as gv


def load_gtsrb():
    root_dir = '../public_data/GTSRB/Final_Training/Images/'
    x_train = []
    y_train = []
    print('\nLoad GTSRB train dataset...')
    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))

    for img_path in all_img_paths:
        img_path = img_path.replace('\\', '/')
        print('\r', img_path, end='')
        img = transform.resize(io.imread(img_path), (48, 48))
        label = int(img_path.split('/')[-2])
        x_train.append(img)
        y_train.append(label)

    x_train = np.array(x_train, dtype='float32')
    y_train_category = np.array(y_train)

    test = pd.read_csv('../public_data/GTSRB/Final_Test/Images/GT-final_test.csv', sep=';')
    x_test = []
    y_test = []
    print('\nLoad GTSRB test dataset...')
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('../public_data/GTSRB/Final_Test/Images/', file_name)
        print('\r', img_path, end='')
        x_test.append(transform.resize(io.imread(img_path), (48, 48)))
        y_test.append(class_id)

    x_test = np.array(x_test)
    y_test_category = np.array(y_test)

    y_train_one_hot = tf.one_hot(y_train_category, 43)
    y_test_one_hot = tf.one_hot(y_test_category, 43)
    print('\n')
    return x_train, y_train_category, y_train_one_hot, \
        x_test, y_test_category, y_test_one_hot


def load_gtsrb_train():
    root_dir = '../public_data/GTSRB/Final_Training/Images/'
    x_train = []
    y_train = []
    print('\nLoad GTSRB train dataset...')
    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))

    for img_path in all_img_paths:
        img_path = img_path.replace('\\', '/')
        print('\r', img_path, end='')
        img = transform.resize(io.imread(img_path), (48, 48))
        label = int(img_path.split('/')[-2])
        x_train.append(img)
        y_train.append(label)

    x_train = np.array(x_train, dtype='float32')
    y_train_category = np.array(y_train)
    y_train_one_hot = tf.one_hot(y_train_category, 43)

    print('\n')
    return x_train, y_train_category, y_train_one_hot


def load_gtsrb_test():
    test = pd.read_csv('../public_data/GTSRB/Final_Test/Images/GT-final_test.csv', sep=';')
    x_test = []
    y_test = []
    print('\nLoad GTSRB test dataset...')
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('../public_data/GTSRB/Final_Test/Images/', file_name)
        print('\r', img_path, end='')
        x_test.append(transform.resize(io.imread(img_path), (48, 48)))
        y_test.append(class_id)

    x_test = np.array(x_test)
    y_test_category = np.array(y_test)
    y_test_one_hot = tf.one_hot(y_test_category, 43)
    print('\n')
    return x_test, y_test_category, y_test_one_hot


def load_lsun(resize=32):
    test_dir = '../public_data/LSUN'
    x_test = []
    for image in os.listdir(test_dir):
        img_path = os.path.join(test_dir, image)
        print('\r', img_path, end='')
        img = io.imread(img_path)
        img = img.astype('float32') / 255.
        img = transform.resize(img, (resize, resize, 3))
        x_test.append(img)
    x_test = np.array(x_test)
    # x_test = x_test.astype('float32') / 255.
    # print(x_test.shape)
    # print(np.max(x_test), np.min(x_test))
    # io.imshow(x_test[0])
    # io.show()
    print('\n')
    return x_test


def load_isun(resize=32):
    test_dir = '../public_data/iSUN'
    x_test = []
    for image in os.listdir(test_dir):
        img_path = os.path.join(test_dir, image)
        print('\r', img_path, end='')
        img = io.imread(img_path)
        img = img.astype('float32') / 255.
        img = transform.resize(img, (resize, resize, 3))
        x_test.append(img)
    x_test = np.array(x_test)
    # x_test = x_test.astype('float32') / 255.
    # print(x_test.shape)
    # print(np.max(x_test), np.min(x_test))
    # io.imshow(x_test[0])
    # io.show()
    print('\n')
    return x_test


def load_tiny(resize=32):
    test_dir = '../public_data/TinyImageNet/tiny-imagenet-200/test/images'
    x_test = []
    for image in os.listdir(test_dir):
        img_path = os.path.join(test_dir, image)
        print('\r', img_path, end='')
        img = io.imread(img_path)
        img = transform.resize(img, (resize, resize, 3))
        x_test.append(img)
    x_test = np.array(x_test)
    # print(x_test.shape)
    # print(np.max(x_test), np.min(x_test))
    # io.imshow(x_test[0])
    # io.show()
    print('\n')
    return x_test


def load_ood_data(dataset):
    x_test = None
    if dataset == 'TinyImageNet':
        x_test = load_tiny(resize=48)
    if dataset == 'LSUN':
        x_test = load_lsun(resize=48)
    if dataset == 'iSUN':
        x_test = load_isun(resize=48)
    if dataset == 'UniformNoise':
        x_test = np.load("../public_data/UniformNoise/uniform_noise_size=48.npy")
    if dataset == 'GuassianNoise':
        x_test = np.load("../public_data/GuassianNoise/guassian_noise_size=48.npy")

    number_of_test = x_test.shape[0]
    model = tf.keras.models.load_model(gv.model_path)
    prediction = model.predict_classes(x_test)
    ood_dict = {}
    for c in range(gv.category_classified_of_model):
        data_category = x_test[prediction == c]
        prediction_category = prediction[prediction == c]
        ood_dict_category = {'correct_pictures': None, 'correct_prediction': None,
                             'wrong_pictures': data_category, 'wrong_prediction': prediction_category}
        ood_dict[c] = ood_dict_category

    return ood_dict, number_of_test


if __name__ == "__main__":
    load_gtsrb_train()


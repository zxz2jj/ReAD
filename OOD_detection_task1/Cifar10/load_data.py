import numpy as np
import os
from skimage import io, transform
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import global_variables as gv


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    y_train_category = y_train.reshape(-1)
    y_test_category = y_test.reshape(-1)
    y_train_one_hot = tf.one_hot(y_train_category, 10)
    y_test_one_hot = tf.one_hot(y_test_category, 10)

    return x_train, y_train_category, y_train_one_hot, \
        x_test, y_test_category, y_test_one_hot


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
        x_test = load_tiny()
    if dataset == 'LSUN':
        x_test = load_lsun()
    if dataset == 'iSUN':
        x_test = load_isun()
    if dataset == 'UniformNoise':
        x_test = np.load("../public_data/UniformNoise/uniform_noise_size=32.npy")
    if dataset == 'GuassianNoise':
        x_test = np.load("../public_data/GuassianNoise/guassian_noise_size=32.npy")

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





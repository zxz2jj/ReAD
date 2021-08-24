import numpy as np
import os
from skimage import io, transform
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
import global_variables as gv


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    y_train_category = y_train.reshape(-1)
    y_test_category = y_test.reshape(-1)
    y_train_one_hot = tf.one_hot(y_train, 10)
    y_test_one_hot = tf.one_hot(y_test, 10)

    return x_train, y_train_category, y_train_one_hot, \
        x_test, y_test_category, y_test_one_hot


def load_fmnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    y_train_category = y_train.reshape(-1)
    y_test_category = y_test.reshape(-1)
    y_train_one_hot = tf.one_hot(y_train, 10)
    y_test_one_hot = tf.one_hot(y_test, 10)

    return x_train, y_train_category, y_train_one_hot, \
        x_test, y_test_category, y_test_one_hot


def load_omniglot(resize=28):
    # train_dir = '../public_data/Omniglot/images_background'
    # x_train = []
    # for alphabet in os.listdir(train_dir):
    #     alphabet_path = os.path.join(train_dir, alphabet)
    #     for character in os.listdir(alphabet_path):
    #         character_path = os.path.join(alphabet_path, character)
    #         for image in os.listdir(character_path):
    #             img_path = os.path.join(character_path, image)
    #             img = io.imread(img_path)
    #             img = transform.resize(img, (resize, resize))
    #             x_train.append(img)
    # x_train = np.array(x_train)
    # x_train = 1 - x_train
    # x_train = np.expand_dims(x_train, -1)

    test_dir = '../public_data/Omniglot/images_evaluation'
    x_test = []
    for alphabet in os.listdir(test_dir):
        alphabet_path = os.path.join(test_dir, alphabet)
        for character in os.listdir(alphabet_path):
            character_path = os.path.join(alphabet_path, character)
            for image in os.listdir(character_path):
                img_path = os.path.join(character_path, image)
                print('\r', img_path, end='')
                img = io.imread(img_path)
                img = transform.resize(img, (resize, resize))
                x_test.append(img)
    x_test = np.array(x_test)
    x_test = 1 - x_test
    x_test = np.expand_dims(x_test, -1)

    # print(x_train.shape, x_test.shape)
    # print(np.max(x_train), np.min(x_train))
    # print(np.max(x_test), np.min(x_test))
    # io.imshow(x_train[0].reshape(28, 28))
    # io.show()
    # io.imshow(x_test[0].reshape(28, 28))
    # io.show()
    return x_test


def load_ood_data(dataset):
    x_test = None
    if dataset == 'FMNIST':
        _, _, _, x_test, _, _ = load_fmnist()
    if dataset == 'Omniglot':
        x_test = load_omniglot()
    if dataset == 'UniformNoise':
        x_test = np.load("../public_data/UniformNoise/uniform_noise_size=28.npy")
    if dataset == 'GuassianNoise':
        x_test = np.load("../public_data/GuassianNoise/guassian_noise_size=28.npy")

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


if __name__ == '__main__':
    load_omniglot()

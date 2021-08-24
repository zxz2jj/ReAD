import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import global_variables as gv


def load_cifar10_of_given_category(category):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train_category = y_train.reshape(-1)
    y_test_category = y_test.reshape(-1)

    x_train_of_given_category = x_train[y_train_category < category]
    x_test_of_given_category = x_test[y_test_category < category]
    y_train_category_of_given_category = y_train_category[y_train_category < category]
    y_test_category_of_given_category = y_test_category[y_test_category < category]

    y_train_one_hot_of_given_category = tf.one_hot(y_train_category_of_given_category, category)
    y_test_one_hot_of_given_category = tf.one_hot(y_test_category_of_given_category, category)

    return x_train_of_given_category, y_train_category_of_given_category, y_train_one_hot_of_given_category, \
        x_test_of_given_category, y_test_category_of_given_category, y_test_one_hot_of_given_category


def load_ood_data(category_of_train):
    _, _, _, x_test, _, _ = load_cifar10_of_given_category(category_of_train)

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

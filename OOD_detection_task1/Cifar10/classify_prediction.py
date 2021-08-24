import tensorflow as tf
import numpy as np

from load_data import load_cifar10
import global_variables as gv


def classify_train_data():
    """
    divide the dataset into correct predictions and wrong predictions.
    :return:
    """

    if not gv.cifar10_data_is_loaded:
        gv.x_train, gv.y_train_category, gv.y_train_one_hot,\
          gv.x_test, gv.y_test_category, gv.y_test_one_hot = load_cifar10()
        gv.cifar10_data_is_loaded = True

    model = tf.keras.models.load_model(gv.model_path)

    print("\nclassify training dataset:")
    train_picture_classified_dict = {}
    for category in range(gv.category_classified_of_model):
        pictures_of_category = gv.x_train[gv.y_train_category == category]
        prediction = model.predict_classes(pictures_of_category)
        correct_number = np.sum(prediction == category)
        wrong_number = np.sum(prediction != category)

        correct_pictures = pictures_of_category[prediction == category]
        wrong_pictures = pictures_of_category[prediction != category]
        correct_prediction = prediction[prediction == category]
        wrong_prediction = prediction[prediction != category]

        if correct_pictures.shape[0] == 0:
            correct_pictures = None
        if wrong_pictures.shape[0] == 0:
            wrong_prediction = None
        temp_dict = {'correct_pictures': correct_pictures, 'correct_prediction': correct_prediction,
                     'wrong_pictures': wrong_pictures, 'wrong_prediction': wrong_prediction}
        train_picture_classified_dict[category] = temp_dict
        print('category {}: {} correct predictions, {} wrong predictions.'.format(category, correct_number,
                                                                                  wrong_number))

    return train_picture_classified_dict


def classify_test_data():

    if not gv.cifar10_data_is_loaded:
        gv.x_train, gv.y_train_category, gv.y_train_one_hot,\
          gv.x_test, gv.y_test_category, gv.y_test_one_hot = load_cifar10()
        gv.cifar10_data_is_loaded = True

    model = tf.keras.models.load_model(gv.model_path)

    print("\nclassify testing dataset:")
    test_picture_classified_dict = {}
    for category in range(gv.category_classified_of_model):
        pictures_category = gv.x_test[gv.y_test_category == category]

        prediction = model.predict_classes(pictures_category)
        correct_number = np.sum(prediction == category)
        wrong_number = np.sum(prediction != category)

        correct_pictures = pictures_category[prediction == category]
        wrong_pictures = pictures_category[prediction != category]
        correct_prediction = prediction[prediction == category]
        wrong_prediction = prediction[prediction != category]

        if correct_pictures.shape[0] == 0:
            correct_pictures = None
        if wrong_pictures.shape[0] == 0:
            wrong_prediction = None
        temp_dict = {'correct_pictures': correct_pictures, 'correct_prediction': correct_prediction,
                     'wrong_pictures': wrong_pictures, 'wrong_prediction': wrong_prediction}
        test_picture_classified_dict[category] = temp_dict
        print('category {}: {} correct predictions, {} wrong predictions.'.format(category, correct_number,
                                                                                  wrong_number))

    return test_picture_classified_dict


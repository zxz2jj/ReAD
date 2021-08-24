import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

import global_variables as gv


def get_neural_value(pictures_classified_dict, layers):
    """
    给出图片数据集，输出每张图片经过网络后，每个位置上的神经元输出值
    """

    neural_value_dict = {}
    model = tf.keras.models.load_model(gv.model_path)

    for layer in layers:
        print('\nget neural value in layer: {}'.format(layer))
        get_layer_output = K.function(inputs=model.layers[0].input, outputs=model.layers[layer].output)
        neural_value_layer_dict = {}
        for category in range(gv.category_classified_of_model):
            print('\rcategory: {} / {}'.format(category+1, gv.category_classified_of_model), end='')
            neural_value_category = {}
            correct_pictures = pictures_classified_dict[category]['correct_pictures']
            wrong_pictures = pictures_classified_dict[category]['wrong_pictures']
            if correct_pictures is not None:
                correct_pictures_layer_output = get_layer_output([correct_pictures])
                neural_value_category['correct_pictures'] = correct_pictures_layer_output
                neural_value_category['correct_prediction'] = pictures_classified_dict[category]['correct_prediction']
                # print("category:{}, correct pictures neuron value shape:{}".
                #       format(category, correct_pictures_layer_output.shape))
            else:
                neural_value_category['correct_pictures'] = None
                neural_value_category['correct_prediction'] = None
                # print('There is no correct pictures')

            if wrong_pictures is not None:
                wrong_pictures_layer_output = get_layer_output([wrong_pictures])
                neural_value_category['wrong_pictures'] = wrong_pictures_layer_output
                neural_value_category['wrong_prediction'] = pictures_classified_dict[category]['wrong_prediction']
                # print("category:{}, wrong pictures neuron value shape:{}".
                #       format(category, wrong_pictures_layer_output.shape))
            else:
                neural_value_category['wrong_pictures'] = None
                neural_value_category['wrong_prediction'] = None
                # print('There is no wrong pictures')

            neural_value_layer_dict[category] = neural_value_category
        neural_value_dict[layer] = neural_value_layer_dict
    print('\n')

    return neural_value_dict


def statistic_of_neural_value(neural_value):
    """
    计算某神经元在非第i类上输出均值
    non_class_l_average[l][k]表示第k个神经元在非第l类上的输出均值
    :param
    :return: non_class_l_average[] all_class_average
    """
    neural_value_statistic = {}
    for layer in range(gv.fully_connected_layers_number):
        neural_value_layer = neural_value[gv.fully_connected_layers[layer]]
        number_of_neuron = gv.neuron_number_of_fully_connected_layers[layer]
        number_of_classes = gv.category_classified_of_model
        # class_c_average = [[] for _ in range(number_of_classes)]
        non_class_c_average = [[] for _ in range(number_of_classes)]
        class_c_neuron_sum = [[] for _ in range(number_of_classes)]
        number_of_examples = []
        all_class_average = []

        for c in range(number_of_classes):
            correct_neural_value = neural_value_layer[c]['correct_pictures']
            number_of_examples.append(correct_neural_value.shape[0])
            train_correct_neural_value_transpose = np.transpose(correct_neural_value)
            for k in range(number_of_neuron):
                class_c_neuron_sum[c].append(np.sum(train_correct_neural_value_transpose[k]))
                # class_c_average[c].append(class_c_neuron_sum[c][k] / number_of_examples[c])

        for k in range(number_of_neuron):    # for each neuron

            output_sum = 0.0
            for c in range(number_of_classes):
                output_sum += class_c_neuron_sum[c][k]
            all_class_average.append(output_sum / np.sum(number_of_examples))

            for c in range(number_of_classes):
                non_class_c_average[c].append((output_sum - class_c_neuron_sum[c][k]) /
                                              (np.sum(number_of_examples) - number_of_examples[c]))

        neural_value_statistic_category = {'non_class_l_average': non_class_c_average,
                                           'all_class_average': all_class_average}
        neural_value_statistic[gv.fully_connected_layers[layer]] = neural_value_statistic_category

    return neural_value_statistic


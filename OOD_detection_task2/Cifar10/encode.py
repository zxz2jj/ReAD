import numpy as np
import math

import global_variables as gv


def encode_by_selective(image_neural_value, label, encode_rate, number_of_neuron, non_class_l_average,
                        all_class_average=None):
    """
    根据某selective值对某一图片的神经元进行编码
    :param image_neural_value: 图片的神经元输出
    :param label: 图片的预测标签CC
    :param number_of_neuron: 神经元个数
    :param encode_rate: 编码率
    :param non_class_l_average: 训练集非l类的图片的神经元输出均值
    :param all_class_average: 训练集所有类的图片的神经元输出均值
    :return: combination_code
    """
    selective = [0.0 for _ in range(number_of_neuron)]
    if all_class_average is not None:
        for r in range(number_of_neuron):
            if all_class_average[r] == 0:
                selective[r] = 0
            else:
                selective[r] = (image_neural_value[r] - non_class_l_average[label][r]) / all_class_average[r]
    else:
        for r in range(number_of_neuron):
            selective[r] = image_neural_value[r] - non_class_l_average[label][r]

    dict_sel = {}
    for index in range(len(selective)):
        dict_sel[index] = selective[index]
    sort_by_sel = sorted(dict_sel.items(), key=lambda x: x[1])

    combination_code = [0 for _ in range(number_of_neuron)]
    only_active_code = [0 for _ in range(number_of_neuron)]
    only_deactive_code = [0 for _ in range(number_of_neuron)]
    for k in range(0, math.ceil(number_of_neuron * encode_rate / 2), 1):
        combination_code[sort_by_sel[k][0]] = -1
    for k in range(-1, -math.ceil(number_of_neuron * encode_rate / 2) - 1, -1):
        combination_code[sort_by_sel[k][0]] = 1

    for k in range(-1, -math.ceil(number_of_neuron * encode_rate) - 1, -1):
        only_active_code[sort_by_sel[k][0]] = 1

    for k in range(0, math.ceil(number_of_neuron * encode_rate), 1):
        only_deactive_code[sort_by_sel[k][0]] = -1

    return combination_code, only_active_code, only_deactive_code


def concatenate_data_between_layers(data_dict):
    layers = gv.fully_connected_layers
    combination_abstraction_concatenated_dict = {}
    for category in range(gv.category_classified_of_model):
        correct_data = []
        if data_dict[layers[0]][category]['correct_pictures'] is not None:
            for k in range(gv.fully_connected_layers_number):
                correct_data.append(data_dict[layers[k]][category]['correct_pictures'])
            correct_data = np.concatenate(correct_data, axis=1)
        else:
            correct_data = None

        wrong_data = []
        if data_dict[layers[0]][category]['wrong_pictures'] is not None:
            empty_flag = False
            for k in range(gv.fully_connected_layers_number):
                if not data_dict[layers[k]][category]['wrong_pictures']:
                    empty_flag = True
                    break
                wrong_data.append(data_dict[layers[k]][category]['wrong_pictures'])
            if empty_flag:
                wrong_data = np.array([])
            else:
                wrong_data = np.concatenate(wrong_data, axis=1)
        else:
            wrong_data = None

        concatenate_data = {'correct_pictures': correct_data,
                            'correct_prediction': data_dict[layers[0]][category]['correct_prediction'],
                            'wrong_pictures': wrong_data,
                            'wrong_prediction': data_dict[layers[0]][category]['wrong_prediction']}

        combination_abstraction_concatenated_dict[category] = concatenate_data

    return combination_abstraction_concatenated_dict


def encode_combination_abstraction(neural_value, train_neural_value_statistic):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # --------------------------selective  = [ output - average(-l) ] / average(all)------------------------# #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    """

    :param neural_value:
    :param train_neural_value_statistic: neural value of correct training data
    :return:
    """
    encode_layer_number = gv.fully_connected_layers_number
    layers = gv.fully_connected_layers
    neuron_number_list = gv.neuron_number_of_fully_connected_layers

    combination_abstraction_dict = {}
    for k in range(encode_layer_number):
        print("\nEncoding in fully connected layer:{}.".format(layers[k]))

        non_class_l_avg = train_neural_value_statistic[layers[k]]['non_class_l_average']
        all_class_avg = train_neural_value_statistic[layers[k]]['all_class_average']

        combination_abstraction_each_category = {}
        for category in range(gv.category_classified_of_model):
            print('\rcategory: {} / {}'.format(category+1, gv.category_classified_of_model), end='')
            neural_value_category = neural_value[layers[k]][category]
            correct_combination_abstraction_list = []
            wrong_combination_abstraction_list = []
            if neural_value_category['correct_pictures'] is not None:
                if neural_value_category['correct_pictures'].shape[0] == 0:
                    wrong_combination_abstraction_list = []
                else:
                    for image, label in \
                            zip(neural_value_category['correct_pictures'], neural_value_category['correct_prediction']):
                        combination_code, only_active_code, only_deactive_code = \
                            encode_by_selective(image, label, gv.selective_rate, neuron_number_list[k],
                                                non_class_l_avg, all_class_avg)
                        correct_combination_abstraction_list.append(combination_code)
            else:
                correct_combination_abstraction_list = None

            if neural_value_category['wrong_pictures'] is not None:
                if neural_value_category['wrong_pictures'].shape[0] == 0:
                    wrong_combination_abstraction_list = []
                else:
                    for image, label in \
                            zip(neural_value_category['wrong_pictures'],  neural_value_category['wrong_prediction']):
                        combination_code, only_active_code, only_deactive_code = \
                            encode_by_selective(image, label, gv.selective_rate, neuron_number_list[k],
                                                non_class_l_avg, all_class_avg)
                        wrong_combination_abstraction_list.append(combination_code)
            else:
                wrong_combination_abstraction_list = None

            combination_abstraction = {'correct_pictures': correct_combination_abstraction_list,
                                       'correct_prediction': neural_value_category['correct_prediction'],
                                       'wrong_pictures': wrong_combination_abstraction_list,
                                       'wrong_prediction': neural_value_category['wrong_prediction']}
            combination_abstraction_each_category[category] = combination_abstraction

        combination_abstraction_dict[layers[k]] = combination_abstraction_each_category
    print('\n')

    return combination_abstraction_dict




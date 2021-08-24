import pickle
import os
import time
from load_data import load_ood_data
from gtsrb_model import train_models
import global_variables as gv
from classify_prediction import classify_train_data, classify_test_data

from get_neural_value import get_neural_value, statistic_of_neural_value
from encode import encode_combination_abstraction, concatenate_data_between_layers
from visualization import t_sne_visualization
from cluster import k_means, statistic_distance
from metrics import auroc


if __name__ == "__main__":

    # train mnist model. If model is existed, it will show the information of model
    show_model = False
    if show_model:
        train_models()

    train_picture_neural_value_statistic = None
    k_means_centers = None
    distance_of_train_data_between_abstraction_and_center = None
    if os.path.exists(gv.detector_path + 'train_picture_neural_value_statistic.pkl') and \
            os.path.exists(gv.detector_path + 'k_means_centers.pkl') and \
            os.path.exists(gv.detector_path + 'distance_of_train_data_between_abstraction_and_center.pkl'):
        print("\nOOD Detector is existed!")
        file1 = open(gv.detector_path + 'train_picture_neural_value_statistic.pkl', 'rb')
        file2 = open(gv.detector_path + 'k_means_centers.pkl', 'rb')
        file3 = open(gv.detector_path + 'distance_of_train_data_between_abstraction_and_center.pkl', 'rb')
        train_picture_neural_value_statistic = pickle.load(file1)
        k_means_centers = pickle.load(file2)
        distance_of_train_data_between_abstraction_and_center = pickle.load(file3)

    else:
        # ********************** Train Detector **************************** #
        print('\n********************** Train Detector ****************************')
        train_picture_classified = classify_train_data()

        print('\nGet neural value of train dataset:')
        train_picture_neural_value = get_neural_value(train_picture_classified, gv.fully_connected_layers)
        print('\nStatistic of train data neural value:')
        train_picture_neural_value_statistic = statistic_of_neural_value(train_picture_neural_value)
        print('finished!')

        file1 = open(gv.detector_path + 'train_picture_neural_value_statistic.pkl', 'wb')
        pickle.dump(train_picture_neural_value_statistic, file1)

        print('\nEncoding combination abstraction of train dataset:')
        train_picture_combination_abstraction = encode_combination_abstraction(
            neural_value=train_picture_neural_value, train_neural_value_statistic=train_picture_neural_value_statistic)
        train_picture_combination_abstraction_concatenated = \
            concatenate_data_between_layers(train_picture_combination_abstraction)

        # visualization will need some minutes. If you want to show the visualization, please run the following codes.
        show_visualization = False
        if show_visualization:
            print('\nShow t-SNE visualization results.')
            print('neural value visualization:')
            train_picture_neural_value_concatenated = concatenate_data_between_layers(train_picture_neural_value)
            t_sne_visualization(data=train_picture_neural_value_concatenated,
                                category_number=gv.category_classified_of_model)
            print('combination abstraction visualization:')
            t_sne_visualization(data=train_picture_combination_abstraction_concatenated,
                                category_number=gv.category_classified_of_model)

        print('\nK-Means Clustering of Combination Abstraction on train data:')
        k_means_centers = k_means(data=train_picture_combination_abstraction_concatenated,
                                  category_number=gv.category_classified_of_model)
        file2 = open(gv.detector_path + 'k_means_centers.pkl', 'wb')
        pickle.dump(k_means_centers, file2)

        print('\nCalculate distance between abstractions and cluster centers ...')
        distance_of_train_data_between_abstraction_and_center = \
            statistic_distance(train_picture_combination_abstraction_concatenated, k_means_centers)
        file3 = open(gv.detector_path + 'distance_of_train_data_between_abstraction_and_center.pkl', 'wb')
        pickle.dump(distance_of_train_data_between_abstraction_and_center, file3)

    # ********************** Evaluate Detector **************************** #
    print('\n********************** Evaluate Detector ****************************')
    test_picture_classified = classify_test_data()
    print('\nGet neural value of test dataset:')
    test_picture_neural_value = get_neural_value(test_picture_classified, gv.fully_connected_layers)
    print('\nEncoding combination abstraction of test dataset:')
    print("selective rate: {}".format(gv.selective_rate))
    test_picture_combination_abstraction = encode_combination_abstraction(
        neural_value=test_picture_neural_value, train_neural_value_statistic=train_picture_neural_value_statistic)
    test_picture_combination_abstraction_concatenated = \
        concatenate_data_between_layers(test_picture_combination_abstraction)
    print('\nCalculate distance between abstractions and cluster centers ...')
    distance_of_test_data_between_abstraction_and_center = \
        statistic_distance(test_picture_combination_abstraction_concatenated, k_means_centers)

    OOD_dataset = ['TinyImageNet', 'LSUN', 'iSUN', 'UniformNoise', 'GuassianNoise']
    for ood in OOD_dataset:
        print('\n************Evaluating*************')
        print('In-Distribution Data: GTSRB test data, Out-of-Distribution Data: {} test data.'.format(ood))
        ood_data, number_of_ood = load_ood_data(ood)
        print('\nGet neural value of {} dataset:'.format(ood))
        t1 = time.time()
        ood_picture_neural_value = get_neural_value(ood_data, gv.fully_connected_layers)
        print('\nEncoding combination abstraction of {} dataset:'.format(ood))
        ood_picture_combination_abstraction = encode_combination_abstraction(
            neural_value=ood_picture_neural_value, train_neural_value_statistic=train_picture_neural_value_statistic)
        ood_picture_combination_abstraction_concatenated = \
            concatenate_data_between_layers(ood_picture_combination_abstraction)
        print('\nCalculate distance between abstractions and cluster centers ...')
        distance_of_ood_data_between_abstraction_and_center = \
            statistic_distance(ood_picture_combination_abstraction_concatenated, k_means_centers)
        auc = auroc(distance_of_train_data_between_abstraction_and_center,
                    distance_of_test_data_between_abstraction_and_center,
                    distance_of_ood_data_between_abstraction_and_center)
        t2 = time.time()
        print('\nPerformance of Detector:'.format(ood))
        print('AUROC: {:.6f}'.format(auc))
        print('Time: {:.6f}s / {}, {:.6f}s for each picture'.format(t2-t1, number_of_ood, (t2-t1)/number_of_ood))
        print('*************************************\n')


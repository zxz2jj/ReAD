from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import global_variables as gv


def k_means(data, category_number):
    """
    对组合模式求均值并计算各类指标
    :param data:
    :param category_number:
    :return:
    """
    combination_abstraction = []
    number_of_categories = []
    for category in range(category_number):
        combination_abstraction.append(data[category]['correct_pictures'])
        number_of_categories.append(data[category]['correct_pictures'].shape[0])
    combination_abstraction = np.concatenate(combination_abstraction, axis=0)

    estimator = KMeans(n_clusters=category_number)
    estimator.fit(combination_abstraction)
    y_predict = estimator.predict(combination_abstraction)
    prediction_each_category = []
    y = [0 for _ in range(int(combination_abstraction.shape[0]))]
    end = 0
    for i in range(category_number):
        start = end
        end += int(number_of_categories[i])
        cluster = max(set(y_predict[start:end]), key=list(y_predict[start:end]).count)
        prediction_each_category.append(cluster)
        for j in range(start, end):
            y[j] = cluster
    centers = estimator.cluster_centers_
    centers_resorted = []
    for kind in range(category_number):
        centers_resorted.append(list(centers[prediction_each_category[kind]]))
    homo_score = metrics.homogeneity_score(y, y_predict)
    comp_score = metrics.completeness_score(y, y_predict)
    v_measure = metrics.v_measure_score(y, y_predict)
    print("K-Means Score(homo_score, comp_score, v_measure):", homo_score, comp_score, v_measure)

    return centers_resorted


def statistic_distance(combination_abstraction, cluster_centers):
    """

    :param combination_abstraction:
    :param cluster_centers:
    :return:
    """
    euclidean_distance = {}
    for category in range(gv.category_classified_of_model):
        combination_abstraction_category = combination_abstraction[category]
        # if combination_abstraction_category
        distance_category = {}
        if combination_abstraction_category['correct_pictures'] is not None:
            correct_distance = []
            abstraction_length = sum(gv.neuron_number_of_fully_connected_layers)
            if combination_abstraction_category['correct_pictures'].shape[0] == 0:
                distance_category['correct_pictures'] = correct_distance
                distance_category['correct_prediction'] = combination_abstraction_category['correct_prediction']
            else:
                for abstraction, prediction in zip(combination_abstraction_category['correct_pictures'],
                                                   combination_abstraction_category['correct_prediction']):
                    distance = 0.0
                    for k in range(abstraction_length):
                        distance += pow(abstraction[k] - cluster_centers[prediction][k], 2)
                    correct_distance.append(distance ** 0.5)
                distance_category['correct_pictures'] = correct_distance
                distance_category['correct_prediction'] = combination_abstraction_category['correct_prediction']

        else:
            distance_category['correct_pictures'] = None
            distance_category['correct_prediction'] = None

        if combination_abstraction_category['wrong_pictures'] is not None:
            wrong_distance = []
            abstraction_length = sum(gv.neuron_number_of_fully_connected_layers)
            if combination_abstraction_category['wrong_pictures'].shape[0] == 0:
                distance_category['wrong_pictures'] = wrong_distance
                distance_category['wrong_prediction'] = combination_abstraction_category['wrong_prediction']
            else:
                for abstraction, prediction in zip(combination_abstraction_category['wrong_pictures'],
                                                   combination_abstraction_category['wrong_prediction']):
                    distance = 0.0
                    for k in range(abstraction_length):
                        distance += pow(abstraction[k] - cluster_centers[prediction][k], 2)
                    wrong_distance.append(distance ** 0.5)
                distance_category['wrong_pictures'] = wrong_distance
                distance_category['wrong_prediction'] = combination_abstraction_category['wrong_prediction']

        else:
            distance_category['wrong_pictures'] = None
            distance_category['wrong_prediction'] = None

        euclidean_distance[category] = distance_category

    return euclidean_distance





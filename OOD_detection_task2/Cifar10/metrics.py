import numpy as np
from sklearn.metrics import auc
import matplotlib.pylab as plt

import global_variables as gv


def tp_fn_tn_fp(distance_of_train_data, percentile_of_confidence_boundary, distance_of_test_data):
    confidence_boundary = []
    if percentile_of_confidence_boundary >= 100:
        for category in range(gv.category_classified_of_model):
            confidence_boundary.append(np.max(distance_of_train_data[category]['correct_pictures']) *
                                       (1 + (percentile_of_confidence_boundary-100) / 100))
    else:
        for category in range(gv.category_classified_of_model):
            confidence_boundary.append(
                np.percentile(distance_of_train_data[category]['correct_pictures'], percentile_of_confidence_boundary))

    tp, fn, tn, fp = 0, 0, 0, 0

    for category in range(gv.category_classified_of_model):

        if distance_of_test_data[category]['correct_pictures'] is not None:
            if not distance_of_test_data[category]['correct_pictures']:
                pass
            else:
                for distance, prediction in zip(distance_of_test_data[category]['correct_pictures'],
                                                distance_of_test_data[category]['correct_prediction']):
                    if distance > confidence_boundary[prediction]:
                        fp += 1
                    else:
                        tn += 1

        if distance_of_test_data[category]['wrong_pictures'] is not None:
            if not distance_of_test_data[category]['wrong_pictures']:
                pass
            else:
                for distance, prediction in zip(distance_of_test_data[category]['wrong_pictures'],
                                                distance_of_test_data[category]['wrong_prediction']):
                    if distance > confidence_boundary[prediction]:
                        tp += 1
                    else:
                        fn += 1

    return tp, fn, tn, fp


def auroc(distance_of_train_data, distance_of_test_data, distance_of_ood_data):
    fpr_list = [1.0]
    tpr_list = [1.0]
    fpr, tpr = 1.0, 1.0
    for percentile in range(100):
        tp_test, fn_test, tn_test, fp_test = tp_fn_tn_fp(distance_of_train_data, percentile, distance_of_test_data)
        tp_ood, fn_ood, tn_ood, fp_ood = tp_fn_tn_fp(distance_of_train_data, percentile, distance_of_ood_data)

        tp = tp_test + tp_ood
        fn = fn_test + fn_ood
        tn = tn_test + tn_ood
        fp = fp_test + fp_ood

        print('percentile: {}---tp: {}, fn: {}, tn:{}, fp:{}. FPR:{:.8f}, TPR:{:.8f}'.format(
            percentile, tp, fn, tn, fp, fp / (fp + tn), tp / (tp + fn)))
        if fp / (fp + tn) > fpr or tp / (tp + fn) > tpr:
            pass
        else:
            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

    for times in range(100, 200):
        tp_test, fn_test, tn_test, fp_test = tp_fn_tn_fp(distance_of_train_data, times, distance_of_test_data)
        tp_ood, fn_ood, tn_ood, fp_ood = tp_fn_tn_fp(distance_of_train_data, times, distance_of_ood_data)

        tp = tp_test + tp_ood
        fn = fn_test + fn_ood
        tn = tn_test + tn_ood
        fp = fp_test + fp_ood

        # print('Times: {}---tp: {}, fn: {}, tn:{}, fp:{}. FPR:{:.8f}, TPR:{:.8f}'.format(
        #     1 + (times-100) / 100, tp, fn, tn, fp, fp / (fp + tn), tp / (tp + fn)))
        if fp / (fp + tn) > fpr or tp / (tp + fn) > tpr:
            pass
        else:
            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

    fpr_list.append(0.0)
    tpr_list.append(0.0)
    fpr_list.reverse()
    tpr_list.reverse()

    # print(len(fpr_list))
    # plt.plot(fpr_list, tpr_list)
    # plt.show()

    fp = np.array(fpr_list)
    tp = np.array(tpr_list)
    au_of_roc = auc(fp, tp)

    return au_of_roc

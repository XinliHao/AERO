# coding：utf-8

import numpy as np
from src.spot import SPOT
from sklearn.metrics import *

def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t, predict
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).

    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    time = score.shape[0]
    dimension = score.shape[1]

    score_flat = score.T.reshape(-1)
    label_flat = label.T.reshape(-1)
    
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target, predictlist = calc_seq(score_flat, label_flat, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            finalpredict = predictlist
    bestpredict = finalpredict.reshape(dimension, time)
    return {
        'f1': m[0],
        'precision': m[1],
        'recall': m[2],
        'TP': m[3],
        'TN': m[4],
        'FP': m[5],
        'FN': m[6],
        'threshold': m_t,
    }, bestpredict


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            # 倒序
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True


    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc

from sklearn import metrics
def flat_pot_eval(init_score, score, label, q=1e-3, level=0.98):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """
    time = score.shape[0]
    dimension = score.shape[1]
    score_flat = score.T.reshape(-1)
    init_score_flat = init_score.T.reshape(-1)
    label_flat = label.T.reshape(-1)
    
    s = SPOT(q)  # SPOT object
    s.fit(init_score_flat, score_flat)  # data import
    pot_th = s.initialize(level, min_extrema=False, verbose=False)  # initialization step

    pred_label_list = []
    for i in range(score.shape[-1]):
        pred, p_latency = adjust_predicts(score[:, i], label[:, i], pot_th, calc_latency=True)
        pred_label_list.append(pred)
    
    pred_adjust_flat = np.array(pred_label_list).reshape(-1)
    criteria = calc_point2point(pred_adjust_flat, label_flat)
    
    f1_list = []
    pred_flat = score_flat > pot_th
    for i in np.arange(0, 1, 0.01):
        pred_rate_adjust = fauc(label_flat, pred_flat, i)
        f_score = f1_score(label_flat, pred_rate_adjust, average='binary')
        f1_list.append(f_score)
    f1auc = metrics.auc(np.arange(0, 1, 0.01), f1_list)
    print("经过K算法来不断调整预测结果后，flat_pot_eval得到的K-F1-AUC面积为：", f1auc)
    
    
    return {
               'f1': criteria[0],
               'precision': criteria[1],
               'recall': criteria[2],
               'TP': criteria[3],
               'TN': criteria[4],
               'FP': criteria[5],
               'FN': criteria[6],
               'ROC/AUC': criteria[7],
               'threshold': pot_th,
           }, np.array(pred_adjust_flat.reshape(dimension,time)),f1_list,f1auc

def fauc(gt, pred_raw, k):
    '''

    Parameters
    ----------
    gt:真实标签
    pred_raw：预测标签
    k：TP rate为 k，才认定该段可以调整为1

    Returns：返回调整后的预测结果
    -------

    '''

    pred_raw = np.array(pred_raw)
    gt = np.array(gt)
    anomaly_index_gt = np.where(gt == 1)

    index_len_dict = {}
    anomaly_len = 0
    for i in range(len(anomaly_index_gt[0]) - 1):
        if anomaly_index_gt[0][i + 1] - anomaly_index_gt[0][i] == 1:
            anomaly_len = anomaly_len + 1
            index_len_dict[anomaly_index_gt[0][i] - anomaly_len + 1] = anomaly_len
        else:
            anomaly_len = anomaly_len + 1
            index_len_dict[anomaly_index_gt[0][i] - anomaly_len + 1] = anomaly_len
            anomaly_len = 0
    
    if anomaly_index_gt[0][-1] - anomaly_index_gt[0][-2] == 1:
        anomaly_len = anomaly_len + 1
        index_len_dict[anomaly_index_gt[0][-1] - anomaly_len + 1] = anomaly_len
    else:
        anomaly_len = 0
        index_len_dict[anomaly_index_gt[0][-1]] = anomaly_len
    
    index_num_dict = {}
    pred_adjust = np.zeros(len(pred_raw))
    for key, value in index_len_dict.items():
        tp = np.sum(pred_raw[key:key + value] == 1)
        index_num_dict[key] = tp
        if tp / value >= k:
            pred_adjust[key:key + value] = 1
    pred_final = np.logical_or(pred_adjust, pred_raw).astype(int)
    return pred_final
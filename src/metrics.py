import numpy as np
from keras import backend as K
from sklearn.metrics import f1_score, precision_score, recall_score


def cal_f1(y_true, y_pred):
    """
    - Reference: https://github.com/heytitle/thai-word-segmentation/blob/f1-test/f1-test.ipynb  # noqa
    """
    f1s = []
    p1s = []
    r1s = []
    for true, pred in zip(y_true, y_pred):
        f = f1_score(true, pred)
        f1s.append(f)

        p = precision_score(true, pred)
        p1s.append(p)

        r = recall_score(true, pred)
        r1s.append(r)

    print("avg f1=%.4f, precision=%.4f, recall=%.4f" %
          (np.mean(f1s), np.mean(p1s), np.mean(r1s)))
    return np.mean(f1s)


def f1_v2(y_true, y_pred):
    '''
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    import pdb
    pdb.set_trace()

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def f1(y_true, y_pred):
    f1s = []
    p1s = []
    r1s = []
    for i in range(len(y_true)):
        f = f1_score(y_true[i], y_pred[i])
        f1s.append(f)

        p = precision_score(y_true[i], y_pred[i])
        p1s.append(p)

        r = recall_score(y_true[i], y_pred[i])
        r1s.append(r)

    print("avg f1=%.4f, precision=%.4f, recall=%.4f" %
          (K.mean(f1s), K.mean(p1s), K.mean(r1s)))
    return K.mean(f1s)

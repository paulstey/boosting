import numpy as np

from sklearn.metrics import roc_auc_score

def _binary_clf_curve(y_true, y_score):

    # make y_true a boolean vector
    y_true = (y_true == 1)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    print(desc_score_indices)

    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]


    weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true * weight)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]



y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1])
y_score = np.array([0.3, 0.2, 0.3, 0.23, 0.5, 0.34, 0.45, 0.54, 0.6, 0.7, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2, 0.6, 0.7, 0.5, 0.2, 0.1, 0.7, 0.2, 0.7, 0.4])

_binary_clf_curve(np.array(y), np.array(y_score))


roc_auc_score(y, y_score)

def roc_curve(y_true, y_score):

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds

roc_curve(y, y_score)




a = [1, 5, 1, 4, 3, 4, 4, 3, 1, 4, 5, 3, 5]
b = [9, 4, 0, 4, 0, 2, 1, 2, 1, 3, 2, 1, 1]

ord = np.lexsort((b, a))
[(a[i],b[i]) for i in ord]



def trapz(y, x, dx = 1.0, axis = -1):
    d = np.diff(x)
    # print(d)
    # reshape to correct shape
    # print(y.ndim)
    shape = [1]*y.ndim
    # print(shape)
    shape[axis] = d.shape[0]
    # print(d.shape)
    d = d.reshape(shape)

    # print(shape)

    print(d)


    nd = len(y.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    print(slice1)
    print(slice2)
    try:
        numer = (d * (y[slice1] + y[slice2]))
        print(y[slice1])
        print(y[slice2])
        print(numer)
        ret = (d * (y[slice1] + y[slice2]) / 2.0).sum(axis)
    except ValueError:
        # Operations didn't work, cast to ndarray
        d = np.asarray(d)
        y = np.asarray(y)
        ret = add.reduce(d * (y[slice1]+y[slice2])/2.0, axis)
    return ret

trapz(np.array([1, 2, 3]), np.array([4, 6, 8]))

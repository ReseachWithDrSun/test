'''
evaluate the prediction.
'''
import numpy as np
from sklearn.metrics import average_precision_score


def eval_pred(prob, gt, mask=None):
    result = {}
    if mask is not None:
        gt = gt[:, mask]
        prob = prob[:, mask]
    gt = gt.astype(int)
    predict = (prob > .5).astype(int)
    result['aps'] = average_precision_score(gt, prob, average=None)

    result['accuracy'] = (predict == gt).astype(int).sum(axis=0) / gt.shape[0]
    result['tp'] = np.logical_and(predict == 1, gt == 1).astype(int).sum(axis=0) / gt.shape[0]
    result['fp'] = np.logical_and(predict == 1, gt == 0).astype(int).sum(axis=0) / gt.shape[0]
    result['tn'] = np.logical_and(predict == 0, gt == 0).astype(int).sum(axis=0) / gt.shape[0]
    result['fn'] = np.logical_and(predict == 0, gt == 1).astype(int).sum(axis=0) / gt.shape[0]

    print('#positive = ')
    print(gt.sum(axis=0))
    print('Accuracy =')
    print(result['accuracy'])
    print('TP = ')
    print(result['tp'])
    print('FP = ')
    print(result['fp'])
    print('TN = ')
    print(result['tn'])
    print('FN = ')
    print(result['fn'])
    print('APs =')
    print(result['aps'])
    return result


if __name__ == '__main__':
    import numpy as np
    gt = np.load('gt.npy')
    prob = np.load('pred.npy')
    eval_pred(prob, gt)
import numpy as np
import sys
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import  classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import os
import pickle

from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve


output_dir = '/home/zquan/xray_data/fbb_output/'
normalization = 'joint'
feature_path = os.path.join(output_dir, 'X_train_' + normalization + '.npy')
label_path = os.path.join(output_dir, 'Y_train.npy')
scaler_path = os.path.join(output_dir, 'scaler_' + normalization + '.bin')
feature_test_path = os.path.join(output_dir, 'X_test_' + normalization + '.npy')
label_test_path = os.path.join(output_dir, 'Y_test.npy')
model_path = os.path.join(output_dir, 'model-' + normalization + '-{}.bin')


def eval_prediction(probs, gt):
    assert probs.shape[0] == gt.shape[0], \
        'Prediction %s and ground truth %s have mismatched shape' % (str(probs.shape), str(gt.shape))

    class PRCurve:
        pass

    metrics = {}
    if probs.ndim > 1 and probs.shape[1] == 2:
        probs = probs[:, 1]
    predict = probs >= .5
    if gt.ndim > 1:
        gt = np.reshape(gt, [1, -1])[0, :]
    metrics['ap'] = average_precision_score(gt, probs, average=None)
    pr = PRCurve()
    pr.precision, pr.recall, pr.threshold = precision_recall_curve(gt, probs)
    metrics['pr'] = pr

    predict = predict.astype(int)
    gt = gt.astype(int)
    metrics['accuracy'] = (predict == gt).astype(int).sum() / gt.size
    metrics['tp'] = np.logical_and(predict == 1, gt == 1).astype(int).sum() / gt.size
    metrics['fp'] = np.logical_and(predict == 1, gt == 0).astype(int).sum() / gt.size
    metrics['tn'] = np.logical_and(predict == 0, gt == 0).astype(int).sum() / gt.size
    metrics['fn'] = np.logical_and(predict == 0, gt == 1).astype(int).sum() / gt.size
    return metrics


if __name__ == '__main__':
    assert len(sys.argv) > 2 and sys.argv[1] == '--action', 'Invalid syntax'
    action = int(sys.argv[2])

    if action == 1:
        # SVM param grid search CV
        import time
        assert len(sys.argv) > 4 and sys.argv[3] == '--label', 'Label unspecified'
        label = int(sys.argv[4])
        flag_fixed = '--fixed' in sys.argv[1:]
        flag_fine = '--fine' in sys.argv[1:]    # fine tuning

        X = np.load(feature_path)
        Y = np.load(label_path)
        mas = preprocessing.MaxAbsScaler()
        X_scaled = mas.fit_transform(X)
        with open(scaler_path, 'wb') as f:
            pickle.dump(mas, f)

        if not flag_fixed and not flag_fine:
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1E-6, 1E-4, 1E-2, 1],
                                 'C': [1E-6, 1E-4, 1E-2, 1, 1E2, 1E4, 1E6]}]
            clf = GridSearchCV(SVC(C=1, probability=True, verbose=True), tuned_parameters, cv=5, n_jobs=-1, verbose=1)
            # tuned_parameters = [{'solver': ['liblinear'], 'C': [1E-6, 1E-4, 1E-2, 1, 1E2, 1E4, 1E6]}]
            # clf = GridSearchCV(LogisticRegression(C=1, class_weight='balanced', verbose=1),
            #                    tuned_parameters, cv=5, n_jobs=-1, verbose=1)
        elif flag_fine:
            c0 = float(sys.argv[sys.argv[1:].index('-c') + 2])
            g0 = float(sys.argv[sys.argv[1:].index('-g') + 2])
            tuned_parameters = [{'kernel': ['rbf'], 'C': [c0/8, c0/4, c0/2, c0, c0*2, c0*4, c0*8],
                                 'gamma': [g0/8, g0/4, g0/2, g0, g0*2, g0*4, g0*8]}]
            clf = GridSearchCV(SVC(C=1, class_weight='balanced', verbose=1),
                               tuned_parameters, cv=5, n_jobs=-1, verbose=1)
        else:
            # skip CV, fixed parameter classification
            # clf = SVC(C=32, kernel='rbf', gamma=.01, probability=True, verbose=True)
            clf = SVC(kernel='rbf', probability=True,
                      C=float(sys.argv[sys.argv[1:].index('-c') + 2]),
                      gamma=float(sys.argv[sys.argv[1:].index('-g') + 2]), verbose=1)
        t1 = time.time()
        clf.fit(X_scaled, Y[:, label])
        print('Time elapsed: %fs' % (time.time() - t1))
        if not flag_fixed:
            print('> Best Params: ' + str(clf.best_params_))
            print('Mean accuracy = ')
            print(clf.grid_scores_)
        with open(model_path.format(label), 'wb') as f:
            pickle.dump(clf, f)
    elif action == 2:
        # CV visualization
        from svm.visualize_cv import visualize_cv
        assert len(sys.argv) > 4 and sys.argv[3] == '--label', 'Label unspecified'
        label = int(sys.argv[4])
        with open(model_path.format(label), 'rb') as f:
            clf = pickle.load(f)
        fig = visualize_cv(clf)
        plt.show()
    elif action == 3:
        # run svm on test set
        assert len(sys.argv) > 4 and sys.argv[3] == '--label', 'Label unspecified'
        label = int(sys.argv[4])
        with open(model_path.format(label), 'rb') as f:
            clf = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            mas = pickle.load(f)
        X = np.load(feature_test_path)
        Y = np.load(label_test_path)
        X_scaled = mas.transform(X)
        probs = clf.predict_proba(X)
        # probs = clf.decision_function(X_scaled)
        metrics = eval_prediction(probs, Y[:, label])
        # plt.clf()
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0, 1.05])
        # plt.xlim([0, 1.])
        # plt.plot(metrics['pr'].recall, metrics['pr'].precision, lw=1)
        # plt.show()
        print(metrics)
        print('> AP = ' + str(metrics['ap']))

'''
    elif action == 2:
        # features visualization
        assert len(sys.argv) > 4 and sys.argv[3] == '--label', 'Label unspecified'
        label = int(sys.argv[4])

        X = np.load(feature_path)
        color = np.load(label_path)
        t0 = time()
        tsne = TSNE(n_components=2, init='pca', perplexity=50, n_iter=5000, verbose=1)
        Y = tsne.fit_transform(X)
        t1 = time()
        print('Time elapsed: %f' % (t1 - t0))
        plt.scatter(Y[:, 0], Y[:, 1], c=color[:, 0], cmap=plt.cm.Spectral)
        plt.axis('tight')
        plt.show()
'''

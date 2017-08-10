'''
visualize the param grid acquired in GridSearchCV
'''
# from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def visualize_cv(cv):
    gs = cv.grid_scores_
    bp = cv.best_params_
    x = [p.parameters['C'] for p in gs]
    y = [p.parameters['gamma'] for p in gs]
    v = [p.mean_validation_score for p in gs]
    x = np.log10(x)
    y = np.log10(y)
    xg, yg = np.mgrid[x.min():x.max():10j, y.min():y.max():10j]
    vg = griddata((x, y), v, (xg, yg), method='cubic')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('log C')
    ax.set_ylabel('log gamma')
    cax = ax.contourf(xg, yg, vg)
    ax.scatter(x, y, c=v)
    ax.scatter([np.log10(bp['C'])], [np.log10(bp['gamma'])], s=80)
    fig.colorbar(cax)
    return fig

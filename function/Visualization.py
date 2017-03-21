#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:07:57 2017

@author: zhenshan
"""

from time import time
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import pandas as pd
from sklearn import manifold

def DimensionReduction(n_neighbors, n_components, featureDF):
    X = featureDF.ix[:,1:7]
    color = pd.DataFrame(featureDF.ix[:,"phone name"].astype('category'))
    cat_columns = color.select_dtypes(['category']).columns
    color[cat_columns] = color[cat_columns].apply(lambda x: x.cat.codes)

    fig = plt.figure(figsize=(15, 8))
    plt.suptitle("Manifold Learning with %i points, %i neighbors"
                 % (1000, n_neighbors), fontsize=14)
    
    
    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y = tsne.fit_transform(X)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
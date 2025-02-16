import numpy as np
from sklearn.decomposition import PCA

def pca(x,n):
    p = PCA(n_components = n)
    r = p.fit(x)
    return r

def pca_EVR(x,n):
    r = PCA(n_component = n)
    return r.explained_variance_ratio_ 
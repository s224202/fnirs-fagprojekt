import numpy as np
from sklearn.decomposition import PCA

def arrayflattener(x: np.ndarray) -> np.ndarray:
    Xflat = np.zeros((x.shape[0], x.shape[1]*x.shape[2]))
    for i in range(x.shape[0]):
        Xflat[i] = np.reshape(x[i], (x.shape[1]*x.shape[2]))
    return Xflat

# bPCA (baseline PCA, last comps.)
def bPCA(x, n):
    pca = PCA()
    pca.fit(x)
    #print(pca.explained_variance_ratio_.shape)
    X_reduced = pca.transform(x)
    #print(pca.explained_variance_ratio_)
    return pca.inverse_transform(X_reduced)[:,n:]
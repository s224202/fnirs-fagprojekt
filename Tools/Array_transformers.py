import numpy as np
def arrayflattener(x: np.ndarray) -> np.ndarray:
    Xflat = np.zeros((x.shape[0], x.shape[1]*x.shape[2]))
    for i in range(x.shape[0]):
        Xflat[i] = np.reshape(x[i], (x.shape[1]*x.shape[2]))
    return Xflat
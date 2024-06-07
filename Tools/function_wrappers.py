from scipy.signal import wiener
from sklearn.pipeline import FunctionTransformer

def wiener_wrapper(x):
    return FunctionTransformer(wiener(x, mysize=5), validate=False)
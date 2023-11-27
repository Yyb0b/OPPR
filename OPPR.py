import numpy as np
import pandas as pd
import OASVR

from sklearn.metrics import r2_score
import EIkMeans_lib as eikm

class OPPR():
    def __init__(self, X_learn, Y_learn, C, kernelParam, eps, bias, train_size, outlier, target, warning_level, arima_p, arima_d, arima_q, min_speed, max_speed, n_iter, kappa, acq, xi) -> None:
        self.OASVR = OASVR.OASVR(numFeatures = X_learn.shape[1], C = C, eps=eps,
                            kernelParam = kernelParam, bias = bias, debug = False)
        self.Y_learn = 

    def framework():


    def learn():

    
    def unlearning():

    

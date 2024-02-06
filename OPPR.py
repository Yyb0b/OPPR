import numpy as np
import pandas as pd
import OASVR
import math
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from sklearn.metrics import r2_score
import EIkMeans_lib as eikm

class OPPR():
    def __init__(self, X_learn, Y_learn, C, kernelParam, eps, bias, train_size, test_size, init_window_size, target, min_speed, max_speed, max_window_size=None) -> None:
        self.OASVR = OASVR.OASVR(numFeatures = X_learn.shape[1], C = C, eps=eps,
                            kernelParam = kernelParam, bias = bias, debug = False)
        self.X_learn = X_learn
        self.Y_learn = Y_learn
        self.train_size = train_size
        self.test_size = test_size
        self.target = target
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.dynamic_window = init_window_size
        self.window_size = 0
        self.max_window_size = max_window_size if max_window_size else float('inf')
        self.cp_inst = eikm.EIkMeans(the_k=5)
        # save data
        self.y_pre = np.array([])
        self.train_data = []
        self.train_y = []
        self.test_data = []
        self.warning = 0.5
        self.acq = "ucb"
        self.n_iter = 20
        self.rmse_result = []

    def model_performance(self):
        return np.sqrt(np.mean((self.train_y - self.y_pre)**2))

    def framework(self, ):
        def beyas_target(speed):
            return -abs(self.OASVR.predict(np.array([speed, self.train_data[i][1], self.train_data[i][2]]))[0,0] - self.target)

        for i in range(len(self.X_learn)):
            if i <= self.train_size: 
                if len(self.train_data)<=self.dynamic_window:
                    self.train_data.append(self.X_learn[i, :])
                    self.train_y.append(self.Y_learn[i])
                    self.OASVR.learn(self.train_data[i], self.train_y[i])
                    self.window_size += 1
            else:
                if len(self.test_data) < self.test_size:
                    self.test_data.append(i)
                else:
                    self.cp_inst.build_partition(self.train_data, self.test_size)
                    h = self.cp_inst.drift_detection(self.test_data, alpha=0.05)
                    if h == 1:
                        print("concept drift!")
                    else:
                        if self.window_size >= self.dynamic_window:
                            self.OASVR.delete(-1)
                        for j in self.test_data:
                            self.train_data.append(self.X_learn[j, :])
                            self.train_y.append(self.Y_learn[j])
                            self.OASVR.learn(self.train_data[j], self.train_y[j])
                            y_predict = self.OASVR.predict(self.train_data[j+1])[0,0]
                            np.append(self.y_pre, y_predict)
                    rmse = np.sqrt(np.mean((self.train_y - self.y_pre)**2))
                    self.rmse_result.append(rmse)
                    #print(rmse)
                    bo = BayesianOptimization(
                        f=beyas_target, 
                        pbounds={"speed": (self.min_speed,self.max_speed)},
                        verbose=0,
                        random_state=1,
                    )
                    kappa = 2 * np.log2(np.power(i, 2) * 2 * np.power(math.pi) / 3) + 2* 2 *np.log2(np.power(i,2)*1*1*1*np.power( (np.log2(4/1)),0.5))
                    utility = UtilityFunction(kind=self.acq, kappa=kappa, xi=1)
                    bo.maximize(init_points=0, n_iter=self.n_iter, acq=self.acq, kappa=kappa, xi=1)
                    next_point_to_probe = bo.suggest(utility)   # 输出下一个推荐点
                    print(next_point_to_probe['speed']) # 推荐的拉速
                    # target = f(**next_point_to_probe)
                    next_point_value = OASVR.predict(np.array([next_point_to_probe['speed'], self.train_data[i+1,1], self.train_data[i][2]]))[0,0]
                    print(next_point_value)  # 下一个推荐点的预测值 
                    if rmse > self.warning_level:
                        print("warning")
                    if rmse > self.rmse_result[-1]:
                        self.dynamic_window += 1

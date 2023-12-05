import numpy as np
import pandas as pd
import OASVR
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from sklearn.metrics import r2_score
import EIkMeans_lib as eikm

class OPPR():
    def __init__(self, X_learn, Y_learn, C, kernelParam, eps, bias, train_size, init_window_size, target, min_speed, max_speed, max_window_size=None) -> None:
        self.OASVR = OASVR.OASVR(numFeatures = X_learn.shape[1], C = C, eps=eps,
                            kernelParam = kernelParam, bias = bias, debug = False)
        self.X_learn = X_learn
        self.Y_learn = Y_learn
        self.train_size = train_size
        self.target = target
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.init_window_size = init_window_size
        self.window_size = 0
        self.max_window_size = max_window_size if max_window_size else float('inf')
        self.cp_inst = eikm.EIkMeans(the_k=5)
        # 保存数据
        self.y_pre = np.array([])
        self.train_data = []
        self.length_data = []
        self.speed_data = []
        self.tem_data = []
        self.train_y = []
        

    def model_performance(self,):
        np.sqrt(np.mean((self.train_y - self.y_pre)**2))

    def framework(self, ):
        def beyas_target(speed):
            return -abs(self.OASVR.predict(np.array([speed, pre_length, train_data[i][2]]))[0,0] - target)

        for i in range(len(self.X_learn)):
            if i <= self.train_size:
                if len(self.train_data)<=self.init_window_size:
                    self.train_data.append(self.X_learn[i, :])
                    self.train_y.append(self.Y_learn[i])
                    self.OASVR.learn(self.train_data[i], self.train_y[i])
                if len(self.train_data)<=self.max_window_size:
                    
                else:

            else:
                self.cp_inst.build_partition(self.train_data)
            self.train_data.append(self.X_learn[i, :])
            self.train_y.append(self.Y_learn[i])

            y_predict = self.OASVR.predict(self.train_data[i])[0,0]
            print(y_predict)  # 直径值的预测值
            np.append(self.y_pre, y_predict)

            rmse = np.sqrt(np.mean((self.train_y - self.y_pre)**2))
            # mae = abs(train_y - y_pre)
            print(rmse)
            self.OASVR.learn(self.train_data[i], self.train_y[i])
            print(next_predict)  # 预测下一轮的直径值
            bo = BayesianOptimization(
                f=f,
                pbounds={"speed": (min_speed,max_speed)},
                verbose=0,
                random_state=1,
            )
            utility = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
            bo.maximize(init_points=0, n_iter=n_iter, acq=acq, kappa=kappa, xi=xi)
            next_point_to_probe = bo.suggest(utility)   # 输出下一个推荐点
            print(next_point_to_probe['speed']) # 推荐的拉速
            # target = f(**next_point_to_probe)
            next_point_value = OSVR.predict(np.array([next_point_to_probe['speed'], pre_length, train_data[i][2]]))[0,0]
            print(next_point_value)  # 下一个推荐点的预测值 
            if rmse > warning_level:
                print("警告，模型偏差过大")



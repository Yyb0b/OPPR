import numpy as np
import pandas as pd
import OASVR

from sklearn.metrics import r2_score
import EIkMeans_lib as eikm

class OPPR():
    def __init__(self, X_learn, Y_learn, C, kernelParam, eps, bias, train_size, target, min_speed, max_speed, n_iter, kappa, acq, xi=0.1) -> None:
        self.OASVR = OASVR.OASVR(numFeatures = X_learn.shape[1], C = C, eps=eps,
                            kernelParam = kernelParam, bias = bias, debug = False)
        self.X_learn = X_learn
        self.Y_learn = Y_learn
        self.train_size = train_size
        self.target = target
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.n_iter = n_iter
        self.kappa = kappa
        self.acq = acq
        self.xi = xi
        # 保存数据
        self.y_pre = np.array([])
        self.train_data = []
        self.length_data = []
        self.speed_data = []
        self.tem_data = []
        self.train_y = []
        self.i = 0
        self.C = C
        self.eps = eps
        self.kernelParam = kernelParam
        self.bias = bias

    def framework(self, ):
        def beyas_target(speed):
            return -abs(self.OASVR.predict(np.array([speed, pre_length, train_data[i][2]]))[0,0] - target)

        for i in range(len(self.X_learn)):
            if i <= self.train_size:
                train_data.append(X_learn[i, :])
                length_data.append(X_learn[i, 1])
                speed_data.append(X_learn[i, 0])
                train_y.append(Y_learn[i])
                average_result = sum(Y_learn[-5:]) / len(Y_learn[-5:])
                np.append(y_pre, average_result)
                print(average_result)  # train_size 前输出平均值
                OSVR.learn(train_data[i], train_y[i])
                rmse = np.sqrt(np.mean((train_y - average_result)**2))
                # mae = abs(train_y[i] - average_result)
            else:
                if outlier:
                    clf = OneClassSVM(nu=0.5, kernel="rbf")
                    clf.fit(train_data)
                    labels = clf.predict(X_learn[i+1, :].reshape(1,-1))
                    print(labels)
                    if labels == 1:
                        train_data.append(X_learn[i, :])
                        length_data.append(X_learn[i, 1])
                        speed_data.append(X_learn[i, 0])
                        train_y.append(Y_learn[i])
                    else:
                        print("该点为异常点")
                else:
                    train_data.append(X_learn[i, :])
                    length_data.append(X_learn[i, 1])
                    speed_data.append(X_learn[i, 0])
                    train_y.append(Y_learn[i])
                length_arima = sm.tsa.arima.ARIMA(length_data[:],order=(arima_p, arima_d,arima_q)).fit() # p、d、q
                predict_length=length_arima.predict(len(length_data),len(length_data) + 1)
                pre_length = predict_length[1]
                print(pre_length) ## 预测的晶体长度

                speed_arima = sm.tsa.arima.ARIMA(speed_data[:],order=(arima_p, arima_d,arima_q)).fit() # p、d、q
                predict_speed=speed_arima.predict(len(speed_data),len(speed_data) + 1)
                pre_speed = predict_speed[1]
                print(pre_speed) ## 预测的晶体拉速

                speed_arima = sm.tsa.arima.ARIMA(speed_data[:],order=(arima_p, arima_d,arima_q)).fit() # p、d、q
                predict_speed=speed_arima.predict(len(speed_data),len(speed_data) + 1)
                pre_speed = predict_speed[1]
                print(pre_speed) ## 预测的液面温度

                y_predict = OSVR.predict(train_data[i])[0,0]
                print(y_predict)  # 直径值的预测值
                np.append(y_pre, y_predict)

                rmse = np.sqrt(np.mean((train_y - y_pre)**2))
                # mae = abs(train_y - y_pre)
                print(rmse)
                OSVR.learn(train_data[i], train_y[i])
                next_predict = OSVR.predict(np.array([pre_speed, pre_length, train_data[i][2]]))[0,0]
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
            i += 1



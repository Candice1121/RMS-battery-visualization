import numpy as np


class RLS:
    def __init__(self, num_params, forgetting_factor):
        self.num_params = num_params
        self.forgetting_factor = forgetting_factor
        self.theta = np.zeros((num_params, 1))  # 参数估计
        self.P = np.eye(num_params)  # 参数估计协方差矩阵

    def update(self, x, y):
        x = np.reshape(x, (self.num_params, 1))
        y = np.reshape(y, (1, 1))

        # 预测输出
        y_pred = np.dot(x.T, self.theta)

        # 预测误差
        error = y - y_pred

        # 增益矩阵
        gain = np.dot(np.dot(self.P, x), 1.0 / (self.forgetting_factor + np.dot(np.dot(x.T, self.P), x)))

        # 参数估计更新
        self.theta = self.theta + np.dot(gain, error)

        # 参数估计协方差矩阵更新
        self.P = (1.0 / self.forgetting_factor) * (self.P - np.dot(np.dot(gain, x.T), self.P))

        return self.theta.flatten()
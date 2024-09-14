from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 单个特征的线性回归 使用sklearn进行训练

data = np.genfromtxt("data.csv", delimiter=",")
xdata = data[:, 0]
ydata = data[:, 1]
# plt.scatter(x_data, y_data)
# plt.show()

x_data = data[:, 0, np.newaxis]
y_data = data[:, 1, np.newaxis]
model = LinearRegression()
model.fit(x_data, y_data)

plt.plot(x_data, y_data, 'c.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()
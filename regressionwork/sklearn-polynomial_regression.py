import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# sklearn的多项式回归
data = np.genfromtxt("job.csv", delimiter=",")
x_data = data[1:, 1]
y_data = data[1:, 2]
# print(x_data)
# print(y_data)

x_data = x_data[:, np.newaxis]
y_data = y_data[:, np.newaxis]

# modle = LinearRegression()
# modle.fit(x_data, y_data)
#
# plt.plot(x_data, y_data, 'c.')
# plt.plot(x_data, modle.predict(x_data), 'r')
# plt.show()

# 定义多项式回归,degree的值可以调节多项式的特征
poly_reg = PolynomialFeatures(degree=8)
# 特征处理
x_poly = poly_reg.fit_transform(x_data)
print(x_poly)
# 定义回归模型
lin_reg = LinearRegression()
# 训练模型
lin_reg.fit(x_poly, y_data)

plt.plot(x_data, y_data, 'b.')
x_test = np.linspace(1, 10, 20)
x_test = x_test[:, np.newaxis]
# print(poly_reg.fit_transform(x_test))
plt.plot(x_data, lin_reg.predict(poly_reg.fit_transform(x_data)), c='r')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# sklearn的多元线性回归

data = np.genfromtxt('Delivery.csv', delimiter=",")
x_data = data[:, :-1]
y_data = data[:, -1]

model = LinearRegression()
model.fit(x_data, y_data)

# 学习之后的最佳特征值
print(model.coef_)

# 学习之后的截距
print(model.intercept_)

ax = plt.figure().add_subplot(111, projection = '3d')
ax.scatter(x_data[:,0], x_data[:,1], y_data, c = 'r', marker = 'o', s = 100) #点为红色三角形
x0 = x_data[:,0]
x1 = x_data[:,1]
# 生成网格矩阵
x0, x1 = np.meshgrid(x0, x1)
z = model.intercept_ + x0*model.coef_[0] + x1*model.coef_[1]
# 画3D图
ax.plot_surface(x0, x1, z)
# 设置坐标轴
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')

# 显示图像
plt.show()
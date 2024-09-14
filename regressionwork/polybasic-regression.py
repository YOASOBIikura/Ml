import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt('Delivery.csv', delimiter=",")
x_data = data[:, :-1]
y_data = data[:, -1]
# print(x_data)
# print(y_data)
# print(len(x_data))

# 定义学习率
Ir = 0.0001

# 设置特征数值初始值
b = 0
k = 0
l = 0

# 迭代次数
epochs = 1000

# 求损失值
def cost_function(x_data, y_data, b, k, l):
    Jsta = 0
    for i in range(0, len(x_data)):
        Jsta += ((b + k * x_data[i, 0] + l * x_data[i, 1]) - y_data[i]) ** 2

    return Jsta / float(len(x_data)) / 2.0


def gradient_descent(x_data, y_data, b, k, l, Ir, epochs):
     m = len(x_data)
     for i in range(0, epochs):
         temp_b = 0
         temp_k = 0
         temp_l = 0
         for j in range(0, len(x_data)):
             temp_b = (1/m) * ((b + k * x_data[j, 0] + l * x_data[j, 1]) - y_data[j])
             temp_k = (1/m) * x_data[j, 0] * ((b + k * x_data[j, 0] + l * x_data[j, 1]) - y_data[j])
             temp_l = (1/m) * x_data[j, 1] * ((b + k * x_data[j, 0] + l * x_data[j, 1]) - y_data[j])

         b -= Ir * temp_b
         k -= Ir * temp_k
         l -= Ir * temp_l

     return b, k, l

print("Starting b = {0}, k = {1}, l = {2}, cost = {3}".
      format(b, k, l, cost_function(x_data, y_data, b, k, l)))
print("Running......")
b, k, l = gradient_descent(x_data, y_data, b, k, l, Ir, epochs)
print("After {0} iteration b = {1}, k = {2}, l = {3}, cost = {4}".
      format(epochs, b, k, l, cost_function(x_data, y_data, b, k, l)))

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', marker='o', s=100)
x0 = x_data[:, 0]
x1 = x_data[:, 1]

# 生成网格矩阵
x0, x1 = np.meshgrid(x0, x1)
z = b + k * x0 + l * x1
print(z)
# 画3D图
ax.plot_surface(x0, x1, z)

# 设置坐标轴
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')

# 显示图像
plt.show()



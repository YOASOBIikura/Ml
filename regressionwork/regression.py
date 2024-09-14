import numpy as np
import matplotlib.pyplot as plt

# 单个特征的线性回归

data = np.genfromtxt("data.csv", delimiter=",")
x_data = data[:, 0]
y_data = data[:, 1]
plt.scatter(x_data, y_data)
plt.show()

# 学习率
Ir = 0.0001
# 截距
b = 0
# 斜率
k = 0
# 最大迭代次数 (? 这个迭代数似乎不能多于样本的数量)
epochs = 100

# 代价函数
def cost_function(x_data, y_data, b, k):
    # 总代价和
    Jsta = 0
    for i in range(0, len(x_data)):
        Jsta += ((k * x_data[i] + b) - y_data[i]) ** 2
    return Jsta / float(len(x_data)) / 2.0


# 梯度下降来求最合适的两个参数b,k
def gradient_descent(x_data, y_data, Ir, b, k, epochs):
    # 数组总数
    m = float(len(x_data))
    epochs_list = []
    cost_list = []
    for i in range(0, epochs):
        j = 0
        temp_b = 0
        temp_k = 0
        for j in range(0, len(x_data)):
            temp_b += (1 / m) * ((k * x_data[j] + b) - y_data[i])
            temp_k += (1 / m) * x_data[j] * ((k * x_data[j] + b) - y_data[i])
        b -= Ir * temp_b
        k -= Ir * temp_k
        epochs_list.append(i)
        cost_list.append(cost_function(x_data, y_data, b, k))
    return b, k, epochs_list, cost_list

cost = cost_function(x_data, y_data, b, k)
print("Starting b = {0}, k ={1}, cost = {2}".format(b, k, cost))
print("Runing.....")
b, k, epochs_list, cost_list = gradient_descent(x_data, y_data, Ir, b, k, epochs)
cost_1 = cost_function(x_data, y_data, b, k)
print("After {0} iterations b = {1}, k = {2}, cost = {3}".format(epochs, b, k, cost_1))
print("epochs:")
print(epochs_list)
print("cost：")
print(cost_list)


# 画图
# plt.xlabel("X valve")
# plt.ylabel("y value")
# plt.plot(x_data, y_data, 'c.')
# plt.plot(x_data, k*x_data+b, 'r')
# plt.show()

f1 = plt.figure()
plt.title("predict")
plt.xlabel("X valve")
plt.ylabel("y value")
plt.plot(x_data, y_data, 'c.')
plt.plot(x_data, k*x_data+b, 'r')
plt.show()

f2 = plt.figure()
plt.title("cost")
plt.xlabel("epochs")
plt.ylabel("cost")
plt.plot(epochs_list, cost_list, 'b')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plot

# 线性神经网络做非线性分类

X = np.array([[1, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 1],
              [1, 1, 0, 1, 0, 0],
              [1, 1, 1, 1, 1, 1]])

# 标签
Y = np.array([-1, 1, 1, -1])

# 初始化权值
W = (np.random.random(6) - 0.5) * 2
# 设置学习率
lr = 0.11
# 计算迭代次数
n = 0
# 神经网络输出
out = 0

def update():
    global X, Y, W, lr, n
    n+=1
    out = np.dot(X, W.T)
    W_C = lr*((Y - out.T).dot(X)) / int(X.shape[0])
    W = W + W_C

for i in range(1000):
    update()

# 正样本
x1 = [0, 1]
y1 = [1, 0]
# 负样本
x2 = [0, 1]
y2 = [0, 1]

def calculate(x, root):
    a = W[5]
    b = W[2] + x*W[4]
    c = W[0] + x*x*W[3]
    if root == 1:
        return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    if root == 2:
        return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)

x_data = np.linspace(-1, 2)

plt.figure()
plt.scatter(x1, y1, marker='o', c='r')
plt.scatter(x2, y2, marker='x', c='y')
plt.plot(x_data, calculate(x_data, 1), 'r')
plt.plot(x_data, calculate(x_data, 2), 'b')
plt.show()

print(np.dot(X, W.T))
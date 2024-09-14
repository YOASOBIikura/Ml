import numpy as np
import matplotlib.pyplot as plt
# 正确率 召回率 等指标
from sklearn.metrics import classification_report
# 数据标准化处理
from sklearn import preprocessing
# 数据是否需要标准化
scale = False

# 梯度下降法的逻辑回归
data = np.genfromtxt("LR-testSet.csv", delimiter=",")
x_data = data[:, :-1]
y_data = data[:, -1]

# print(x_data)
# print(y_data)

# 画出散点图
def plot():
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for i in range(0, len(x_data)):
        if y_data[i] == 0:
            x0.append(x_data[i, 0])
            y0.append(x_data[i, 1])
        else:
            x1.append(x_data[i, 0])
            y1.append(x_data[i, 1])
    # 画图
    scatter1 = plt.scatter(x0, y0, c='b', marker='o')
    scatter2 = plt.scatter(x1, y1, c='r', marker='x')

    # 画图例
    plt.legend(handles = [scatter1, scatter2], labels = ['label0', 'label1'], loc = 'best')

# plot()
# plt.show()

y_data = data[:, -1, np.newaxis]

X_data = np.concatenate((np.ones((100, 1)), x_data), axis=1)

def sigmoid(x):

    return 1.0/(1+np.exp(-x))

def cost_function(xMat, yMat, ws):
    # np.multiply()矩阵点乘，在一次性相加就省略多次循环求和了
    left = np.multiply(yMat, np.log(sigmoid(xMat*ws)))
    right = np.multiply(1-yMat, np.log(1-sigmoid(xMat*ws)))
    # np.sum()矩阵中所有元素相加
    return np.sum(left + right) / -(len(xMat))

def gradAscent(xArr, yArr):
    # 数据标准化
    if scale == True:
        xArr = preprocessing.scale(xArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    lr = 0.001
    epochs = 10000
    costlist = []
    m, n = np.shape(xMat)
    # 初始化权值
    ws = np.mat(np.ones((n, 1)))

    for i in range(0, epochs+1):
        # xMat和weights矩阵相乘
        h = sigmoid(xMat*ws)
        # 计算误差
        ws_grad = xMat.T*(h-yMat)/m
        ws = ws - lr * ws_grad

        if i % 50 == 0:
            costlist.append(cost_function(xMat, yMat, ws))

    return ws, costlist


# 训练模型，得到权值和cost值的变化
ws, costlist = gradAscent(X_data, y_data)
print(ws)

if scale == False:
    # 画图决策边界
    plot()
    x_test = [[-4], [3]]
    y_test = (-ws[0] - x_test*ws[1])/ws[2]
    plt.plot(x_test, y_test, 'k')
    plt.show()

# 画图 loss值的变化
x = np.linspace(0, 10000, 201)
plt.plot(x, costlist, c='r')
plt.title('Train')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()

# 预测
def predict(x_data, ws):
    if scale == True:
        x_data = preprocessing.scale(x_data)
    xMat = np.mat(x_data)
    ws = np.mat(ws)
    return [1 if x >= 0.5 else 0 for x in sigmoid(xMat*ws)]

prediction = predict(X_data, ws)

print(classification_report(y_data, prediction))

























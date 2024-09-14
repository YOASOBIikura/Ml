import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
# 数据是否需要标准化
scale = False

# 非线性的逻辑回归

data = np.genfromtxt("LR-testSet2.txt", delimiter=",")
x_data = data[:, :-1]
y_data = data[:, -1, np.newaxis]

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

    scatter0 = plt.scatter(x0, y0, c='b', marker='o')
    scatter1 = plt.scatter(x1, y1, c='r', marker='x')

    plt.legend(handles=[scatter0, scatter1], labels=['label0', 'label1'], loc='best')


# plot()
# plt.show()

# 定义多项式回归,degree的值可以调节多项式的特征
poly_reg = PolynomialFeatures(degree=3)
# 特征处理
print(x_data[1])
x_poly = poly_reg.fit_transform(x_data)
print(x_poly[1])

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def cost_function(xMat, yMat, ws):
    left = np.multiply(yMat, np.log(sigmoid(xMat*ws)))
    right = np.multiply(1-yMat, np.log(1-sigmoid(xMat*ws)))
    return np.sum(left+right) / -(len(xMat))

def gradAscent(xArr, yArr):

    if scale == True:
        xArr = preprocessing.scale(xArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    lr = 0.03
    epochs = 50000
    costList = []
    # 计算数据的列数，有几列就有几个权值
    m, n = np.shape(xMat)
    ws = np.mat(np.ones((n, 1)))

    for i in range(0, epochs+1):
        h = sigmoid(xMat*ws)
        ws_grad = xMat.T*(h-yMat)/m
        ws = ws - lr*ws_grad

    if i % 50 == 0:
        costList.append(cost_function(xMat, yMat, ws))
    return ws, costList

ws, costList = gradAscent(x_poly, y_data)
# print(ws)
# print(len(ws))

x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

# 生成网格矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
# print(xx.ravel())
# print(xx.ravel().shape)
# print(np.c_[xx.ravel(), yy.ravel()])
# print("-----------------------------------------")
print(poly_reg.fit_transform(np.c_[xx.ravel(), yy.ravel()]).shape)
z = sigmoid(poly_reg.fit_transform(np.c_[xx.ravel(), yy.ravel()]).dot(np.array(ws)))
for i in range(len(z)):
    if z[i] > 0.5:
        z[i] = 1
    else:
        z[i] = 0


z = z.reshape(xx.shape)
# print(z)
# print(z.shape)

cs = plt.contour(xx, yy, z)
plot()
plt.show()

# # 预测
# def predict(x_data, ws):
# #     if scale == True:
# #         x_data = preprocessing.scale(x_data)
#     xMat = np.mat(x_data)
#     ws = np.mat(ws)
#     return [1 if x >= 0.5 else 0 for x in sigmoid(xMat*ws)]
#
# predictions = predict(x_poly, ws)
#
# print(classification_report(y_data, predictions))
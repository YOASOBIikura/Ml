import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn import preprocessing
# 数据是否需要标准化
scale = False


data = np.genfromtxt("LR-testSet.csv", delimiter=",")
x_data = data[:, :-1]
y_data = data[:, -1]

def plot():
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    # 切分不同类别的数据
    for i in range(len(x_data)):
        if y_data[i] == 0:
            x0.append(x_data[i, 0])
            y0.append(x_data[i, 1])
        else:
            x1.append(x_data[i, 0])
            y1.append(x_data[i, 1])

    # 画图
    scatter0 = plt.scatter(x0, y0, c='b', marker='o')
    scatter1 = plt.scatter(x1, y1, c='r', marker='x')
    plt.legend(handles=[scatter0, scatter1], labels=['label0', 'label1'], loc='best')

# plot()
# plt.show()

logistic = linear_model.LogisticRegression()
logistic.fit(x_data, y_data)

if scale == False:
    # 画图决策边界
    plot()
    x_test = np.array([[-4], [3]])
    y_test = (-logistic.intercept_ - x_test*logistic.coef_[0][0])/logistic.coef_[0][1]
    plt.plot(x_test, y_test, 'k')
    plt.show()

prediction = logistic.predict(x_data)

print(classification_report(y_data, prediction))

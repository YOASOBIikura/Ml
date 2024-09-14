import numpy as np
import matplotlib.pyplot as plt
import operator

# 已知分类数据
x1 = np.array([3, 2, 1])
y1 = np.array([104, 100, 81])
x2 = np.array([101, 99, 98])
y2 = np.array([10, 5, 2])
scatter1 = plt.scatter(x1, y1, c='r')
scatter2 = plt.scatter(x2, y2, c='b')

# 未知数据
x = np.array([18])
y = np.array([90])
scatter3 = plt.scatter(x, y, c='k')


#画图例
plt.legend(handles=[scatter1, scatter2, scatter3], labels=['labelA', 'labelB', 'X'], loc='best')

plt.show()

x_data = np.array([
    [3, 104],
    [2, 100],
    [1, 18],
    [101, 10],
    [99, 5],
    [81, 2]
])

y_data = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
x_test = np.array([18, 90])

# 计算样本数量
x_data_size = x_data.shape[0]
# diffMat = np.tile(x_test, (x_data_size, 1))
# print(diffMat)
# np.tile是该元素复制为x_data_size行， 1列 如下:
# [[18 90]
#  [18 90]
#  [18 90]
#  [18 90]
#  [18 90]
#  [18 90]]
diffMat = np.tile(x_test, (x_data_size, 1)) - x_data

sqDiffMat = diffMat ** 2

sqDistances = sqDiffMat.sum(axis=1)

distances = sqDistances ** 0.5

sortedDistances = distances.argsort()

classCount = {}

# 设置K
k = 5
for i in range(5):
    # 获取标签
    votelabel = y_data[sortedDistances[i]]
    # 统计标签数量
    classCount[votelabel] = classCount.get(votelabel, 0) + 1

# 根据operator.itemgetter(1)-第1个值对classCount排序，然后再取倒序
sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

# 获取数量最多的标签， 也即是对测试点的分类
knnclass = sortedClassCount[0][0]
print(knnclass)
from sklearn import svm

x = [[3, 3], [4, 3], [1, 1]]
y = [1, 1, -1]

model = svm.SVC(kernel='linear')
model.fit(x, y)

# 打印支持向量
print(model.support_vectors_)

model.predict([[4, 3]])

# 参数
print(model.coef_)
# 截距
print(model.intercept_)


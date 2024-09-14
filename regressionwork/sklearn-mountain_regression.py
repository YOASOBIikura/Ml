import numpy as np
import sklearn.linear_model as Model
import matplotlib.pyplot as plt

# sklearn的岭回归的实现

data = np.genfromtxt("longley.csv", delimiter=",")
x_data = data[1:, 2:]
y_data = data[1:, 1]

# print(x_data)
# print(y_data)

x_test = np.linspace(0.001, 1)
model = Model.RidgeCV(alphas=x_test, store_cv_values=True)
model.fit(x_data, y_data)

# 岭系数
print(model.alpha_)
# loss值
# print(model.cv_values_)

# print(model.cv_values_.mean(axis=0))
plt.plot(x_test, model.cv_values_.mean(axis=0))
# 选取的岭回归系数值的位置
plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)), 'ro')
plt.show()

print(model.predict(x_data[2, np.newaxis]))

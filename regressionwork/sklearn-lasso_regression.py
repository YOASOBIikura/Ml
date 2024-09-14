import numpy as np
import sklearn.linear_model as Model
import matplotlib.pyplot as plt

data = np.genfromtxt("longley.csv", delimiter=",")
x_data = data[1:, 2:]
y_data = data[1:, 1]

# print(x_data)
# print(y_data)

model = Model.LassoCV()
model.fit(x_data, y_data)

# lasso系数
print(model.alpha_)
# 相关系数
print(model.coef_)

print(model.predict(x_data[-2, np.newaxis]))
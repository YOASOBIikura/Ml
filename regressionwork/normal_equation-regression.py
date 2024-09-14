import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data.csv", delimiter=",")
x_data = data[:, 0, np.newaxis]
y_data = data[:, 1, np.newaxis]
# plt.scatter(x_data, y_data)
# plt.show()


# np.concatenate((np.ones((100,1)),x_data),axis=1)
X_data = np.concatenate((np.ones((100, 1)), x_data), axis=1)
# print(X_data)

def weights(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix cannot do inverse")
        return
    ws = xTx.I * xMat.T * yMat
    return ws

ws = weights(X_data, y_data)

print("This temp_0 = {0}, temp_1 = {1}".format(ws[0], ws[1]))

x_test = np.array([[20], [80]])
print(x_test)
y_test = ws[0] + x_test*ws[1]
plt.plot(x_data, y_data, 'c.')
plt.plot(x_test, y_test, 'r')
plt.show()

# x_test = np.array([[20],[80]])
# print(x_test)
# y_test = ws[0] + x_test*ws[1]
# plt.plot(x_data, y_data, 'b.')
# plt.plot(x_test, y_test, 'r')
# plt.show()
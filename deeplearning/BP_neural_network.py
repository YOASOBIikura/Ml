import numpy as np

X = np.array([[1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

Y = np.array([[0, 1, 1, 0]])

# 权值初始化，去值范围-1到1
# 第一层影藏层网络
V = np.random.random((3, 4))*2-1
# 输出层网络
W = np.random.random((4, 1))*2-1
# print(V)
# print(W)
# 设置学习率
lr = 0.11

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

def update():
    global X, Y, W, V, lr

    L1 = sigmoid(np.dot(X, V))
    L2 = sigmoid(np.dot(L1, W))

    L2_delta = (Y.T - L2) * dsigmoid(L2)
    L1_delta = L2_delta.dot(W.T) * dsigmoid(L1)

    W_C = lr*L1.T.dot(L2_delta)
    V_C = lr*X.T.dot(L1_delta)

    W = W + W_C
    V = V + V_C

for i in range(20000):
    update()
    if i % 500 == 0:
        L1 = sigmoid(np.dot(X, V))
        L2 = sigmoid(np.dot(L1, W))
        print('Error:', np.mean(np.abs(Y.T - L2)))

print(W)
print(V)


def judge(x):
    if x>=0.5:
        return 1
    else:
        return 0

for i in map(judge,L2):
    print(i)

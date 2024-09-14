import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
# MultinomialNB多项式模型，BernoulliNB伯努利模型，GaussianNB高斯模型

# 载入数据
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

model = GaussianNB()
model.fit(x_train, y_train)

print(classification_report(model.predict(x_test), y_test))

print(confusion_matrix(model.predict(x_test), y_test))
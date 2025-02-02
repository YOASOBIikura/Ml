from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np

iris = datasets.load_iris()
# 载入数据
# 只要1,2列的特征
x_data, y_data = iris.data[:, 1:3], iris.target

# 定义三个不同的分类器
# n_neighbors=1选择最邻近的一个点
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = DecisionTreeClassifier()
clf3 = LogisticRegression()

# 定义一个次级分类器
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)

for clf, label in zip([clf1, clf2, clf3, sclf], ['KNN', 'Decision Tree', 'LogisticRegression',
                                                 'StackingClassifier']):
    scores = model_selection.cross_val_score(clf, x_data, y_data, cv=3, scoring='accuracy')
    print("Accuracy: %0.2f [%s]" % (scores.mean(), label))
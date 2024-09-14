from sklearn.feature_extraction import  DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv
import graphviz

# 读入数据
Dtree = open('AllElectronics.csv')
reader = csv.reader(Dtree)

# 读取第一行数据
headers = reader.__next__()
# print(headers)

# 定义特征列表和标签列表
featureList = []
labelList = []

for row in reader:
    labelList.append(row[-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    # 把数据字典存入List
    featureList.append(rowDict)

# print(featureList)

# 把数据转换成0/1表示
vec = DictVectorizer()
x_data = vec.fit_transform(featureList).toarray()
# print("x_data:" + str(x_data))

# 打印标签
# print("labelList:" + str(labelList))

# 把数据转换成0/1表示
lb = preprocessing.LabelBinarizer()
y_data = lb.fit_transform(labelList)
# print("y_data:" + str(y_data))

# 创建决策树模型
model = tree.DecisionTreeClassifier(criterion='entropy')
# 输入数据建立模型
model.fit(x_data, y_data)

# 测试
x_test = x_data[0]

predict = model.predict(x_test.reshape(1, -1))
print("predict:" + str(predict))

dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=vec.get_feature_names(),
                                class_names=lb.classes_, filled=True,
                                rounded=True, special_characters=True)


graph = graphviz.Source(dot_data)
print(1)
graph.render('computer')
print(2)
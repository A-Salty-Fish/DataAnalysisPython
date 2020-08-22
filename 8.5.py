# coding:utf-8
import numpy as np
from sklearn import tree
import graphviz

X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = [0, 1, 1, 1, 2, 3, 3, 4]
clf = tree.DecisionTreeClassifier()  # 创建决策树分类器
clf.fit(X, y)  # 拟合

clf.predict([[1, 0, 0]])  # 分类

dot_data = tree.export_graphviz(clf, out_file="tree.dot")  # 导出决策树

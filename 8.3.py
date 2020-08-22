from sklearn import linear_model   # 导入线型模型模块
regression = linear_model.LinearRegression()
# 创建线型回归模型
X = [[3], [8]]                     # 观察值的x坐标
y = [1, 2]                         # 观察值的y坐标
regression.fit(X, y)               # 拟合
regression.score([[6],[7],[8],[9],[10]],     # 对模型进行评分
                     [1.6,1.8,1.99,2.2,2.401])   # 结果越大越好

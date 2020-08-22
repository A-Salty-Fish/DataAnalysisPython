from sklearn.linearmodel import Ridge
ridgeRegression = Ridge(alpha=10)       # 创建岭回归模型
                                            # 设置约束项系数为10
X = [[3], [8]]
y = [1, 2]
ridgeRegression.fit(X, y)               # 拟合

ridgeRegression.predict([[6]])          # 预测

ridgeRegression.coef_                   # 查看回归系数

ridgeRegression.intercept_              # 截距
ridgeRegression = Ridge(alpha=1.0)      # 设置约束项系数为1.0
ridgeRegression.fit(X, y)

ridgeRegression.coef

ridgeRegression.intercept_
ridgeRegression.predict([[6]])

ridgeRegression = Ridge(alpha=0.0)      # 约束项系数为0
                                            # 等价于线性回归
ridgeRegression.fit(X, y)

ridgeRegression.coef_

ridgeRegression.intercept

ridgeRegression.predict([[6]])

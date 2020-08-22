from sklearn.linearmodel import Ridge
ridgeRegression = Ridge(alpha=10)       # ������ع�ģ��
                                            # ����Լ����ϵ��Ϊ10
X = [[3], [8]]
y = [1, 2]
ridgeRegression.fit(X, y)               # ���

ridgeRegression.predict([[6]])          # Ԥ��

ridgeRegression.coef_                   # �鿴�ع�ϵ��

ridgeRegression.intercept_              # �ؾ�
ridgeRegression = Ridge(alpha=1.0)      # ����Լ����ϵ��Ϊ1.0
ridgeRegression.fit(X, y)

ridgeRegression.coef

ridgeRegression.intercept_
ridgeRegression.predict([[6]])

ridgeRegression = Ridge(alpha=0.0)      # Լ����ϵ��Ϊ0
                                            # �ȼ������Իع�
ridgeRegression.fit(X, y)

ridgeRegression.coef_

ridgeRegression.intercept

ridgeRegression.predict([[6]])

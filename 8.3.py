from sklearn import linear_model   # ��������ģ��ģ��
regression = linear_model.LinearRegression()
# �������ͻع�ģ��
X = [[3], [8]]                     # �۲�ֵ��x����
y = [1, 2]                         # �۲�ֵ��y����
regression.fit(X, y)               # ���
regression.score([[6],[7],[8],[9],[10]],     # ��ģ�ͽ�������
                     [1.6,1.8,1.99,2.2,2.401])   # ���Խ��Խ��

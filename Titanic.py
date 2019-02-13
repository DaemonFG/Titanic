"""
信息熵(单位：比特)
公式：
H(X) = ∑x∈X P(x)*logP(X)

信息增益：表示得知特征X的信息而是的类Y的信息的不确定性减少的程度
公式：
g(D, A) = H(D) - H(D|A)
H(D):初始信息熵大小(计算时是否都要加，外面取负才能得正)
H(D|A):A条件下信息熵的大小

决策树常用算法
ID3 信息增益 最大的准则

C4.5 信息增益比 最大的准则

CART
回归树：平方误差 最小
分类树：基尼系数 最小的准则 在sklearn中可以选择划分的默认原则

class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, max_depth=None,random_state=None)
决策树分类器
criterion:默认是’gini’系数，也可以选择信息增益的熵’entropy’
max_depth:树的深度大小
random_state:随机数种子
method:
decision_path:返回决策树的路径

决策树优点：
1、简单的理解和解释，树木可视化。
2、需要很少的数据准备，其他技术通常需要数据归一化。

缺点：
1、决策树学习者可以创建不能很好地推广数据的过于复杂的树，这被称为过拟合。
2、决策树可能不稳定，因为数据的小变化可能会导致完全不同的树被生成。

改进：
1、减枝cart算法
2、随机森林

案例：泰坦尼克号生还预测
在泰坦尼克号数据中描述了泰坦尼克号上的个别乘客的生存状态。它包含了乘客的一半的实际年龄。
关于泰坦尼克号旅客的数据的主要来源是百科全书Titanica。这里使用的数据集是由各种研究人员开始的。
其中包括许多研究人员创建的旅客名单，由Michael A. Findlay编辑。
我们提取的数据集中的特征是票的类别，存活，乘坐班，年龄，登陆，home.dest，房间，票，船和性别。
乘坐班是指乘客班（1，2，3），是社会经济阶层的代表。
其中age数据存在缺失。

数据来源：http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt

分析：
1、获取数据
2、将pclass, age, sex作为特征值，将是否生还作为目标值
3、处理缺失值
4、将数据集分割为测试集和训练集
5、进行one-hot编码(one hot编码是将类别变量转换为机器学习算法易于利用的一种形式的过程。)
6、网格搜索交叉验证出最佳深度5
7、生还预测，模型评估
8、导出决策树的结构
"""
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
import pandas as pd


def desicion():
    """
    决策树对泰坦尼克号进行生还预测
    :return: None
    """
    # 获取数据
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 处理数据，找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']

    # 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理(特征工程，类别，one_hot编码)
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))  # orient="records"一行转换成一个字典
    x_test = dict.transform(x_test.to_dict(orient="records"))
    print(dict.get_feature_names())

    # 用决策树进行预测
    dec = DecisionTreeClassifier(max_depth=5)
    dec.fit(x_train, y_train)

    # 预测准确率
    print("预测准确率:", dec.score(x_test, y_test))

    # 构造一些参数值进行搜索
    # param = {"max_depth": [3, 5, 7, 9]}

    # 进行网格搜索
    # gc = GridSearchCV(dec, param_grid=param, cv=10)
    # gc.fit(x_train, y_train)

    # 预测准确率
    # print("在测试集上准确率：", gc.score(x_test, y_test))
    # print("在交叉验证中的最好结果：", gc.best_score_)
    # print("选择最好的模型：", gc.best_estimator_)
    # print("每个超参数每次交叉验证的结果：", gc.cv_results_)

    # 导出决策树结构
    export_graphviz(dec, out_file="./titanic.dot",
                    feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male'])

    return None


if __name__ == '__main__':
    desicion()

"""
集成学习:
通过建立几个模型组合的来解决单一预测问题。它的工作原理是生成多个分类器/模型，各自独立地学习和作出预测。
这些预测最后结合成单预测，因此优于任何一个单分类的做出预测。

随机森林:
包含多个决策树的分类器，并且其输出的类别是由个别树输出的类别的众数而定。

单个树建立过程：
1、随机在N个样本中选择一个样本，重复N次，样本可能存在重复，但这是需要的。
2、随机在M个特征中选择m个特征，m<<M。
3、建立10棵决策树，样本、特征大多不一样，采用随机又放回的抽样(bootstrap抽样)，如果不放回，结果将是有偏的、片面的。

API：
class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, bootstrap=True, random_state=None)
随机森林分类器
n_estimators：integer，optional（default = 10） 森林里的树木数量
criteria：string，可选（default =“gini”）分割特征的测量方法
max_depth：integer或None，可选（默认=无）树的最大深度
bootstrap：boolean，optional（default = True）是否在构建树时使用放回抽样

优点：
1、在当前所有算法中，具有极好的准确率
2、能够有效地运行在大数据集上
3、能够处理具有高维特征的输入样本，而且不需要降维
4、能够评估各个特征在分类问题上的重要性
5、对于缺省值问题也能够获得很好得结果
6、参数选的好几乎没缺点
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
import pandas as pd


def desicion_rf():
    """
    随机森林对泰坦尼克号进行生还预测
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
    # print(dict.get_feature_names())

    # 随机森林进行预测(超参数调优)
    rf = RandomForestClassifier()

    # 构造一些参数值进行搜索
    param = {"n_estimators": [25, 30, 40, 50], "max_depth": [2, 3, 4, 5 ]}

    # 进行网格搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)

    # 预测准确率
    print("在测试集上准确率：", gc.score(x_test, y_test))
    print("在交叉验证中的最好结果：", gc.best_score_)
    print("选择最好的模型：", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果：", gc.cv_results_)


if __name__ == '__main__':
    desicion_rf()

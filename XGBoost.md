# 什么是XGBoost

XGBoost是梯度提升决策树算法的一个实现，XGBoost模型比随机森林等技术需要更多的知识和模型调整

# 代码实现

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
data=pd.read_csv("...")
y=data.SalePrice
X=data.drop(['SalePrice'],axis=1).select_dtypes(exclude=['object'])
train_X,test_X,train_y,test_y=train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

from xgboost import XGBRegressor
my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
```

# 模型调整

XGBoost有几个参数可以极大地影响你的模型的准确性和训练速度。

## n_estimators  和  early_stopping_rounds

n_estimators指的是要经历多少次上述的建模循环。

在欠拟合与过拟合图中，n_estimators使你进一步向右移动。太低的值会导致欠拟合，这是对训练数据和新数据的不准确预测。太大的数值会导致过度拟合，即对训练数据的预测准确，但对新数据的预测不准确（这正是我们所关心的）。你可以用你的数据集进行实验，找到理想的值。

典型的值在100-1000之间，不过这在很大程度上取决于学习率。

参数early_stopping_rounds提供了一种自动寻找理想值的方法。使模型在验证分数停止提高时停止迭代，即使我们没有达到n_estimators的硬停止。为n_estimators设置一个较高的值，然后使用early_stopping_rounds来找到停止迭代的最佳时间。

由于随机机会有时会导致单轮验证分数没有提高，你需要指定一个数字，说明在停止之前允许多少轮连续的恶化。假设，我们在验证分数连续5轮恶化后停止。

```python
my_model = XGBRegressor(n_estimators=1000)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
```

## 学习率

我们不是通过简单地将每个组件模型的预测值相加来获得预测值，而是将每个模型的预测值乘以一个小数，然后再将它们加进去。这意味着我们添加到集合体中的每一棵树对我们的帮助都很小。在实践中，这减少了模型的过拟合倾向。

因此，可以使用更高的n_estimators值，而不会出现过拟合。如果使用早期停止，树的数量将被自动设置。

一般来说，小的学习率（和大量的估计器）会产生更准确的XGBoost模型，尽管它也会花费更长的时间来训练模型，因为它在循环中做了更多的迭代。

```python
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
```

## n_jobs

在较大的数据集上，运行时间是一个考虑因素，可以使用并行性来快速建立模型。通常将参数n_job设置为等于你机器上的核心数量。在较小的数据集上，这没有帮助。所得到的模型也不会更好，所以对拟合时间进行微观优化通常只是一种干扰。但是，在大数据集上，这很有用，否则你会在拟合命令中花费很长的时间等待。


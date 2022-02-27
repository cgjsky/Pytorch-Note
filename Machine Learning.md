```python
import pandas as pd
from sklearn.model_selection import train_test_split
```

## 解决缺失值

```python
isnull().sum() 很方便的统计列中na的个数
#统计所有列的缺失值
missing_data_by_columns=x.isnull().sum()
#统计大于0的列
missing_data_lagger_0=missing_data_by_columns[missing_data_by_columns>0]
```



```python
#一般用到dropna函数
DataFrme.dropna(axis=0,how=’any’,thresh=None,subset=None,inplace=False)
axis: 默认axis=0。0为按行删除,1为按列删除
how: 默认 ‘any’。 ‘any’指带缺失值的所有行/列;'all’指清除一整行/列都是缺失值的行/列
thresh: int,保留含有int个非nan值的行
subset: 删除特定列中包含缺失值的行或列
inplace: 默认False，即筛选后的数据存为副本,True表示直接在原数据上更改
```



### 1.删除有缺失值的列

很少使用

```python
#删除特定列中na的行
x_full=pd.read_csv("...",index_col='Id')
x_full.dropna(axis=0,subset=[""],inplace=True)
y=x_full.SalePrice
x_full.drop(["SalePrice"],axis=1,inplace=True)

#也可以筛选出含na的列，然后drop
cols_with_missing=[cols for cols in X_train.columns
                   if X_train[col].isnull().any()]
X_train.drop(cols_with_missing,axis=1)
```



### 2.归纳

用一些值去填补缺失值

一般使用sklearn.Imputer中的SimpleImputer去填充缺失值

```python
from sklearn.Imputer import SimpleImputer
class sklearn.impute.SimpleImputer(*,missing_values=nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)[source]
###
missing_values,也就是缺失值是什么，一般情况下缺失值是空值,也就是np.nan

strategy:也就是你采取什么样的策略去填充空值，总共有4种选择。分别是mean,median, most_frequent,以及constant,如果是constant,则可以将空值填充为自定义的值，这就要涉及到后面一个参数了，也就是fill_value。如果strategy='constant',则填充fill_value的值。

copy:则表示对原来没有填充的数据的拷贝。

add_indicator:如果该参数为True，则会在数据后面加入n列由0和1构成的同样大小的数据，0表示所在位置非空，1表示所在位置为空。相当于一种判断是否为空的索引。
#实例,先定义一个imp方法，再进行数据transform
from sklearn.Imputer import SimpleImputer
imp_mean=SimpleImputer(missing_values=np.nan,strategy='mean')  
imp_mean=imp_mean.fit_transform(age)
```

## 

## 分类变量编码

即变量的结果只取固定的某些值

考虑一项调查，询问你吃早餐的频率并提供四个选项。"从不"、"很少"、"大部分时间"、"每天"。在这种情况下，数据是分类的，因为回答属于一组固定的类别。

首先应该确定每一列中unique的值

```python
#统计每一列unique的值
object_unique=list(map(lambda col: X_train[col].nunique(),object_cols))
#与列进行组合，zip用来组合,dict用来包装
d = dict(zip(object_cols, object_nunique))
#排序展示
sorted(d.items(), key=lambda x: x[1])
```

处理方法

### 序数编码

顺序编码将每个独特的值分配给一个不同的整数。即类别存在一定的顺序

```python
#选择出要进行编码的列
object_cols=[cols for cols in X_train.columns if X_train[cols].dtype=="object"]
```



Scikit-learn有一个OrdinalEncoder类，可以用来获取序数编码。我们对分类变量进行循环，并对每一列分别应用序数编码器。

```python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
#保持[object_cols]可以使结果仍具有相应的columns
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
```



### one-hot编码

类别不存在顺序，如果分类变量有大量的值，one-hot编码通常表现不佳

```python
from sklearn.preprocessing import OneHotEncoder
#设置参数避免在验证数据中包含训练数据中没有的类时出现错误，列以numpy数组形式返回
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
#转成df形式，方便后续粘贴
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
#要注意的是，one-hot编码removed index，所以要加回来
OH_cols_train.index = X_train.index
#之后将原列drop，替换为one-hot列
num_X_train = X_train.drop(object_cols, axis=1)
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
```

## 

## 交叉验证

在交叉验证中，我们在不同的数据子集上运行我们的建模过程，以获得对模型质量的多种测量。例如，我们可以先将数据分成5块，每块占整个数据集的20%。在这种情况下，我们说我们把数据分成了5个 "折"。然后，我们为每个折叠运行一个实验。

一般情况下，小数据集使用交叉验证，对于较大数据集，单一的验证集已经足够了

我们用scikit-learn的cross_val_score()函数获得交叉验证的分数。

```python
from sklearn.model_selection import cross_val_score
cross_val_score(model_name,X,y，cv=k)
返回每次交叉验证运行时估算器得分的数组。
```



## 
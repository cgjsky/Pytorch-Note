## Introduction

pandas主要有两个数据类型，DataFrame与Series

Series可以理解为一维数组，除了数组元素外还包含index索引.

Series可以直接用list或np.array转化，这时index默认为0,1,2...，也可以用index=来指定索引

```python
a=[1,2,3,4]
b=np.array(a)
c=pd.Series(b)
type(a) list
type(b) numpy.ndarray
type(c) pandas.core.series.Series
```

DataFrame可以理解为是一个表格型的数据结构。它提供有序的列和不同类型的列值。

```python
a=pd.DataFrame({"a":[1,2,3],"b":[1,2,3]},index=["x","y","z"])
```

## 数据筛选

Pandas可以通过索引和属性进行选择，但是一般用loc与iloc操作

loc 和 iloc 都是行优先

loc是根据标签属性进行选择，iloc则是把dataframe当作矩阵，通过行列数字确定位置

值得注意的：iloc的0:10不包含10，而loc包含

```python
a.loc[(a.country=="Italy")&(a.point>=9)]
#多个条件用&、｜、连接
```

```python
#筛选出符合要求的行列
ind=[0,1,10,100]
col=["country","province","region_1","region_2"]
df = reviews.loc[ind,col]
```

```
describe() 对数据进行简要分析
value_counts() 每个value出现的次数进行统计
```

## 数据操作

```python
median()
unique()
value_counts()
mean()
idxmax()/argmax() 返回最大值的索引
#一般先通过计算找出索引，再loc找出符合要求的数据
a=(r.points / r.price).idxmax()
b = r.loc[a,"title"]

```

apply map对df操作

apply：应用在DataFrame的行或列中，也可以应用到单独一个Series的每个元素中

map：应用在单独一个Series的每个元素中

```python
DataFrame.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)
#axis=0按列进行，axis=1按行进行
df.apply() 对df操作
df.country.map() 对一列进行操作
```

```python
n_t=r.description.apply(lambda d:"tropical" in d).sum()
n_f=r.description.apply(lambda d:"fruity" in d).sum()
dc = pd.Series([n_t, n_f], index=['tropical', 'fruity'])
```

## 数据分组与排序

groupby 对df进行切片划分

agg   通常用于调用groupby（）函数之后，对数据做一些聚合操作，包括sum，min,max以及其他一些聚合函数

```python
c = r.groupby(['country', 'province']).description.agg([len])
```

groupby操作后会出现multiindex，此时可以重置为单index

```python
c.reset_index()
```

一般使用sort_values来进行sort

```python
c.sort_values(by='len')
```

sort_values()默认为升序排序，即最低值优先。然而，大多数情况下，我们希望采用降序排序，即高的数字先排序。

```python
c.sort_values(by='len', ascending=False)
```

如果想按索引排序

```python
c.sort_index()
```

## 数据类型

```python
df.dtypes
```

通过使用astype()函数，可以将一个类型的列转换为另一个类型，只要这种转换有意义。例如，我们可以将点数列从其现有的int64数据类型转换为float64数据类型。

```python
a.astype('float64')
```

## 缺失值

输入的缺失值被赋予NaN值，即 "Not a Number "的缩写。由于技术原因，这些NaN值总是属于float64

```python
df[pd.isnull(df.country)]
```

可以使用fillna()来替换缺失值

```python
df.country.fillna("Unknown")
```

也可以replace()替换

```python
df.country.replace("AAA", "BBB")
```

## index/column重命名

```python
df.rename(columns={"points":"scores"})
df.rename(index={0: 'firstEntry', 1: 'secondEntry'})
```

## 组合

concat()  join() 和merge()进行DataFrames和/或Series的组合

最简单的组合方法是concat()。给定一个元素的列表，这个函数将沿着一个轴线把这些元素压在一起。

当我们在不同的DataFrame或Series对象中拥有相同字段（列）的数据时，这很有用。
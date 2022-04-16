## Introduction

pandas主要有两个数据类型，DataFrame与Series

## Series

###### 可以理解为一维数组，除了数组元素外还包含index索引.

Series可以直接用list或np.array转化，这时index默认为0,1,2...，也可以用index=来指定索引

```python
a=[1,2,3,4]
b=np.array(a)
c=pd.Series(b)
type(a) list
type(b) numpy.ndarray
type(c) pandas.core.series.Series
```

还可以将Series看成是一个定长的有序字典，因为它是索引值到数据值的一个映射。它可以用在许多原本需要字典参数的函数中,如果数据被存放在一个Python字典中，也可以直接通过这个字典来创建Series：

```python
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj=pd.Series(sdata)
```

如果只传入一个字典，则结果Series中的索引就是原字典的键（有序排列）。你可以传入排好序的字典的键以改变顺序：

```python
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
```

在这个例子中，sdata中跟states索引相匹配的那3个值会被找出来并放到相应的位置上，但由于"California"所对应的sdata值找不到，所以其结果就为NaN（即“非数字”（not a number），在pandas中，它用于表示缺失或NA值）。因为‘Utah’不在states中，它被从结果中除去。因为Series是index优先

Series的索引可以通过赋值的方式就地修改：

```python
 obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
```



## DataFrame

DataFrame可以理解为是一个表格型的数据结构。它提供有序的列和不同类型的列值。它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔值等）。DataFrame既有行索引也有列索引，它可以被看做由Series组成的字典（共用同一个索引）。DataFrame中的数据是以一个或多个二维块存放的（而不是列表、字典或别的一维数据结构）。

```python
a=pd.DataFrame({"a":[1,2,3],"b":[1,2,3]},index=["x","y","z"])
```

如果指定了列序列，则DataFrame的列就会按照指定顺序进行排列：

```python
pd.DataFrame(data, columns=['year', 'state', 'pop'])
```

可以通过列名来选取dataframe中的一列，也就是series，并且，返回的Series拥有原DataFrame相同的索引，且其name属性也已经被相应地设置好了。

del方法可以用来删除

```python
del df['eastern']
```



# 基本功能

## 重新索引

pandas对象的一个重要方法是reindex，其作用是创建一个新对象，它的数据符合新的索引。

```python
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
 
d    4.5
b    7.2
a   -5.3
c    3.6

obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])

a   -5.3
b    7.2
c    3.6
d    4.5
e    NaN
```

列也可以用columns关键字重新索引：

```python
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)
```

## 丢弃指定轴上的项

丢弃某条轴上的一个或多个项很简单，只要有一个索引数组或列表即可。由于需要执行一些数据整理和集合逻辑，所以drop方法返回的是一个在指定轴上删除了指定值的新对象：

```python
obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])

In [106]: obj
Out[106]: 
a    0.0
b    1.0
c    2.0
d    3.0
e    4.0
dtype: float64

In [107]: new_obj = obj.drop('c')

In [108]: new_obj
Out[108]: 
a    0.0
b    1.0
d    3.0
e    4.0
dtype: float64
```

对于DataFrame，可以删除任意轴上的索引值。为了演示，先新建一个DataFrame例子

```python
In [110]: data = pd.DataFrame(np.arange(16).reshape((4, 4)),
   .....:                     index=['Ohio', 'Colorado', 'Utah', 'New York'],
   .....:                     columns=['one', 'two', 'three', 'four'])

In [111]: data
Out[111]: 
          one  two  three  four
Ohio        0    1      2     3
Colorado    4    5      6     7
Utah        8    9     10    11
New York   12   13     14    15
```

用标签序列调用drop会从行标签（axis 0）删除值：

```python
data.drop(['Colorado','Ohio'])
out:
 					one  two  three  four
Utah        8    9     10    11
New York   12   13     14    15
```

通过传递axis=1或axis='columns'可以删除列的值：

```python
data.drop('two', axis=1)
Out[113]: 
          one  three  four
Ohio        0      2     3
Colorado    4      6     7
Utah        8     10    11
New York   12     14    15
```

许多函数，如drop，会修改Series或DataFrame的大小或形状，可以就地修改对象，不会返回新的对象：

```python
obj.drop('c', inplace=True)

In [116]: obj
Out[116]: 
a    0.0
b    1.0
d    3.0
e    4.0
dtype: float64
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

![image-20220405171603008](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220405171603008.png)

## 函数应用和映射

一个常见的操作是，将函数应用到由各列或行所形成的一维数组上。DataFrame的apply方法即可实现此功能：

```python
frame
           b         d         e
Utah   -0.204708  0.478943 -0.519439
Ohio   -0.555730  1.965781  1.393406
Texas   0.092908  0.281746  0.769023
Oregon  1.246435  1.007189 -1.296221

f = lambda x: x.max() - x.min()
frame.apply(f)#默认每列进行
b    1.802165
d    1.684034
e    2.689627

  
frame.apply(f, axis='columns')
Utah      0.998382
Ohio      2.521511
Texas     0.676115
Oregon    2.542656
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

输入的缺失值被赋予NaN值，即 "Not a Number "的缩写。由于技术原因，这些NaN值总是属于float64。

```python
isna()和 isnull()区别：
isnan判断是否nan（not a number），一般是数值字段的null
isnull()主要是判断字符型是否有值， 可以判断所有的空值，但是python的数值字段比如int float 为空的时候默认是Nan
```



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

## 唯一值、值计数

```python
obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
obj.unique()
array(['c', 'a', 'd', 'b'], dtype=object)

obj.value_counts()#返回每个值出现的频数
c    3
a    3
b    2
d    1
dtype: int64
```
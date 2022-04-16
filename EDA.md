探索性数据分析（Exploratory Data Analysis，简称EDA），摘抄网上的一个中文解释，是指对已有的数据（特别是调查或观察得来的原始数据）在尽量少的先验假定下进行探索，通过作图、制表、方程拟合、计算特征量等手段探索数据的结构和规律的一种数据分析方法。

## 导包

```python
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns 
%matplotlib line
import warnings
warnings.filterwarnings('igore')
```

## 导入数据

```python
df=pd.read_cav(" ")
```

## 变量探索

```
data.info()
data.describe()
data.head()
data.tail()
...
```

```python
#缺失值
df.head()
df.describe()
missing_value=df.isnull().sum()
pct=(total/len(df))*100
mv_df=pd.DataFrame({"属性":pct.index,"缺失百分比":pct.values})
mv_df.sort_values(bt="缺失百分比",ascending=False)


#关联矩阵
import seaborn as sns
import matplotlib.pyplot as plt
#cmap颜色 annot是否显示数字 fmt数值格式
sns.heatmap(df.corr(),cmap="OrRd",annot=True,fmt="0.2f")
#countplot，统计属性类别不一致对结果的影响，例如性别与存活人数关系
#fig画布，axes位置，设定一个1*3的画布，每个大小都是12*5
fig,axes=plt.subplots(1,3,figsize=(12,5))
sns.countplot(x="Sex",hue="Survived",data=df,ax=axes[0])
sns.countplot(x="Pclass",hue="Survived",data=df,ax=axes[1])
sns.countplot(x="Embarked",hue="Survived",data=df,ax=axes[2])
plt.tight_layout()
```

![image-20220328201722470](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220328201722470.png)

但是countplot有一个缺陷，就是如果该属性类别过多，全部显示就很难看清楚，这个时候需要设置区间，使用distplot，一般对于age这种属性，使用distplot

```python
figure,axes=plt.subplots(1,2,figsize=(12,5))
sns.distplot(x=df["Age"],ax=axes[0],bins=20,kde=False)
axes[0].set_title("Age of Passagers")
plt.tight_layout()
```

![image-20220328203804557](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220328203804557.png)

一般想要分析x与y的关系，使用barplot，x放在横轴，y放在纵轴

```python
#先将age均值求出，进行groupby，之后barplot画图
mean_age=df.groupby("Survived").Age.mean().round(2).reset_index()["Age"]
figure=plt.figsize=(12,5)
sns.barplot(x=mean_age.index,y=mean_age.values)
```



![image-20220328204954512](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220328204954512.png)

但是年龄的均值可能不太能得到足够的信息，我们需要去看各个年龄阶段关于survived的情况

```python
df.groupby("Survived").Age.plot(kind="hist",alpha=0.3,ec="black",legend=True,figsize=(10,6))
```

![image-20220328210544847](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220328210544847.png)

## 画图

不同的图对应不同的目的：

**Trends**：折线图，看y随着x的发展趋势

**Relationship** ：

条形图：对比不同组别的数量

回归散点图：在散点图中包括一条回归线，使我们更容易看到两个变量之间的任何线性关系。

**Distribution** 

```python
plt.figure(figsize=(16,6))
#折线图
sns.lineplot(x=...,y=...,label="...")
plt.title("...")
plt.xlabel=...
plt.ylabel=...
#条形图
sns.barplot(x=...,y=...)
#热力图
#annot是否显示每个cell的数字
sns.heatmap(data=...,annot=True)
#散点图
sns.scatterplot(x=...,y=...)
#增加回归线的散点图
sns.regplot(x=...,y=...)
#增加颜色编码的散点图，研究三个变量的关系,一般hue是一个yes/no的二值变量
sns.scatterplot(x=..., y=..., hue=...)
#增加回归线的彩色散点图,注意此时x，y，hue不再是具体值而是属性
sns.lmplot(x="..",y="..",hue="..",data=...)
#分布图
sns.distplot(a=data,kde=False)
plt.legend()
```


# 元组

元组是一个固定长度，不可改变的Python序列对象。

用括号定义

可以使用tuple将任意序列或迭代器转换成元组：

```python
tuple([4, 0, 2])
Out[5]: (4, 0, 2)

tup = tuple('string')
Out[7]: ('s', 't', 'r', 'i', 'n', 'g')
```

可以用方括号访问元组中的元素。和C、C++、JAVA等语言一样，序列是从0开始的：

```python
tuple('string')
tup[0]
's'
```

一旦创建了元组，元组中的对象就不能修改了：

## 拆分元组

如果你想将元组赋值给类似元组的变量，Python会试图拆分等号右边的值：

```python
tup=(4,5,6)
a,b,c=tup
```

因此python交换变量更为方便，不需要中间变量tmp

```python
a,b=b,a
```

## tuple方法

因为元组的大小和内容不能修改，它的实例方法都很轻量。其中一个很有用的就是`count`（也适用于列表），它可以统计某个值得出现频率：

```python
a=(1,2,3,4,5)
a.count(2)=1
```

# 列表

与元组对比，列表的长度可变、内容可以被修改。你可以用方括号定义，或用`list`函数,等于vector

```python
a_list=[1,2,3,4]
```

可以用`append`在列表末尾添加元素,等于push_back

```
a.append(6)
```

`insert`可以在特定的位置插入元素,插入的序号必须在0和列表长度之间。

```python
b_list.insert(1, 'red')
```

insert的逆运算是pop，它移除并返回指定位置的元素：

```python
 b_list.pop(2) 返回值'peekaboo'
```

可以用`remove`去除某个值，`remove`会先寻找第一个值并除去：

```
b_list.append('foo')
```

用`in`可以检查列表是否包含某个值：

否定`in`可以再加一个not

```python
'dwarf' not in b
```

在列表中检查是否存在某个值远比字典和集合速度慢，因为Python是线性搜索列表中的值，但在字典和集合中，在同样的时间内还可以检查其它项（基于哈希表）。

## 串联和组合列表

与元组类似，可以用加号将两个列表串联起来：

```python
[4, None, 'foo'] + [7, 8, (2, 3)]
[4, None, 'foo', 7, 8, (2, 3)]
```

如果已经定义了一个列表，用`extend`方法可以追加多个元素：

```python
x.extend([7, 8, (2, 3)])
```

## 排序

你可以用`sort`函数将一个列表原地排序（不创建新的对象）：

```python
a = [7, 2, 5, 1, 3]
a.sort()
```

`sort`有一些选项，有时会很好用。其中之一是二级排序key，可以用这个key进行排序。例如，我们可以按长度对字符串进行排序：

```python
b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len)
```

## 二分搜索和维护已排序的列表

`bisect`模块支持二分查找，和向已排序的列表插入值。`bisect.bisect`可以找到插入值后仍保证排序的位置，`bisect.insort`是向这个位置插入值：

```python
import bisect

c = [1, 2, 2, 2, 3, 4, 7]

bisect.bisect(c, 2)//返回2应该在的位置的idx
Out[69]: 4

bisect.bisect(c, 5)
Out[70]: 6

bisect.insort(c, 6)

c
[1, 2, 2, 2, 3, 4, 6, 7]
```

## 切片

用切边可以选取大多数序列类型的一部分，切片的基本形式是在方括号中使用`start:stop`：左闭右开

```python
seq = [7, 2, 3, 6, 3, 5, 6, 0, 1]
seq[1:5]
[2, 3, 6, 3]
```

在第二个冒号后面使用`step`，可以隔一个取一个元素：

```python
seq[::2]
[7, 3, 3, 6, 1]
```

一个聪明的方法是使用`-1`，它可以将列表或元组颠倒过来，相当于reverse

```
seq[::-1]
[1, 0, 6, 5, 3, 6, 3, 2, 7]
```

# 序列函数

## enumerate函数

迭代一个序列时，你可能想跟踪当前项的序号。手动的方法可能是下面这样：

C++版本

```python
i = 0
for value in collection:
   # do something with value
   i += 1
```

因为这么做很常见，Python内建了一个`enumerate`函数，可以返回`(i, value)`元组序列，省了定义变量idx

```python
for i,value in enumerate(collection):
 	# do something with value
```

当你索引数据时，使用`enumerate`的一个好方法是计算序列（唯一的）`dict`(map)映射到位置的值：

```python
some_list = ['foo', 'bar', 'baz']
mapping = {}

for i, v in enumerate(some_list):
   mapping[v] = i

mapping
{'bar': 1, 'baz': 2, 'foo': 0}
```

## sorted函数

sort与sorted的不同在于，sort是在**原位**重新排列**列表**，而sorted是产生一个新的**列表**。

sort 是应用在 **list** 上的方法，sorted 可以对**所有可迭代的对象**进行排序操作。

sort 只是应用在 list 上的方法（就地排序无返回值）。

sorted 是内建函数，可对所有可迭代的对象进行排序操作，（返回新的list）。

`sorted`函数可以从任意序列的元素返回一个新的排好序的列表：

```python
sorted([7, 1, 2, 6, 0, 3, 2])
[0, 1, 2, 2, 3, 6, 7]
```

## zip函数

`zip`可以将多个列表、元组或其它序列成对组合成一个元组列表：

```python
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zipped=zip(seq1,seq2)
list(zipped)
[('foo', 'one'), ('bar', 'two'), ('baz', 'three')]
```

`zip`的常见用法之一是同时迭代多个序列，可能结合`enumerate`使用：

```python
for i,(a,b) in enumerate(zip(seq1,seq2)):
	print('{0}: {1}, {2}'.format(i, a, b))
0: foo, one
1: bar, two
2: baz, three
```

给出一个“被压缩的”序列，`zip`可以被用来解压序列。也可以当作把行的列表转换为列的列表。

```python
pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'),
('Schilling', 'Curt')]

first_names, last_names = zip(*pitchers)

first_names
('Nolan', 'Roger', 'Schilling')

last_names
('Ryan', 'Clemens', 'Curt')
```

## reversed函数

`reversed`可以从后向前迭代一个序列：

要记住`reversed`是一个生成器（后面详细介绍），只有实体化（即列表或for循环）之后才能创建翻转的序列。

```python
list(reversed(range(10)))
[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

## 字典

字典可能是Python最为重要的数据结构。它更为常见的名字是哈希映射或关联数组。它是键值对的大小可变集合，键和值都是Python对象。创建字典的方法之一是使用尖括号，用冒号分隔键和值：

```python
empty_dict = {}
d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}
d1
{'a': 'some value', 'b': [1, 2, 3, 4]}
```

可以用`del`关键字或`pop`方法（返回值的同时删除键）删除值：

```python
d1[5] = 'some value'
d1['dummy'] = 'another value'
del d1[5]
ret = d1.pop('dummy')#ret保存的是返回值，跟c++不一样，c++pop（）不返回值
```

`keys`和`values`是字典的键和值的迭代器方法。虽然键值对没有顺序，这两个方法可以用相同的顺序输出键和值：

```python
list(d1.keys())
['a', 'b', 7]
```

## 用序列创建字典

```python
mapping={}
for key,value in zip(key_list,value_list):
	mapping[key]=value
```

因为字典本质上是2元元组的集合，dict可以接受2元元组的列表：

```python
mapping = dict(zip(range(5), reversed(range(5))))
mapping
{0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
```

## 默认值

```
value = some_dict.get(key, default_value)
```

get默认会返回None，如果不存在键，pop会抛出一个例外。关于设定值，常见的情况是在字典的值是属于其它集合，如列表。例如，你可以通过首字母，将一个列表中的单词分类：

```python
words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter={}
for word in words:
	letter=word[0]
	if letter in by_letter:
		by_letter[letter].append(word)
	else:
		by_letter[letter]=[word]
```

可以简化为

```python
for word in words:
    letter = word[0]
    by_letter.setdefault(letter, []).append(word)
```

## 有效的键类型

字典的值可以是任意Python对象，而键通常是不可变的标量类型（整数、浮点型、字符串）或元组（元组中的对象必须是不可变的）。这被称为“可哈希性”。可以用`hash`函数检测一个对象是否是可哈希的（可被用作字典的键）

要用列表当做键，一种方法是将列表转化为元组，只要内部元素可以被哈希，它也就可以被哈希：

```python
d={}
d[tuple([1,2,3])]=5
{(1, 2, 3): 5}
```

## 集合

集合是无序的不可重复的元素的集合。你可以把它当做字典，但是只有键没有值。可以用两种方式创建集合：通过set函数或使用尖括号set语句：

```python
set([2, 2, 2, 1, 3, 3])
{1, 2, 3}
```

集合支持合并、交集、差分和对称差等数学集合运算。考虑两个示例集合：

```python
a = {1, 2, 3, 4, 5}

b = {3, 4, 5, 6, 7, 8}
a.union(b) 补集
a.intersection(b) 交集
```

![image-20220404202515012](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220404202515012.png)

## 列表、集合和字典推导式

列表推导式是Python最受喜爱的特性之一。它允许用户方便的从一个集合过滤元素，形成列表，在传递参数的过程中还可以修改元素。形式如下：

```python
[expr for val in collection if condition]
```

等同于

```python
result=[]
for val in collection:
	if condition:
		result.append(exper)
```

filter条件可以被忽略，只留下表达式就行。例如，给定一个字符串列表，我们可以过滤出长度在2及以下的字符串，并将其转换成大写：

```python
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
[x.upper() for x in strings if len(x) > 2]
```

## 嵌套列表推导式

假设我们有一个包含列表的列表，包含了一些英文名和西班牙名：

```python
all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'],
['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]
```

你可能是从一些文件得到的这些名字，然后想按照语言进行分类。现在假设我们想用一个列表包含所有的名字，这些名字中包含两个或更多的e。可以用for循环来做：

```python
result=[]
for names in all_data:
	enough_es=[name for name in names if name.count('e')>=2]
  result.extend(enough_es)
```

# 函数

## 返回多个值

函数可以返回多个值。

```python
def f():
    a = 5
    b = 6
    c = 7
    return a, b, c
a, b, c = f()
```

或者写成

```
return_value = f()
```

这里的return_value将会是一个含有3个返回值的三元元组



## 函数也是对象

由于Python函数都是对象，因此，在其他语言中较难表达的一些设计思想在Python中就要简单很多了。假设我们有下面这样一个字符串数组，希望对其进行一些数据清理工作并执行一堆转换：

```python
states = ['   Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda',
   .....:           'south   carolina##', 'West virginia?']
```

做法之一是使用内建的字符串方法和正则表达式`re`模块：

```python
import re
def clean_strings(strings):
		result=[]
		for value in strings:
				value=value.strip()
				value = re.sub('[!#?]', '', value)
				value = value.title()
				result.append(value)
		return result
```

其实还有另外一种不错的办法：将需要在一组给定字符串上执行的所有运算做成一个列表：

```python
def remove_punctuation(value):
    return re.sub('[!#?]', '', value)
clean_ops = [str.strip, remove_punctuation, str.title]#把函数当作对象建立列表
def clean_strings(strings, ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result
```

## 匿名（lambda）函数

Python支持一种被称为匿名的、或lambda函数。它仅由单条语句组成，该语句的结果就是返回值。它是通过lambda关键字定义的，这个关键字没有别的含义，仅仅是说“我们正在声明的是一个匿名函数”。

```
def short_function(x):
    return x * 2

equiv_anon = lambda x: x * 2
```

## 生成器

能以一种一致的方式对序列进行迭代（比如列表中的对象或文件中的行）是Python的一个重要特点。这是通过一种叫做迭代器协议（iterator protocol，它是一种使对象可迭代的通用方式）的方式实现的，一个原生的使对象可迭代的方法。比如说，对字典进行迭代可以得到其所有的键：

# 文件和操作系统

为了打开一个文件以便读写，可以使用内置的open函数以及一个相对或绝对的文件路径：

```python
path = 'examples/segismundo.txt'
f = open(path)
```

默认情况下，文件是以只读模式（'r'）打开的。然后，我们就可以像处理列表那样来处理这个文件句柄f了，比如对行进行迭代：

```
for line in f:
	pass
```

用with语句可以更容易地清理打开的文件,这样可以在退出代码块时，自动关闭文件。

```
with open(path) as f:
		lines=[x.rstrip() for x in f]
```

![image-20220404204759938](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220404204759938.png)

```python
with open('tmp.txt', 'w') as handle:
	  handle.writelines(x for x in open(path) if len(x) > 1)
```


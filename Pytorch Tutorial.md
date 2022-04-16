# Pytorch Tutorial 

```python
#import工作
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
```

在pytorch中，矩阵被叫做张量

![image-20220401155036673](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220401155036673.png)

```python
#创建tensor
import torch
torch.Tensor(2,3)/torch.Tensor([1,2,3])#直接创建

torch.ones_like(x)
torch.ones(2,3)

torch.from_numpy(x)
```



```python
# from numpy to tensor
torch.from_numpy(): 从numpy到张量。
# from tensor to numpy
.numpy(): 从tensor到numpy
```

基本操作

```python
#Basic Math with Pytorch
Resize: view()
Addition: torch.add(a,b) = a + b
Subtraction: a.sub(b) = a - b
Element wise multiplication: torch.mul(a,b) = a * b
Element wise division: torch.div(a,b) = a / b
Mean: a.mean()
Standart Deviation (std): a.std()
```

pytorch可以计算梯度

```python
from torch.autograd import Variable
#requires_grad是参不参与误差反向传播, 要不要计算梯度
#tensor不能反向传播，variable可以反向传播。
var = Variable(torch.ones(3), requires_grad = True)
```

![image-20220401160500585](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220401160500585.png)

## 实例

### Linear Regression

```python
a = [3,4,5,6,7,8,9]
a_np = np.array(a,dtype=np.float32)
#转化成一列reshape(-1,1) ,(1,-1)是一行
a_np = a_np.reshape(-1,1)
a_tensor = Variable(torch.from_numpy(a_np))

b = [ 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
b_np = np.array(b,dtype=np.float32)
b_np = b_np.reshape(-1,1)
b_tensor = Variable(torch.from_numpy(b_np))

# lets visualize our data
import matplotlib.pyplot as plt
plt.scatter(a,b)
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Car Price$ VS Number of Car Sell")
plt.show()
```

```python
# Linear Regression with Pytorch
import torch          
import torch.nn as nn 
import warnings
warnings.filterwarnings("ignore")
x_train = torch.randn(25,2)
y_train = torch.randn(25,1)
# create class
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)
    def forward(self,x):
        return self.linear(x)

#根据输入输出指定
input_dim = 2
output_dim = 1
model = LinearRegression(input_dim,output_dim) 

# MSE
mse = nn.MSELoss()

# Optimization (find parameters that minimize error)
learning_rate = 0.02   # how fast we reach best parameters
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

# train model
loss_list = []
iteration_number = 1001
for iteration in range(iteration_number):    
    outputs = model(x_train)
    loss = mse(outputs, y_train)
    #每一轮的累积梯度要清零
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    # store loss
    loss_list.append(loss.data)
    # print loss
    if(iteration % 50 == 0):
        print('epoch {}, loss {}'.format(iteration, loss.data))

plt.plot(range(iteration_number),loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.show()
```

### Logistic Regression

- Linear regression is not good at classification
- linear regression + logistic function(softmax) = logistic regression

```python
#LR
import torch as torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#超参数
input_size=28*28
num_classes = 10
epochs=7
learning_rate=0.02
batch_size =20 
# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='input', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='input', 
                                          train=False, 
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

model = nn.Linear(input_size, num_classes)
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
total_step = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, epochs, i+1, total_step, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
```



### ANN

- Steps of ANN:
  1. Import Libraries
  2. Prepare Dataset
  3. Create ANN Model
     - 3 hidden layers.
     - use ReLU, Tanh and ELU activation functions for diversity.
  4. Instantiate Model Class
     - input_dim = 28*28 
     - output_dim = 10 # labels 0,1,2,3,4,5,6,7,8,9
     - Hidden layer dimension is 150.
     - create model
  5. Instantiate Loss
     - Cross entropy loss
     - It also has softmax(logistic function) in it.
  6. Instantiate Optimizer
     - SGD Optimizer
  7. Traning the Model
  8. Prediction

```python
class ann(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(ann,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.elu3 = nn.ELU()
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.elu3(out)
        out = self.fc4(out)
        return out
input_dim = 28*28
#隐藏层作为超参数可以随意调整
hidden_dim = 150 
output_dim = 10
model = ann(input_dim, hidden_dim, output_dim)
error = nn.CrossEntropyLoss()
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model)
"""ann(
  (fc1): Linear(in_features=784, out_features=150, bias=True)
  (relu1): ReLU()
  (fc2): Linear(in_features=150, out_features=150, bias=True)
  (tanh2): Tanh()
  (fc3): Linear(in_features=150, out_features=150, bias=True)
  (elu3): ELU(alpha=1.0)
  (fc4): Linear(in_features=150, out_features=150, bias=True)
)"""
```

### CNN

```python
class CNNModel(nn.Module):
		def __init__(self):
      super(CNNModel,self).__init__()
      #Convolution 1
      #in_channels--输入图像通道数,out_channels--卷积产生的通道数
      #kernel_size--卷积核尺寸,stride--步长
      #padding--填充，dilation--扩充操作
      self.cnn1=nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
      self.relu1 = nn.ReLU()
      # Max pool 1
      self.maxpool1 = nn.MaxPool2d(kernel_size=2)
      # Convolution 2
      self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
      self.relu2 = nn.ReLU()
      # Max pool 2
      self.maxpool2 = nn.MaxPool2d(kernel_size=2)
      # Fully connected 1
      self.fc1 = nn.Linear(32 * 4 * 4, 10) 
       def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out）
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2 
        out = self.maxpool2(out)
        # flatten
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        return out
# batch_size, epoch and iteration
batch_size = 100
n_iters = 2500
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
model = CNNModel()

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

### RNN

RNN本质上是重复的ANN，但信息从以前的非线性激活函数输出中获得传递。



## 关于Module

 我们在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法。但有一些注意技巧：
（1）一般把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中，

（2）一般把不具有可学习参数的层(如ReLU、dropout、BatchNormanation层)可放在构造函数中，

（3）forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。

```python
class Module(object):
  #重构前两个函数
    def __init__(self):
    def forward(self, *input):
 
    def add_module(self, name, module):
    def cuda(self, device=None):
    def cpu(self):
    def __call__(self, *input, **kwargs):
    def parameters(self, recurse=True):
    def named_parameters(self, prefix='', recurse=True):
    def children(self):
    def named_children(self):
    def modules(self):  
    def named_modules(self, memo=None, prefix=''):
    def train(self, mode=True):
    def eval(self):
    def zero_grad(self):
    def __repr__(self):
    def __dir__(self):
```

```python
#例子
import torch
import torch.nn.functional as F
 
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1)
 
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
 
model = MyNet()
print(model)
'''运行结果为：
MyNet(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)
'''
```

## 损失函数

### Classification Error（分类错误率）

```
错误个数/总个数
```

### Mean Squared Error (均方误差)

$$
MSE=1/n*\sum(\hat yi-yi)^2
$$

### Cross Entropy Loss Function（交叉熵损失函数）

1.二分类

![image-20220225154526701](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220225154526701.png)

2.多分类

![image-20220225154604752](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220225154604752.png)
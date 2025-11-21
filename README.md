# **Experiment5--LeNet_MINST** (实验5--实现LeNet卷积神经网络)
##### The study constructed a LeNet-based convolutional neural network on the MINST dataset to perform model training and digit classification tasks. 
###### 本实验基于MINST数据集构建了LeNet卷积神经网络架构，并实现了模型训练和数字分类任务。

##

## 1.Exprimental Purpose
##### 1.Understand core concepts such as convolution,pooling,and activation functions.
##### 2.Implement the classic LeNet-5 model structure.
##### 3.Master the complete workflow of model building,training,and evaluation.
##### 4.Comprehend the complete processing pipline from raw pixels to category prediction.
##### 5.Observe how convolutional layers extract image features.

###### 1.掌握卷积神经网络基本原理：理解卷积、池化、激活函数等核心概念。
###### 2.学习LeNet网络架构：实现经典的LeNet-5模型结构。
###### 3.熟悉PyTorch深度学习框架：掌握模型构建、训练、评估的全流程。
###### 4.理解图像分类任务：从原始像素到类别预测的完整处理流程。
###### 5.可视化特征学习过程：观察卷积层如何提取图像特征。

##

## 2.Experimental Content
##### Implement PyTorch Dataloader based on the MINST dataset and independently build the LeNet network architecture.Train the model monitoring loss and accuracy metrics,and visualize the feature maps generated during the training process.
##### Supplementary requirement:Utilize the trained model weights for interactive recognition of user-input handwritten digits(0-9).
###### 用MINST数据集合构建数据批量处理器，并且自行补充实现LeNet网络结构，训练模型并输出loss和accuracy,最后观察训练过程中的特征图样式。附加：用训练权重识别交互式输入的手写数字0-9。
##
#### 2.1.MNIST Dataset Loading and Processing
##### （1）Import Pytorch and related tool libraries
###### 导入 PyTorch 及相关工具库
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
##
##### （2）Define data loaders for the training and test sets,automatically download/read the MNIST dataset and convert its format via ToTensor(),configure batch size and shuffling parameters.
###### 定义训练集和测试集的数据加载器，自动下载 / 读取 MNIST 数据集并通过 ToTensor () 转换格式，同时配置批次大小与打乱等参数。
```
batch_size=512
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
trainloader = torch.utils.data.DataLoader(datasets.MNIST('data',train=True,download=True,
                                                         transform=transforms.Compose([transforms.ToTensor()])),batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(datasets.MNIST('data',train=False,download=True,
                                                         transform=transforms.Compose([transforms.ToTensor()])),batch_size=batch_size,shuffle=True)
```
##
##### （3）Define LeNet-5 model architecture, which  consists of two convolution layers,two dense layers,and one classification layer.Then, configure the learning rate and optimizer(e.g., Adam).
###### 定义LeNet-5的网络结构（2个卷积层+2个全连接层+1个分类层），并设置了优化器学习率等参数。
```
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.fc1=nn.Linear(5*5*16,120)
        self.fc2=nn.Linear(120,84)
        self.clf=nn.Linear(84,10)
        self.relu=nn.Sigmoid()
        self.pool=nn.AvgPool2d(kernel_size=2,stride=2)
        self.flatten=nn.Flatten(start_dim=1)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.flatten(x)
        x=self.fc1(x)
        x=self.relu(x)

        x=self.fc2(x)
        x=self.relu(x)

        x=self.clf(x)
        return x
    
model=Net().to(device)
optimizer=optim.Adam(model.parameters(),lr=1e-2)
```

##
#### 2.2.Model Training and Evaluation
##### Configure the number of training epochs and demonstrate the procedure, including the forward pass, loss computation, and backpropagation.
###### 设置训练轮数并且展示数据训练过程的前向传播，损失计算，反向传播等流程。
###### Training Process
```
epochs=30
accs,losses=[],[]
for epoch in range(epochs):
    for batch_idx,(x,y) in enumerate(trainloader):
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out,y)
        loss.backward()
        optimizer.step()
```
###### Testing Process
```
with torch.no_grad():
        for batch_idx,(x,y) in enumerate(testloader):
            x,y=x.to(device),y.to(device)
            out=model(x)
            testloss +=F.cross_entropy(out,y).item()
            pred=out.max(dim=1,keepdim=True)[1]
            correct +=pred.eq(y.view_as(pred)).sum().item()
```

##
#### 2.3.Feature Map Visualization
##### Extract and visualize feature maps from convolutional layers.
###### 提取并绘制卷积层的输出特征图。
```
feature1=F.sigmoid(model.conv1(x))
feature2=F.sigmoid(model.conv2(feature1))
n=5
img = x.detach().cpu().numpy()[:n]
feature_map1=feature1.detach().cpu().numpy()[:n]
feature_map2=feature2.detach().cpu().numpy()[:n]

fig,ax=plt.subplots(3,n,figsize=(10,10))
for i in range(n):
    ax[0,i].imshow(img[i].sum(0),cmap='gray')
    ax[1,i].imshow(feature_map1[i].sum(0),cmap='gray')
    ax[2,i].imshow(feature_map2[i].sum(0),cmap='gray')
plt.show()
```
###### 代码中的一些参数解释：
###### feature_map2=feature2.detach().cpu().numpy()[:n]
###### x.detach() - 从计算图中分离，不跟踪梯度
###### .cpu() - 从GPU移到CPU（如果用了GPU）
###### .numpy() - 转成numpy数组，便于matplotlib显示
###### [:n] - 只取前n个样本
###### 结果：原始输入图像的numpy数组






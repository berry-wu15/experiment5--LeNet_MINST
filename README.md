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

##
#### 2.4.Supplemently requriement:interactive recognition of user-input handwritten digits(0-9)
##### (1)Import related tool libraries.
```
import numpy as np
import cv2
from PIL import Image
```
##
##### (2)Process the input data.
```
def preprocess_image(image):
    """预处理手写数字图像"""
    # 转换为灰度图 - 如果输入是彩色图像(3通道)，转换为单通道灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 调整大小为28x28 - MNIST数据集的标准输入尺寸
    image = cv2.resize(image, (28, 28))
    
    # 反转颜色（MNIST是白底黑字，我们绘制的是黑底白字）
    # 因为我们是在白色画布上画黑色数字，而MNIST是黑色背景白色数字
    image = 255 - image
    
    # 归一化并转换为tensor - 将像素值从0-255缩放到0-1之间
    image = image.astype(np.float32) / 255.0
    # 添加batch和channel维度: (28,28) -> (1,1,28,28)
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  
    return image.to(device)  # 移动到GPU或CPU设备
```
##
##### (3)Initialize a canvas.
```
def draw_digit():
    """绘制数字并识别"""
    # 创建画布 - 280x280的白色画布（255表示白色）
    canvas = np.ones((280, 280), dtype=np.uint8) * 255
    drawing = False  # 标记是否正在绘制
    
    def draw_circle(event, x, y, flags, param):
        nonlocal drawing, canvas
        if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下
            drawing = True
            ...............
```
##
##### (4)Predict digit from input canvas.
```
if key == ord('s'):  # 按's'键进行识别
            # 预处理图像 - 将绘制的图像转换为模型可接受的格式
            processed_img = preprocess_image(canvas)
            
            # 使用训练好的模型进行预测
            with torch.no_grad():  # 不计算梯度，节省内存
                output = model(processed_img)  # 模型前向传播
                prediction = torch.argmax(output, dim=1).item()  # 获取预测结果（最大概率的类别）
                probabilities = F.softmax(output, dim=1)[0]  # 计算每个类别的概率
    
            # 显示结果 - 在原画布上添加识别结果文字
            result_canvas = canvas.copy()
            # 在画布上显示预测结果
            cv2.putText(result_canvas, f'Prediction: {prediction}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)  # 字体、大小、颜色、粗细
            # 在画布上显示置信度
            cv2.putText(result_canvas, f'Confidence: {probabilities[prediction]:.2f}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
            cv2.imshow('Result', result_canvas)  # 显示结果窗口
```

##
## 3.Experimental Results and Analysis
#### 3.1.Model Structure Output
```
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=256, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (clf): Linear(in_features=84, out_features=10, bias=True)
  (relu): Sigmoid()
  (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (flatten): Flatten(start_dim=1, end_dim=-1)
)
```
##
#### 3.2.Training Log(epoch,loss,accuracy)
```
epoch:0,loss:2.301466,acc:0.113500
epoch:1,loss:2.301534,acc:0.113500
epoch:2,loss:2.301126,acc:0.113500
epoch:3,loss:2.301366,acc:0.113500
epoch:4,loss:2.300994,acc:0.113500
epoch:5,loss:2.095140,acc:0.225700
epoch:6,loss:0.536351,acc:0.821200
epoch:7,loss:0.176625,acc:0.946100
epoch:8,loss:0.112669,acc:0.963600
.........
epoch:25,loss:0.033249,acc:0.988900
epoch:26,loss:0.039878,acc:0.988200
epoch:27,loss:0.039106,acc:0.988600
epoch:28,loss:0.039390,acc:0.987900
epoch:29,loss:0.038174,acc:0.989000
```
##
#### 3.3.Display Feature Maps
<table>
 <tr>
 <td><img src="./Experimental Image/feature maps.png"  height="350px" alt="feature maps"></td>
 <tr>
</table>

##
#### 3.4.Interactive Digit Recognition Demo
<table>
 <tr>
 <td><img src="./Experimental Image/digit 0.png"  width="200px" height="300px" alt="digit 0"></td>
 <td><img src="./Experimental Image/interative recognition 0.png" width="200px" height="300px" alt="interactive recognition 0"></td>
 </tr>
</table>
<table>
 <tr>
 <td><img src="./Experimental Image/digit 1.png"  width="200px" height="300px" alt="digit 1"></td>
 <td><img src="./Experimental Image/interative recognition 1.png" width="200px" height="300px" alt="interactive recognition 1"></td>
 </tr>
</table>
<table>
 <tr>
 <td><img src="./Experimental Image/digit 2.png"  width="200px" height="300px" alt="digit 2"></td>
 <td><img src="./Experimental Image/interative recognition 2.png" width="200px" height="300px" alt="interactive recognition 2"></td>
 </tr>
</table>
<table>
 <tr>
 <td><img src="./Experimental Image/digit 5.png"  width="200px" height="300px" alt="digit 5"></td>
 <td><img src="./Experimental Image/interative recognition 5.png" width="200px" height="300px" alt="interactive recognition 5"></td>
 </tr>
</table>
<table>
 <tr>
 <td><img src="./Experimental Image/digit 6.png"  width="200px" height="300px" alt="digit 6"></td>
 <td><img src="./Experimental Image/interative recognition 6.png" width="200px" height="300px" alt="interactive recognition 6"></td>
 </tr>
</table>

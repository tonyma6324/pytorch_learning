import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#torch.nn命名空间提供了构建自己的神经网络所需的所有构建块

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() #调用父类的init
        self.flatten = nn.Flatten() 
        #为全连接层或者说是线性层做准备 ---就是进行线性变换，需要一维的输入，这里将二维转一维
        self.linear_relu_stack = nn.Sequential(
            #nn.Sequential是模块的有序容器。数据按照定义的顺序传递到所有模块。您可以使用顺序容器来快速组合网络
            nn.Linear(28*28, 512),
            #线性变换，把28*28映射到512个输出特征
            nn.ReLU(),
            #非线性使得神经网络能够学习复杂的函数
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

#ReLU:(Rectified Linear Unit)----f(x)=max(0,x)
#Sigmoid:f(x)= 1/(1+e**(-z))
#Tanh:f(x)=tanh(x)
#定义向前传递的过程
    def forward(self, x): 
        x = self.flatten(x)
        print(x.size())
        logits = self.linear_relu_stack(x)
        return logits
    


model = NeuralNetwork().to(device)
print(model)

X = torch.rand(3, 28, 28, device=device)
#3在这里是批次，表示一次性处理的图像数量
#print(X.size())
logits = model(X)
#模型的输出是一个 logits 向量，表示每个类别的得分
#在训练过程中，通常会将这些 logits 传递给损失函数（如交叉熵损失）来计算损失，并进行反向传播以更新模型参数。
pred_probab = nn.Softmax(dim=1)(logits)
'''
最后一层线性层返回logits - [-infty, infty] 中的原始值
nn.Softmax(dim=1) 是 PyTorch 中的一个层，用于计算输入张量的 Softmax。
Softmax 函数将 logits 转换为概率分布，使得每个类别的概率值在 0 到 1 之间，并且所有类别的概率之和为 1。
dim=1 指定了在第 1 维（即类别维度）上应用 Softmax
'''
y_pred = pred_probab.argmax(1)
'''

pred_probab.argmax(1) 用于获取每个样本的预测类别。argmax 函数返回沿指定维度的最大值的索引。这里就是1维
'''
print(f"Predicted class: {y_pred}")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

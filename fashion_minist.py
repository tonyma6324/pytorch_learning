import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#训练数据
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
#测试数据
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    #转化成pytorch张量，ToTensor 将 PIL 图像或 NumPy 转换ndarray为FloatTensor. 并将图像的像素强度值缩放到范围 [0., 1.] 内
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

#创建一个8x8图形
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    #torch.randint(low,high,size)  item()拿来转换pytorch能接受的张量
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    #添加子图，位置由i指定
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")  
'''
img.squeeze()：
squeeze() 是一个用于去掉张量中维度为 1 的维度的方法。在处理图像时，尤其是灰度图像，图像的形状通常是 (1, H, W)，其中 1 是通道数（表示这是一个单通道的灰度图像），H 是高度，W 是宽度。
使用 squeeze() 后，图像的形状将变为 (H, W)，这使得它可以被 plt.imshow() 正确处理
'''
plt.show()

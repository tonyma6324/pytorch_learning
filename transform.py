import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
#lambda是我们自定义的
#torch.zeros(10, dtype=torch.float)长度为10的零张量
#scatter_ 是一个原地操作方法，用于在指定的维度上根据索引填充张量
#scatter_(0, torch.tensor(y), value=1) ，torch.tensor(y)要填充的索引地方
#最后转化成独热编码
'''
如果 y 是 0输出将是 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
如果 y 是 1输出将是 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
如果 y 是 2输出将是 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
'''
#就是转换标签

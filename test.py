import torch 
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data)
#print(x_data)
np_array = np.array(data)
#print(np_array)
x_np = torch.from_numpy(np_array)
#print(x_np)

#tensor 维度
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
'''
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#tensor的属性,类型
print(f"Shape of tensor: {rand_tensor.shape}")
print(f"Datatype of tensor: {rand_tensor.dtype}")
print(f"Device tensor is stored on: {rand_tensor.device}")
'''

#取tensor不同行列
print(f"first row of tensor: {rand_tensor[0]}")
print(f"first colum of tensor: {rand_tensor[:,2]}")
print(f"first colum of tensor: {rand_tensor[...,-1]}")

t1 = torch.cat([rand_tensor,rand_tensor],dim=1)
print(t1)

#矩阵相乘
y1 = rand_tensor @  rand_tensor.T
#Hadamard乘积
y2 = rand_tensor * rand_tensor

#张量求和
agg = rand_tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


#就地运算，直接改变了原值
print(rand_tensor, "\n")
rand_tensor.add_(5)
print(rand_tensor)

#numpy和张量的使用，张量转numpy数组

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

#numpy数组转张量

n = np.ones(5)
t = torch.from_numpy(n)



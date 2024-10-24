import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output

w = torch.randn(5, 3, requires_grad=True) #requires_grad=True 表示在后续的计算中需要计算这些张量的梯度，以便进行反向传播。
#w---weight 即权重张量 这里是一个[5,3]形状的张量
b = torch.randn(3, requires_grad=True)
#b-- bias 即偏置 这里是[3,]
#可以看出神经元所需的雏形

z = torch.matmul(x, w)+b
#这行代码计算了线性变换 z=x⋅w+b 
#用的是矩阵乘法
#torch.matmul(x, w) 进行矩阵乘法，将输入张量 x（形状为 (5,)）与权重张量 w（形状为 (5, 3)）相乘，得到一个形状为 (3,) 的张量。
#然后将偏置 b 加到结果上，得到最终的输出 z。

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
#这行代码计算了二元交叉熵损失（binary cross-entropy loss），并且使用了 logits（未经过 sigmoid 函数的输出）
#print(loss)
#目前来看这个函数输出就是个标量
loss.backward(retain_graph=True)
#其实就是反向传播的过程 ，这里如果前面设置了梯度requires_grad=True，就会去计算loss相对于张量的梯度
#在我们向后传递的时候，backward在DAG的根上调用，前面的所有节点根据链式法则（就是复合函数求导）来依次计算梯度
print(w.grad)
print(b.grad)

#这里我们获得了w和b相对loss的梯度，一般是loss/w 的导数
#出于性能原因，我们只能 backward在给定的图上使用一次来执行梯度计算。如果我们需要backward在同一张图上进行多次调用，则需要传递 retain_graph=True给backward调用。
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    #停止梯度的跟踪
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
#detach() 方法用于创建一个新的张量，该张量与原始张量共享相同的数据，但不再跟踪梯度
#这样可以达成同样不再跟踪梯度的效果，一般我们只想向前传递以及冻结一些参数的时候，会想用这种方法

inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

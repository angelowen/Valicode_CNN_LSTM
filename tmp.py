import torch
import torch.nn.functional as F
from torch import nn
# input = torch.randn(4,2,3)
# # print(input)
# m = F.softmax(input,dim=2)
# print(m)
# _, preds = torch.max(m.data, 2)
# print(preds)
outputs = torch.rand(16 ,4,10, requires_grad=True)
target = torch.randint(4,(16,10))
# outputs = F.log_softmax(outputs, dim=2)
# print(outputs)
# print(target)
loss_function = nn.CrossEntropyLoss()
loss = loss_function(outputs, target)
print(loss.item())
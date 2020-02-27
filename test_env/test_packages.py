import numpy as np
import torch
import torch.nn as nn


u = np.array([1, 2, 3])
v = np.array([2, 2, 2])

A = torch.Tensor(np.array([[1, 2, 3]]))
B = torch.Tensor(np.array([[2, 2, 2]])).t()

print('numpy: ', np.dot(u, v))

print('torch: ', A.mm(B))
print('torch: ', B.mm(A))
print('torch: ', A.mv(torch.Tensor(v)))

import torch
import numpy as np

def f(x):
    return torch.exp(x[0]*x[1])


x = torch.tensor([1.,2.])
# create graph allows optimizer to step over -delta(u)
hessian = torch.autograd.functional.hessian(f, x, create_graph=True)

print ("Done")


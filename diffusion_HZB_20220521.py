import torch
import xlrd
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''

def import_excel_matrix(path):
    table = xlrd.open_workbook(path).sheets()[0]
    row = table.nrows 
    col = table.ncols 
    data_matrix = np.zeros((row,col))
    for i in range(col):
        cols = np.matrix(table.col_values(i)) 
        data_matrix[:,i] = cols
    return data_matrix

data_file = ''
data = import_excel_matrix(data_file)


X = data[:,[0]]
C = data[:,[1]]
X = torch.tensor(X,dtype=torch.float32,requires_grad=False).cuda()
C = torch.tensor(C,dtype=torch.float32).cuda()


def diffusion_func(X,X_0,t,a,b):
    C_t = 0.5*(torch.erfc((X-X_0)/(2*torch.sqrt(t))))*(a-b)+ b
    return C_t

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.X_0 = Parameter(torch.tensor([0.0]))
        self.t = Parameter(torch.tensor([0.0]))
        self.a = Parameter(torch.tensor([0.0]))
        self.b = Parameter(torch.tensor([0.0]))
    def forward(self, X):
        C_t = 0.5*(2-(torch.erfc((X-self.X_0)/(2*torch.sqrt(self.t)))))*(self.a-self.b)+self.b
        return C_t
model = mymodel()
model.cuda()
lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
for it in range(0):
    C_t = model(X)
    loss = torch.sum((C_t-C)*(C_t-C))

    if it%1000==0:
        print(it, loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    
print(loss.item(),
      "X_0:",model.X_0,
      "t:",model.t,
      "a:",model.a,
      "b:",model.b
     )

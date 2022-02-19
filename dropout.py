import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.optim.optimizer import Optimizer, required
## two layer neural network

N = 1000
d = 5
X = torch.zeros([N, d]).normal_(0,1).cuda()
Xtest = torch.zeros([N, d]).normal_(0,1).cuda()

eps = torch.zeros([N, d]).normal_(0,0.1).cuda()
Y = X ** 3 + eps

class Net(nn.Module):

    def sinx(self, x):
        return x + (torch.sin(1 * x) ** 2) / 1

    def __init__(self, width=10):
        super(Net, self).__init__()
        self.w2 = nn.Linear(d, width)
        self.w3 = nn.Linear(width, 5)
        self.act = torch.relu

    def forward(self, x):

        x = self.act(self.w2(x))
        x = F.dropout(x, p=0.1)
        x = (self.w3(x))
        return x

model = Net()

#model = Net(width=1000).cuda()
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


exp = np.arange(1, 14)
width_range = 2 ** exp

vars = []
test_var = []
for width in width_range:

  model = Net(width=width).cuda()
  lr = 0.04
  #opt = optim.SGD(model.parameters(), lr=lr , momentum=0.9)
  opt = optim.Adam(model.parameters(), lr=lr)

  STEP = 4500
  criterion = F.mse_loss

  for i in range(STEP):
    if i % 1500 ==0:
      lr = lr / 10
      opt = optim.Adam(model.parameters(), lr=lr)
    opt.zero_grad()
    output = model(X)
    #print(output.shape, Y.shape)
    loss = criterion(output, Y)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    opt.step()


  mean = 0
  second = 0
  with torch.no_grad():
    for j in range(1000):
      out = model(X)
      mean = out + mean
      second = out ** 2 + second
    mean = mean / 1000
    second = second / 1000
  print(loss.item(), (second - (mean ** 2)).mean())
  var = (second - (mean ** 2)).mean().item()
  vars.append(var)

  mean = 0
  second = 0
  with torch.no_grad():
    for j in range(1000):
      out = model(Xtest)
      mean = out + mean
      second = out ** 2 + second
    mean = mean / 1000
    second = second / 1000
  print(loss.item(), (second - (mean ** 2)).mean())
  var = (second - (mean ** 2)).mean().item()
  test_var.append(var)

plt.figure(figsize=(3.5,2.5))
plt.plot(width_range, vars, linewidth=0, marker='s', label='training set')
plt.plot(width_range, test_var, linewidth=0, marker='^', label='testing set')
plt.ylabel('variance')
plt.xlabel('width')
plt.loglog()
plt.legend()
plt.savefig('dropout.png', dpi=200, bbox_inches='tight')

## linear net

class Net(nn.Module):

    def sinx(self, x):
        return x + (torch.sin(1 * x) ** 2) / 1

    def __init__(self, width=10):
        super(Net, self).__init__()
        self.w2 = nn.Linear(d, width)
        self.w3 = nn.Linear(width, 5)
        #self.act = torch.relu

    def forward(self, x):

        x = (self.w2(x))
        x = F.dropout(x, p=0.1)
        x = (self.w3(x))
        return x

model = Net()

exp = np.arange(1, 14)
width_range = 2 ** exp

vars = []
for width in width_range:

  model = Net(width=width).cuda()
  lr = 0.04 #/ (width ** 0.2)
  opt = optim.SGD(model.parameters(), lr=lr) #, momentum=0.9)
  #opt = optim.Adam(model.parameters(), lr=lr)

  STEP = 4500
  criterion = F.mse_loss

  for i in range(STEP):
    opt.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    
    loss.backward()
    opt.step()


  mean = 0
  second = 0
  with torch.no_grad():
    for j in range(1000):
      out = model(X)
      mean = out + mean
      second = out ** 2 + second
    mean = mean / 1000
    second = second / 1000
  print(loss.item(), (second - (mean ** 2)).mean())
  var = (second - (mean ** 2)).mean().item()
  vars.append(var)

plt.plot(width_range, vars, linewidth=0, marker='s')

plt.loglog()

## student-t distribution

width = 100

M = torch.distributions.studentT.StudentT(torch.zeros([width]) + 2)

M.sample()

class Net(nn.Module):

    def sinx(self, x):
        return x + (torch.sin(1 * x) ** 2) / 1

    def __init__(self, width=10):
        super(Net, self).__init__()
        self.w2 = nn.Linear(d, width)
        self.w3 = nn.Linear(width, 5)
        self.act = torch.relu

    def forward(self, x):

        x = self.act(self.w2(x))
        x = x + M.sample().cuda()
        #x = F.dropout(x, p=0.1)
        x = (self.w3(x))
        return x

model = Net()

from numpy.core.fromnumeric import repeat
exp = np.arange(1, 14)
width_range = 2 ** exp
REPEAT = 5

vars = []
test_var = []
for width in width_range:
  M = torch.distributions.studentT.StudentT(torch.zeros([width]) + 5, scale=0.1)
  model = Net(width=width).cuda()
  lr = 0.01 #/ (width ** 0.2)
  #opt = optim.SGD(model.parameters(), lr=lr , momentum=0.9)
  opt = optim.Adam(model.parameters(), lr=lr)

  STEP = 8000
  criterion = F.mse_loss

  for i in range(STEP):
    if i % 2000 ==0:
      lr = lr / 10
      opt = optim.Adam(model.parameters(), lr=lr)
    opt.zero_grad()

    loss = 0 
    for j in range(REPEAT):
      output = model(X)
      #print(output.shape, Y.shape)
      loss = criterion(output, Y) + loss
    
    loss = loss / REPEAT
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
    opt.step()


  mean = 0
  second = 0
  with torch.no_grad():
    for j in range(1000):
      out = model(X)
      mean = out + mean
      second = out ** 2 + second
    mean = mean / 1000
    second = second / 1000
  print(loss.item(), (second - (mean ** 2)).mean())
  var = (second - (mean ** 2)).mean().item()
  vars.append(var)

  mean = 0
  second = 0
  with torch.no_grad():
    for j in range(1000):
      out = model(Xtest)
      mean = out + mean
      second = out ** 2 + second
    mean = mean / 1000
    second = second / 1000
  print(loss.item(), (second - (mean ** 2)).mean())
  var = (second - (mean ** 2)).mean().item()
  test_var.append(var)

plt.figure(figsize=(3.5,2.5))
plt.plot(width_range, vars, linewidth=0, marker='s', label='training set')
plt.plot(width_range, test_var, linewidth=0, marker='^', label='testing set')
plt.ylabel('variance')
plt.xlabel('width')
plt.loglog()
plt.legend()
plt.savefig('student-t.png', dpi=200, bbox_inches='tight')
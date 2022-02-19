import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.optim.optimizer import Optimizer, required

N = 100
d = 20
Xtrain = torch.zeros([N, d]).normal_(0,1).cuda()
Xtest = torch.zeros([N, d]).normal_(0,1).cuda()

eps = torch.zeros([N, d]).normal_(0,0.1).cuda()
#Y = X ** 3 + eps

#latent_size = 128
class VAE_Simple(nn.Module):

    def __init__(self, latent_size=10):
        super(VAE_Simple, self).__init__()
        self.w = torch.nn.Linear(20, 32)
        self.w1mu = torch.nn.Linear(32, latent_size, bias=True)
        self.w1sig = torch.nn.Linear(32, latent_size)
        self.act = F.relu
        self.dw1 = torch.nn.Linear(latent_size, 32, bias=True)
        self.dw3 = torch.nn.Linear(32, 20)

    def musig(self, x):
        x = self.act(self.w(x))
        mu = (self.w1mu(x))
        sig_sq = torch.exp(self.w1sig(x))
        return mu, sig_sq

    def z(self, mu, sig_sq):
      z = (sig_sq ** 0.5) * torch.zeros_like(mu).normal_(0, 1) + mu
      return z

    def forward(self,x):
      mu, sig_sq = self.musig(x)
      z = self.z(mu, sig_sq)

      x = self.act(self.dw1(z))
      x = (self.dw3(x))
      return x, mu, sig_sq, z
    
    def generate(self, z):
      x = self.act(self.dw1(z))
      x = (self.dw3(x))
      return x

model = VAE_Simple()

def KL(z, sig, mu):
  return 0.5 * (1 +  torch.log(sig) - mu ** 2  - (sig)).sum()

losses = []
epochs = 4500

f = VAE_Simple(latent_size=10000).cuda()

lr = 0.04
for epoch in range (epochs):
  if epoch % 1500 == 0:
    lr =  lr/10
    optim = torch.optim.Adam(f.parameters(), lr=lr)

  optim.zero_grad()
  loss_gen = 0


  for j in range(1):
    x_gen,mu,sig_sq,z = f(Xtrain)
    loss_gen = ((x_gen -Xtrain) ** 2).mean() + loss_gen
  

  loss_kl = KL(z[-1], sig_sq, mu)
  loss_gen = loss_gen 
  loss = loss_gen - loss_kl
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
  optim.step()
  
  losses.append(loss.detach())
  

  if epoch % 100 == 0:
      mean = 0 
      second = 0
      with torch.no_grad():
        for j in range(100):
          x_gen,mu,sig_sq,z = f(Xtrain)
          
          mean = x_gen + mean
          second = x_gen ** 2 + second
        mean = mean / 100
        second = second / 100
        var = (second - (mean**2)).mean()
      print(epoch+1, ", total loss=", loss_gen.item(), loss.item(), var.item())

print(loss_gen.item(), z[-1].mean().item(), z[-1].var().item())
print(loss.item())

losses = []
vars = []
test_var = []
epochs = 6000

exp = np.arange(1, 14)
width_range = 2 ** exp

for width in width_range:
  f = VAE_Simple(latent_size=width).cuda()

  lr = 0.1
  for epoch in range (epochs):
    if epoch % 2000 == 0:
      lr =  lr/10
      optim = torch.optim.RMSprop(f.parameters(), lr=lr)
    optim.zero_grad()
    loss_gen = 0


    for j in range(5):
      x_gen,mu,sig_sq,z = f(Xtrain)

      loss_gen = ((x_gen -Xtrain) ** 2).mean() + loss_gen
    

    loss_kl = KL(z[-1], sig_sq, mu)
    loss_gen = loss_gen 
    loss = loss_gen-  0.01 * loss_kl / width
    loss.backward()
    optim.step()
    
    losses.append(loss.detach())
    
  mean = 0 
  second = 0
  with torch.no_grad():
    for j in range(100):
      x_gen,mu,sig_sq,z = f(Xtrain)
      
      mean = x_gen + mean
      second = x_gen ** 2 + second
    mean = mean / 100
    second = second / 100
    var = (second - (mean**2)).mean()
    vars.append(var.item())

  mean = 0 
  second = 0
  with torch.no_grad():
    for j in range(100):
      x_gen,mu,sig_sq,z = f(Xtest)
      
      mean = x_gen + mean
      second = x_gen ** 2 + second
    mean = mean / 100
    second = second / 100
    var = (second - (mean**2)).mean()
    test_var.append(var.item())
  print(epoch+1, ", total loss=", loss_gen.item(), loss.item(), var.item())

plt.plot(width_range, vars, marker='s', linewidth=0)
plt.plot(width_range, test_var, marker='^', linewidth=0)
plt.loglog()
plt.ylim(2e-5,2e-1)
import torch
import torch.autograd as autograd         
from torch import Tensor                
import torch.nn as nn                     
import torch.optim as optim               



import matplotlib.pyplot as plt
from math import e
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split


import numpy as np
import time
from pyDOE import lhs       
import scipy.io



def plot1(x,t,y):
  tt =t.squeeze(1)
  xx =x.squeeze(1) 
  X,T= torch.meshgrid(xx,tt)
  ftx = y
  figure,axi=plt.subplots(1,1)
  c = axi.contourf(T,X, ftx,20,cmap="rainbow")
  figure.colorbar(c) 
  axi.set_title('f(t,x)')
  axi.set_xlabel('t')
  axi.set_ylabel('x')
  plt.show()

  axi = plt.axes(projection='3d')
  axi.plot_surface(T.numpy(), X.numpy(), ftx.numpy(),cmap="rainbow")
  axi.set_xlabel('t')
  axi.set_ylabel('x')
  axi.set_zlabel('f(t,x)')
  plt.show()



def plot3(U_predicao,X_U_Treino,U_Treino, T, X, x, t, solucao):


 
    figure, axi = plt.subplots()
    axi.axis('off')    
    g0 = gridspec.GridSpec(1, 2)
    g0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    axi = plt.subplot(g0[:, :])
    m = axi.imshow(U_predicao, interpolation='nearest', cmap='rainbow', 
                extent=[T.min(), T.max(), X.min(), X.max()], 
                origin='lower', aspect='auto')
    d = make_axes_locatable(axi)
    c = d.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(m, cax=c)
    axi.plot(X_U_Treino[:,1], X_U_Treino[:,0], 'kx', label = 'Dados', markersize = 4, clip_on = False)
    l = np.linspace(x.min(), x.max(), 2)[:,None]
    axi.plot(t[25]*np.ones((2,1)), l, 'w-', linewidth = 1)
    axi.plot(t[50]*np.ones((2,1)), l, 'w-', linewidth = 1)
    axi.plot(t[75]*np.ones((2,1)), l, 'w-', linewidth = 1)    
    axi.set_xlabel('$t$')
    axi.set_ylabel('$x$')
    axi.legend(frameon=False, loc = 'best')
    axi.set_title('$u(t,x)$', fontsize = 10)


    
    g1 = gridspec.GridSpec(1, 3)
    g1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    axi = plt.subplot(g1[0, 0])
    axi.plot(x,solucao.T[25,:], 'b-', linewidth = 2, label = 'Sol_Exata')       
    axi.plot(x,U_predicao.T[25,:], 'r--', linewidth = 2, label = 'Sol_Predita')
    axi.set_xlabel('$x$')
    axi.set_ylabel('$u(t,x)$')    
    axi.set_title('$t = 0.25s$', fontsize = 10)
    axi.axis('square')
    axi.set_xlim([-1.1,1.1])
    axi.set_ylim([-1.1,1.1])
    axi = plt.subplot(g1[0, 1])
    axi.plot(x,solucao.T[50,:], 'b-', linewidth = 2, label = 'Sol_Exata')       
    axi.plot(x,U_predicao.T[50,:], 'r--', linewidth = 2, label = 'Sol_Predita')
    axi.set_xlabel('$x$')
    axi.set_ylabel('$u(t,x)$')
    axi.axis('square')
    axi.set_xlim([-1.1,1.1])
    axi.set_ylim([-1.1,1.1])
    axi.set_title('$t = 0.50s$', fontsize = 10)
    axi.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    axi = plt.subplot(g1[0, 2])
    axi.plot(x,solucao.T[75,:], 'b-', linewidth = 2, label = 'Exact')       
    axi.plot(x,U_predicao.T[75,:], 'r--', linewidth = 2, label = 'Prediction')
    axi.set_xlabel('$x$')
    axi.set_ylabel('$u(t,x)$')
    axi.axis('square')
    axi.set_xlim([-1.1,1.1])
    axi.set_ylim([-1.1,1.1])    
    axi.set_title('$t = 0.75s$', fontsize = 10)
    
    plt.savefig('Resultado_Eq_Burgers.png',dpi = 500)   

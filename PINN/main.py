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

import re

from Plt import plot1, plot3
from Dados import Dados
from Treinamento import Treinamento
from ProcessDados import ProcDados

torch.set_default_dtype(torch.float); torch.manual_seed(1234); np.random.seed(1234); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


max_it=15000
cd = np.array([2,20,20,20,20,20,20,20,20,1]) 
NNu = 200 ; NNf = 15000 
L_r=1e-3 ; mu = 0.01/np.pi


# Parte 1
data, x, t, solucao, X, T = Dados()


# Parte 2    
class NN(nn.Module): 
    
    def __init__(self,cd):
        super().__init__()     
        self.ls = nn.ModuleList([nn.Linear(cd[i], cd[i+1]) for i in range(len(cd)-1)])
        self.F_Perda = nn.MSELoss(reduction ='mean')  
        self.atv = nn.Tanh()
        self.it = 0
        for j in range(len(cd)-1):
           nn.init.xavier_normal_(self.ls[j].weight.data, gain=1.0)
           nn.init.zeros_(self.ls[j].bias.data)
    def parte1(self,t):
        if torch.is_tensor(t)!= True:         
            t = torch.from_numpy(t)                
        U = torch.from_numpy(Ub).float().to(device)
        L = torch.from_numpy(Lb).float().to(device)
        t = (t - U)/(U - L) 
        h = t.float()
        for i in range(len(cd)-2):
            k = self.ls[i](h)
            h = self.atv(k)
        h = self.ls[-1](h)
        return h                
    def parte2(self,x,y):
        L_u = self.F_Perda(self.parte1(x), y)
        return L_u
    def parte3(self, X_Treino_NNf):                
        p = X_Treino_NNf.clone()      
        p.requires_grad = True
        U = self.parte1(p)
        Utx = autograd.grad(U,p,torch.ones([X_Treino_NNf.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]     
        Uttxx = autograd.grad(Utx,p,torch.ones(X_Treino_NNf.shape).to(device), create_graph=True)[0]
        Ux = Utx[:,[0]]
        Ut = Utx[:,[1]]
        Uxx = Uttxx[:,[0]]              
        F = Ut + (self.parte1(p))*(Ux) - (mu)*Uxx 
        L_f = self.F_Perda(F,Tx_f)               
        return L_f
    def parte4(self,x,y,X_Treino_NNf):
        L_u = self.parte2(x,y)
        L_f = self.parte3(X_Treino_NNf)
        L_total = L_u + L_f
        return L_total                                    
    def parte5(self):
        otim.zero_grad()
        L = self.parte4(X_Treino_NNu, U_Treino_NNu, X_Treino_NNf)
        self.it+=1
        L.backward()
        if self.it%100== 0:
            erro, _ = PINNs.parte6()
            print('\n\nProcesso:', self.it/100,'\n')
            print(L,erro)
        return L        
    def parte6(self): 
        U_predicao = self.parte1(X_teste)
        L2_norm = torch.linalg.norm((U-U_predicao),2)/torch.linalg.norm(U,2)        
        U_predicao = U_predicao.cpu().detach().numpy()
        U_predicao = np.reshape(U_predicao,(256,100),order='F')
        return L2_norm, U_predicao          

                        

# Parte 3
X_teste, Ub, Lb, U_sol, X_CI, U_CI, X_BC, U_BC, X_TBC, U_TBC, X_Treino, dx, U_Treino, X_Treino_NNu, U_Treino_NNu, X_Treino_NNf = ProcDados(X, T, solucao, NNf, NNu)


# Parte 4
X_Treino_NNf, X_Treino_NNu, U_Treino_NNu, X_teste, Tx_f, U = Treinamento(X_Treino_NNf, X_Treino_NNu, U_Treino_NNu, X_teste, U_sol, device)
PINNs = NN(cd); PINNs.to(device)
otim = torch.optim.LBFGS(PINNs.parameters(), L_r, max_iter = max_it, max_eval = None, tolerance_grad = 1e-10, tolerance_change = 1e-10, history_size = 100, line_search_fn = 'strong_wolfe')

# Parte 5 (final)
ini_tempo = time.time()
otim.step(PINNs.parte5)
fim_tempo = time.time() - ini_tempo                
print('Tempo de Processo: %.3f'%(fim_tempo))

# Resultados 
erro_L2, U_Predicao = PINNs.parte6()
print('Erro máximo: %.4f'  % (erro_L2))
plot3(U_Predicao,X_Treino_NNu.cpu().detach().numpy(),U_Treino_NNu, T, X, x, t, solucao)






    

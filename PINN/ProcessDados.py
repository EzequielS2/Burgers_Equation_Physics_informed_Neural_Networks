
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



def ProcDados(X, T, solucao, NNf, NNu):

	X_teste = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))  
	Ub = X_teste[-1]
	Lb = X_teste[0]
	U_sol = solucao.flatten('F')[:,None] 


	# Parte do Treinamento de Dados
	X_CI = np.hstack((X[0,:][:,None], T[0,:][:,None])); U_CI = solucao[:,0][:,None]
	X_BC = np.hstack((X[:,0][:,None], T[:,0][:,None])); U_BC = solucao[-1,:][:,None]
	X_TBC = np.hstack((X[:,-1][:,None], T[:,0][:,None])); U_TBC = solucao[0,:][:,None]
	X_Treino = np.vstack([X_CI, X_BC, X_TBC])  
	dx = np.random.choice(X_Treino.shape[0], NNu, replace=False) 
	U_Treino = np.vstack([U_CI, U_BC, U_TBC])
	X_Treino_NNu = X_Treino[dx, :] 
	U_Treino_NNu = U_Treino[dx,:]      
	X_Treino_NNf = Lb + (Ub-Lb)*lhs(2,NNf); X_Treino_NNf = np.vstack((X_Treino_NNf, X_Treino_NNu)) 
	
	return X_teste, Ub, Lb, U_sol, X_CI, U_CI, X_BC, U_BC, X_TBC, U_TBC, X_Treino, dx, U_Treino, X_Treino_NNu, U_Treino_NNu, X_Treino_NNf   

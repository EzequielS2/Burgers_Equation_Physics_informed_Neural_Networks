

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


def Treinamento(X_Treino_NNf, X_Treino_NNu, U_Treino_NNu, X_teste, U_sol, device):

	X_Treino_NNf = torch.from_numpy(X_Treino_NNf).float().to(device)
	X_Treino_NNu = torch.from_numpy(X_Treino_NNu).float().to(device)
	U_Treino_NNu = torch.from_numpy(U_Treino_NNu).float().to(device)
	X_teste = torch.from_numpy(X_teste).float().to(device)

	Tx_f = torch.zeros(X_Treino_NNf.shape[0],1).to(device)
	U = torch.from_numpy(U_sol).float().to(device)
	
	return X_Treino_NNf, X_Treino_NNu, U_Treino_NNu, X_teste, Tx_f, U




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

from Plt import plot1


def Dados():

	data = scipy.io.loadmat('Solucao_Eq_Burges.mat') 
	t = data['t']  	
	x = data['x']                                                                    
	solucao = data['usol']                             
	X, T = np.meshgrid(x,t)                         
	plot1(torch.from_numpy(x),torch.from_numpy(t),torch.from_numpy(solucao)) 
	
	return data, x, t, solucao, X, T

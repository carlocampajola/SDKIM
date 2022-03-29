#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:31:55 2021

@author: carlo
"""

from src.sdkim_class import beta_tv_torch
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(precision=6)
dtype = torch.float64
device = torch.device("cpu")

## MODEL SUBTYPE: choose between 'DyEKIM', 'DyEKIMext' or 'DyNoKIM'. Defaults to DyNoKIM if unspecified or misspecified
modtype = 'DyEKIM'

if modtype == 'DyEKIM':
    ## DyEKIM without h0
    inds = torch.eye(5)
    inds = inds[:4,:]

elif modtype == 'DyEKIMext':
    ## DyEKIM with h0
    inds = torch.eye(5)

else:
    ## DyNoKIM
    inds = torch.ones((1,5))
    inds[0,4] = 0


model = beta_tv_torch(beta_scal_inds = inds)

np.random.seed(1234)
# set number of spins
N = 30
# set number of samples
N_sample = 1
# set sample length
T = 500

# random J matrix with Gaussian distributed values
J_static_np = np.random.normal(loc = 0/np.sqrt(N), scale = (1/np.sqrt(N)), size = (N,N))
J_static = torch.tensor(J_static_np, dtype=dtype, requires_grad = True)

# random initial configuration
s0_np =  2* np.random.binomial(1,0.5,N) -1# ones(N)
s0 = torch.tensor(s0_np, dtype=dtype)

# h parameters (extfields), can be set to 0 or any value
#extfields =  model.tens(np.zeros(N))
extfields = model.tens(np.random.normal(loc = 0, scale = 1/np.sqrt(N), size = N))

# covariates time series and coupling parameters of covariates (defaults to all zeros)
covariates_T =torch.zeros(T, 1)
covcouplings = torch.zeros(N, 1)

# use reasonable default parameters for the score-driven dynamics
sdd_pars = model.reasonable_pars

npar = sdd_pars.shape[0]
s_T = np.zeros((T, N, N_sample))
f_T = np.zeros((T, npar, N_sample))
scores = np.zeros((T, npar, N_sample))

for sam in range(N_sample):
    # simulate the model - outputs are the simulated score-driven parameters, the simulated spins and the time series of the score
    f_T_sampled,s_T_samp,score_T = model.filter_dyn_par(s0, J_static, extfields, 
                                                        sdd_pars, covariates_T = covariates_T,  
                                                        covcouplings=covcouplings, T=T)
    f_T[:,:,sam] = f_T_sampled.data.numpy()
    s_T[:,:,sam] = s_T_samp.data.numpy()
    scores[:,:,sam] = score_T.data.numpy()
# plot one trajectory
plt.plot(torch.exp(f_T_sampled).data.numpy())
#plt.plot(torch.mean(s_T_samp,1).data.numpy())

# save trajectory for future estimations
np.savez('example_simulation.npz', f_T=f_T, s_T=s_T, J=J_static_np, h=extfields.data.numpy(),
         covariates_T=covariates_T.data.numpy(),
         covcouplings = covcouplings.data.numpy(), sdd_pars = sdd_pars.data.numpy())
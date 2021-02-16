#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:32:26 2021

@author: carlo
"""
## Run simulate.py first to create a time series

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

npz = np.load('example_simulation.npz')

f_T = torch.tensor(npz['f_T'])
s_T = torch.tensor(npz['s_T'])
J = torch.tensor(npz['J'])
extfields = torch.tensor(npz['h'])
covariates_T = torch.tensor(npz['covariates_T'])
covcouplings = torch.tensor(npz['covcouplings'])
sdd_pars = torch.tensor(npz['sdd_pars'])

N_sample = f_T.shape[2]

for sam in range(N_sample):
    s_T_samp = s_T[:,:,sam]
    f_T_samp = f_T[:,:,sam]
    infJ, infh = model.estimate_MS(s_T_samp)

    J_est = torch.tensor(infJ)
    extfields_est = torch.tensor(infh)

    plt.scatter(J.view(-1).data.numpy(), infJ.ravel())
    plt.scatter(extfields.data.numpy(), infh)

    unc_mean_est = model.estimate_const_beta(s_T_samp, J_est, extfields_est, covariates_T, covcouplings, lr=0.05, Steps=300)

    sdd_est = model.estimate_targeted(unc_mean_est, s_T_samp, J_est, extfields_est, covariates_T, covcouplings, lr=0.15, Steps=300, 
                                      rel_improv_tol = 5e-9)
        
    f_T_est = model.filter_dyn_par(s_T_samp, J_est, extfields_est, sdd_est, covariates_T, covcouplings)

    plt.plot(torch.exp(f_T_samp).data.numpy())
    plt.plot(torch.exp(f_T_est-torch.mean(f_T_est)).data.numpy())        
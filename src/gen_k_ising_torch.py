"""
Created on Mon Jul 22 16:21:02 2019

Python code for Score Driven Kinetic Ising Models

@author: Campajola Carlo, Di Gangi Domenico
"""

import numpy as np  
import torch
import math
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
# import matplotlib.pyplot as plt
dtype = torch.float64
torch.set_default_dtype(torch.float64)



class k_ising_torch(object):
    """ This is the parent class for all Kinetic Ising models. It contains all
    the generic methods that are applicable regardless of the specific time-varying
    parametrization
    """

    def __init__(self ):
        self.sT = None #npj.zeros(100,10) # matrix of observations : rows = time, cols = spin ind


    def step_prob(self,st, J, extfields, covariates_t, covcouplings):
        """ Computes the N probabilities of observing a 1 in each element of stp1, given  st
        """
        unnorm_prob = torch.mv(J, st) + extfields + torch.mv(covcouplings, covariates_t)
        norm_prob = torch.mul(torch.exp(unnorm_prob) , 1/(2 * torch.cosh(unnorm_prob)))
        return norm_prob


    def sample_stp1_k_ising(self,st_in,J,extfields, covariates_t , covcouplings):
        """ Given all the parameters sample stp1 conditional on st
        """
        st = st_in.clone()
        sampled_stp1 = torch.ones(st.shape)
        probs1 = self.step_prob(st, J, extfields, covariates_t, covcouplings)
        sampled_stp1[torch.rand(st.shape[0]) > probs1] = -1

        return sampled_stp1.clone()


    def sample_seq_k_ising(self,T,J,extfields,covariates_T, covcouplings,st = None):
        N = J.shape[0]
        s_T = torch.ones((T, N))
        if st is None:
            inds = torch.rand(N) < 0.5
            s_T[0, inds] = -1
            st = s_T[0, :].clone()
        for t in range(0, T-1):
            stp1 = self.sample_stp1_k_ising(st_in=st, J=J, extfields=extfields,
                                              covariates_t=covariates_T[t, :],
                                              covcouplings=covcouplings)
            s_T[t+1,:] = stp1.clone()
            st = stp1.clone()
        return s_T


    def loglik_t(self, st, stp1, covariates_t, J, extfields, covcouplings):
        """This method is equal for all kinetic Ising models considered, i.e. it does not
        change when we consider different update rules or set of time varying parameters.
        Considering N spins and M covariates, it calculates the t-th element of the log-likelihood.
        It takes  as input:
            current (st) and next step (stp1) configurations (N-dimensional np arrays)
            covariates_t (M-dimensional np array)
            J (NxN np array)
            extfields (N-dimensional np array)
            covcouplings (NxM np array)
        """

        if (covariates_t is None) and (extfields is None):
            g = torch.mv(J,st)# + extfields + torch.mv(covcouplings, covariates_t)
        elif (covariates_t is None) and (extfields is not None):
            g = torch.mv(J,st) + extfields #+ torch.mv(covcouplings, covariates_t)
        elif (covariates_t is not None) and (extfields is not None):
            g = torch.mv(J,st) + extfields + torch.mv(covcouplings, covariates_t)

        sg = torch.sum(stp1 *g)
        lt = sg - torch.sum( torch.log(2 * torch.cosh(g)) )
        return lt


    def estimate_J(self, s_T, J_0=None, extfields_0=None, covariates_T=None,
                   covcouplings_0=None, opt_n=1, parsel=[True, False, False], Jdiagonly = False, Steps=2000, lr=0.001, rel_improv_tol=5e-7):
        """
        Estimate static J and eventually also ext_fields and covariate_couplings
        parsel is a vector of booleans that indicates what parameters need to be estimated: J, extfields, covcouplings

        
          allowed combinations: - 100
                                - 110
                                - 111
                                - 010
        """
        T, N = s_T.shape
        # if not parsel[0]:
        #     raise
        if J_0 is None:
            J_0 = self.tens(np.random.normal(loc = 0, scale = (1/np.sqrt(N)), size = (N,N)))  # torch.tensor(J_static_np,requires_grad = True)#
        unPar0 = J_0.view(-1)
        n_par_j = N**2
        if Jdiagonly:
            unPar0 = J_0.diag()
            n_par_j = N
        # estimate J and extfields
        if extfields_0 is None:
            extfields_0 = torch.rand(N, dtype=dtype)
        n_par_ef = N
        if (covcouplings_0 is None) and (covariates_T is not None):
            covcouplings_0 = torch.rand((N, covariates_T.shape[1]), dtype=dtype)

        if parsel[0] & (not parsel[1]) & (not parsel[2]):
            #estimate only J
            def obj_fun(unPar):
                logl_T = 0
                if Jdiagonly:
                    for t in range(T - 1):
                        logl_T += self.loglik_t(s_T[t, :], s_T[t + 1, :], None, torch.diag_embed(unPar),
                                                None, None)
                else:
                    for t in range(T - 1):
                        logl_T += self.loglik_t(s_T[t, :], s_T[t + 1, :], None, unPar.view(N, N),
                                                None, None)
                return - logl_T

        elif parsel[0] & (parsel[1]) & (not parsel[2]):
            unPar0 = torch.cat((unPar0, extfields_0))

            def obj_fun(unPar):
                logl_T = 0
                if Jdiagonly:
                    for t in range(T - 1):
                        logl_T += self.loglik_t(s_T[t,:], s_T[t+1,:], None, torch.diag_embed(unPar[:n_par_j]),
                                                unPar[n_par_j:], None)
                else:
                    for t in range(T - 1):
                        logl_T += self.loglik_t(s_T[t, :], s_T[t + 1, :], None, unPar[:n_par_j].view(N, N),
                                                unPar[n_par_j:], None)
                return - logl_T

        elif parsel[0] & (parsel[1]) & (parsel[2]):
            #estimate J, etfields and covcouplings
            unPar0 = torch.cat((unPar0, extfields_0, covcouplings_0.view(-1)))
            def obj_fun(unPar):
                logl_T = 0
                if Jdiagonly:
                    for t in range(T - 1):
                        covariates_t = covariates_T[t, :]
                        logl_T += self.loglik_t(s_T[t,:], s_T[t+1,:], covariates_t, torch.diag_embed(unPar[:n_par_j]),
                                                unPar[n_par_j:n_par_j+n_par_ef], unPar[n_par_j+n_par_ef:].view(N, covariates_T.shape[1]))
                else:
                    for t in range(T - 1):
                        covariates_t = covariates_T[t, :]
                        logl_T += self.loglik_t(s_T[t, :], s_T[t + 1, :], covariates_t, unPar[:n_par_j].view(N, N),
                                                unPar[n_par_j:n_par_j+n_par_ef], unPar[n_par_j+n_par_ef:].view(N, covariates_T.shape[1]))
                return - logl_T
            
        elif (not parsel[0]) & (parsel[1]) & (not parsel[2]):
            unPar0 = extfields_0
            J_0 = torch.zeros((N,N))
            def obj_fun(unPar):
                logl_T = 0
                for t in range(T - 1):
                    logl_T += self.loglik_t(s_T[t,:], s_T[t+1,:], None, J_0, unPar, None)
                return - logl_T


        unPar_est, diag = self.optim_torch(obj_fun, unPar0, lRate=lr, print_par = False,
                                           opt_n=opt_n, opt_steps=Steps, rel_improv_tol=rel_improv_tol)

        if parsel[0] & (not parsel[1]) & (not parsel[2]):
            if Jdiagonly:
                return torch.diag_embed(unPar_est)
            else:
                return unPar_est.view(N, N)
        elif parsel[0] & parsel[1] & (not parsel[2]):
            if Jdiagonly:
                return torch.diag_embed(unPar_est[:n_par_j]), unPar_est[n_par_j:]
            else:
                return unPar_est[:n_par_j].view(N, N), unPar_est[n_par_j:]
        elif (not parsel[0]) & parsel[1] & (not parsel[2]):
            return unPar_est
        elif all(parsel):
            if Jdiagonly:
                return torch.diag_embed(unPar_est[:n_par_j]), unPar_est[n_par_j:n_par_j+n_par_ef], \
                        unPar_est[n_par_j+n_par_ef:].view(N, covariates_T.shape[1])
            else:
                return unPar_est[:n_par_j].view(N, N), unPar_est[n_par_j:n_par_j+n_par_ef], \
                        unPar_est[n_par_j+n_par_ef:].view(N, covariates_T.shape[1])
        
                        
    def integrand(self, x, u, delta):
        return np.tanh(u + x*np.sqrt(delta))*np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
    
    def integral(self, u, delta):
        if abs(u/math.sqrt(delta) > 4 or delta > 200):
            integ = 1 - 2*math.erf(-u/math.sqrt(delta))
        else:
            integ = quad(self.integrand, a=-np.inf, b=np.inf, args=(u,delta))[0]
        return integ
                        
    def funct(self, u, delta, mi):
        return (self.integral(u,delta) - mi)**2
    
    def f2(self, x,u,delta):
        return (1 - np.tanh(u+x*np.sqrt(delta))**2) * np.exp(-x**2 / 2)/np.sqrt(2*np.pi)
    
    def estimate_MS(self, s_T, with_h=True, with_autocorr=True):
        T,N = s_T.shape
        m = torch.mean(s_T,0).data.numpy()
        s_Tnumpy = np.transpose(s_T.data.numpy())
        C = np.cov(s_Tnumpy)
        D = np.cov(s_Tnumpy[:,1:], s_Tnumpy[:,:(T-1)])[:N,N:]
        invC = np.linalg.inv(C)
        
        b = np.dot(D,invC)
        if not with_autocorr:
            np.fill_diagonal(b, 0)
        infJ = np.zeros((N,N))
        infh = np.zeros(N)
        
        for i in range(N):
            bi = b[i,:]
            mi = m[i]
            
            gamma = np.dot(np.square(bi), 1-np.square(m))
            
            if i==0:
            
                deltahat = 2
                delta = 1
                ite = 0
                while (abs(delta-deltahat)/delta > 0.001):
                    ite += 1
                    delta = deltahat
                
                    u = minimize_scalar(self.funct, args=(delta, mi), method='brent').x
                
                    a = quad(self.f2, a=-np.inf, b=np.inf, args=(u, delta))[0]
                #print(a)
                    if a == 0:
                        raise Exception("got a = 0")
                    
                    deltahat = gamma/a**2
                        
            else:
                u = minimize_scalar(self.funct, args=(delta, mi), method='brent').x
                a = quad(self.f2, a=-np.inf, b=np.inf, args=(u, delta))[0]
            
            infJ[i,:] = bi/a
            g = np.dot(infJ[i,:], m)
            if with_h:
                infh[i] = u - g
            
        return infJ, infh
    
    def estimate_MS_nonstat(self, s_T, with_h=True):
        T,N,N_sample = s_T.shape
        

    def loglik_tot(self, s_T, J, extfields, covcouplings, covariates_t):
        logl_T = 0
        T,N = s_T.shape
        for t in range(T-1):
            covariates_tt = covariates_t[t,]
            logl_T += self.loglik_t(s_T[t,:], s_T[t+1,:], covariates_tt, J, extfields, covcouplings)

        return logl_T


    def decimation_J(self, s_T, J, extfields, covcouplings, covariates_t):
        T,N = s_T.shape

        emptyJ = torch.tensor(np.zeros((N,N)), dtype=dtype)

        maxlik = self.loglik_tot(s_T, J, extfields, covcouplings, covariates_t)
        indlik = self.loglik_tot(s_T, emptyJ, extfields, covcouplings, covariates_t)
        npJ = torch.clone(J).data.numpy()
        sorted_npJ_idx = np.dstack(np.unravel_index(np.argsort(np.abs(npJ.ravel())), (N,N)))
        k = cancidx = 0
        cancellation_sched = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        cancelled = np.floor(np.cumsum(np.multiply(cancellation_sched,N**2)))
        tlhistory = [0]
        deciJhist = [J]
        canchist = [0]
        for i,j in sorted_npJ_idx[0]:
            npJ[i,j] = 0
            deciJ = torch.tensor(npJ, dtype=dtype)
            k += 1
            if cancidx < len(cancellation_sched):
                if (k >= cancelled[cancidx]) & (k-1 < cancelled[cancidx]):
                    cancidx += 1
                    lik = self.loglik_tot(s_T, deciJ, extfields, covcouplings, covariates_t)
                    frac = k/(N**2)
                    tiltlik = lik - (1 - frac) * maxlik - frac * indlik
                    tlhistory.append(tiltlik)
                    deciJhist.append(deciJ)
                    canchist.append(k)
                    print("cancelled " + str(k) + "/" + str(N**2) + ", tilted likelihood " + str(float(tiltlik.data)))
            else:
                lik = self.loglik_tot(s_T, deciJ, extfields, covcouplings, covariates_t)
                frac = k/(N**2)
                tiltlik = lik - (1 - frac) * maxlik - frac * indlik
                tlhistory.append(tiltlik)
                deciJhist.append(deciJ)
                canchist.append(k)
                print("cancelled " + str(k) + "/" + str(N**2) + ", tilted likelihood " + str(float(tiltlik.data)))
        
        max_tl = max(tlhistory)
        maxtl_idx = [i for i,j in enumerate(tlhistory) if j == max_tl]
        maxtl_idx = maxtl_idx[len(maxtl_idx)-1]
        deciJ_maxtl = deciJhist[maxtl_idx]
        cancel_maxtl = canchist[maxtl_idx]
        
        return deciJ_maxtl, cancel_maxtl, maxtl_idx, tlhistory
    
    
    def optim_torch(self, obj_fun_, unPar0, opt_steps=2000, opt_n=1, lRate=0.05, rel_improv_tol=5e-7, no_improv_max_count=25,
                    min_n_iter=30, bandwidth=30, small_grad_th=1e-6,
                    print_flag=True, print_every=1, print_par=True, print_fun=None, plot_flag=False):
        """given a function and a starting vector, run one of different pox optimizations"""
        unPar = unPar0.clone().detach()
        unPar.requires_grad = True
    
        optimizers = [torch.optim.SGD([unPar], lr=lRate, nesterov=False),
                      torch.optim.Adam([unPar], lr=lRate),
                      torch.optim.SGD([unPar], lr=lRate, momentum=0.5, nesterov=True),
                      torch.optim.SGD([unPar], lr=lRate, momentum=0.7, nesterov=True)]
        legend = ["SGD", "Adam", "Nesterov 1", "Nesterov 2"]
    
        def obj_fun():
            return obj_fun_(unPar)
    
        last_print_it=0
        diag = []
        rel_im = np.ones(0)
        num_par = unPar.shape[0]
        i = 0
        loss = obj_fun()
        last_loss = loss.item()
        no_improv_flag = False
        small_grad_flag = False
        nan_flag = False
        no_improv_count = 0
        while (i <= opt_steps) and (not no_improv_flag) & (not small_grad_flag) & (not nan_flag):
            # define the loss
            loss = obj_fun()
            # set all gradients to zero
            optimizers[opt_n].zero_grad()
            # compute the gradients
            loss.backward(retain_graph=True)
            # take a step
            optimizers[opt_n].step()
    
            # check improvement
            #print((i, loss.item()))
            rel_improv = (last_loss - loss.item())
            if not (loss.item() == 0):
                rel_improv = rel_improv/loss.abs().item()
            rel_im = np.append(rel_im, rel_improv)
            last_loss = loss.item()
            if i > min_n_iter:
                roll_rel_im = rel_im[-bandwidth:].mean()
                if roll_rel_im < rel_improv_tol:
                    no_improv_count = no_improv_count + 1
                else:
                    no_improv_count = 0
                if no_improv_count > no_improv_max_count:
                    no_improv_flag = True
            else:
                roll_rel_im = rel_im.mean()
    
            # check the gradient's norm
            grad_norm = unPar.grad.norm().item()
            small_grad_flag = unPar.grad.norm() < small_grad_th
    
            # check presence of nans in opt vector
            nan_flag = torch.isnan(unPar).any().item()
    
            #store info and Print them when required
            if print_fun is not None:
                fun_val = print_fun(unPar.clone().detach())
                diag.append((loss.item(), fun_val))
            else:
                diag.append((loss.item()))
            if print_flag:
                tmp = unPar.data
                if i//print_every > last_print_it:
                    last_print_it = i//print_every
                    if print_fun is not None:
                        print((i, opt_steps, loss.item(), grad_norm, rel_improv, roll_rel_im, no_improv_count, num_par, lRate, fun_val))
                    elif print_par:
                        print((i, opt_steps, loss.item(), grad_norm, rel_improv, roll_rel_im, no_improv_count, num_par, lRate, tmp[-4:]))
                    else:
                        print((i, opt_steps, loss.item(), grad_norm, rel_improv, roll_rel_im, no_improv_count, num_par, lRate))
    
            i = i+1

        print(no_improv_flag, small_grad_flag, nan_flag)

        # if plot_flag:
            # plt.figure()
            # plt.plot(diag)
            # plt.legend(legend[opt_n])
    
        unPar_est = unPar.clone()
        return unPar_est, diag

    def tens(self, vec):
        return torch.tensor(vec, dtype=dtype)















#

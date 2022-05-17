#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:21:02 2019

@author: domenico
"""

from src.gen_k_ising_torch import k_ising_torch
import torch

dtype = torch.float64




class beta_tv_torch(k_ising_torch):
    """Kinetic Ising Model with Score Driven beta (inverse temperature)
    f_t is log(beta) in this case (to ensure positivity of beta)
    Input:
        beta_scal_inds: a M x 5 binary (0/1) matrix, where M is the number of time-varying parameters (M <= 5),
        identifying which KIM parameters the time-varying parameters should multiply.
        If element (i,j) is 1, then the i-th score-driven parameter multiplies the j-th KIM parameter.
        Columns should not contain more than one non-zero element to avoid endogeneity of score-driven parameters.
        Columns correspond to: diagonal of J, rest of J, h, b, h0. Default is 5 time-varying parameters,
        one for each.
        If a column is entirely 0, parameter is assumed to be absent.
        Only the last row can be associated to h0
    """
    def __init__(self, beta_scal_inds = torch.eye(5)):
        k_ising_torch.__init__(self)
        npar = beta_scal_inds.shape[0]
        self.reasonable_pars = torch.tensor([0,0.95,0.01],requires_grad = True).repeat(1,npar).view(-1,3) #  for testing dgp or opt starting points
        self.opt_start_pars =  self.reasonable_pars
        self.beta_scal_inds = beta_scal_inds

    def sd_unc_mean(self,ssd_pars):
        unc_mean = ssd_pars[0,:]/(1-ssd_pars[1,:])
        return unc_mean


    def scale_par_beta(self, beta, J, extfields, covcouplings, \
                     inds = None):
        """
        return a scaled version of the parameters if the corresponding inds
        is true. The same parameter otherwise
        """
        if inds is None:
            inds_0 = self.beta_scal_inds
            inds = inds_0[torch.where(inds_0[:,4] == 0)[0],:]
            
        k = J.shape[0]
        
        if any(inds[:,0]):
            Jd = J.clone() * torch.eye(k) * torch.matmul(beta,inds)[0]
        else:
            Jd = J.clone() * torch.eye(k)
        if any(inds[:,1]):
            Jo = J.clone() *(torch.ones((k,k)) - torch.eye(k)) * torch.matmul(beta,inds)[1]
        else:
            Jo = J.clone() *(torch.ones((k,k)) - torch.eye(k))
            
        J = Jd + Jo
        #J = J.clone()*torch.eye(k)* torch.matmul(beta,inds)[0]\
        #+ J.clone()*(torch.ones((k,k)) - torch.eye(k))*torch.matmul(beta,inds)[1]
        
        if any(inds[:,2]):
            extfields = extfields.clone() * torch.matmul(beta,inds)[2]
        
        if any(inds[:,3]):
            covcouplings = covcouplings.clone() * torch.matmul(beta,inds)[3]
        
        return J, extfields, covcouplings
    
    def beta_scal_g1_g2(self, st, J, extfields, covcouplings, covariates_t, inds=None):
        """
        return a g vector that will multiply beta and a g_none vector that will not
        """
        N= st.shape[0]
        if extfields is None:
            extfields = torch.zeros(N)
        if covariates_t is None:
            covcouplings = torch.zeros(N, 1)
            covariates_t = torch.zeros(1)
        if inds is None:
            inds = self.beta_scal_inds
        g_none = torch.zeros(N)
        
        if torch.sum(inds,0)[4] == 1:
            g = torch.zeros((inds.shape[0]-1, N))
        else:
            g = torch.zeros((inds.shape[0], N))
        
        for i in range(g.shape[0]):
            g[i,:] = J.diag()*st*inds[i,0] + torch.matmul((J - torch.diag(J.diag())), st)*inds[i,1] +\
                extfields*inds[i,2] + torch.matmul(covcouplings, covariates_t)*inds[i,3]
        
        indsum = torch.sum(inds,0)
        g_none = J.diag()*st*(1-indsum[0]) + torch.matmul((J - torch.diag(J.diag())), st)*(1-indsum[1]) +\
            extfields*(1-indsum[2]) + torch.matmul(covcouplings, covariates_t)*(1-indsum[3])
        
        return g,g_none

    def sample_stp1_scaled(self,beta,st, J, extfields,covariates_t , covcouplings):
        """ sample the K ising where not all parameters are rescaled by beta"""
        # first scale
        J, extfields, covcouplings = self.scale_par_beta(beta, J, extfields, covcouplings)
        #then sample
        stp1_ = self.sample_stp1_k_ising(st_in=st, J=J, extfields=extfields,
                                            covariates_t=covariates_t ,
                                            covcouplings=covcouplings)
        return stp1_


    def loglik_t_beta_scal(self,beta,st,stp1, J, extfields,covariates_t , covcouplings):
        """ compute the loglikelihood of K ising where not all parameters are rescaled by beta"""
        # first scale
        J, extfields, covcouplings = self.scale_par_beta(beta, J, extfields, covcouplings)
        #then sample
        loglike = self.loglik_t(st=st,stp1=stp1,J= J, extfields= extfields,\
                                           covariates_t=covariates_t , covcouplings= covcouplings)

        return loglike


    def score(self, st_in, stp1_in, J, extfields, f_t, covariates_t,
                  covcouplings):
        """ The score of the single observation likelihood
        """
        st = st_in.clone()
        stp1 = stp1_in.clone()
        
        inds = self.beta_scal_inds

        #
        g_vec, g_none = self.beta_scal_g1_g2(st, J, extfields, covcouplings, covariates_t,\
                     inds = inds)
        
        beta_t = torch.exp(f_t)[torch.where(inds[:,4] == 0)[0]]
        
        g = torch.mv(g_vec.transpose(0,1), beta_t) + g_none
        tgh_beta_g = torch.tanh(g)
        
        d_dbeta = torch.mv(g_vec, (stp1 - tgh_beta_g ))
        score_tmp = beta_t*d_dbeta # derivative wrt to log(beta)

        d2_dbeta2_tmp =  torch.mv(g_vec**2, 1 - tgh_beta_g**2) # here the hessian is just the diagonal
        d2_dbeta2 = torch.where(d2_dbeta2_tmp == 0, torch.ones_like(d2_dbeta2_tmp), d2_dbeta2_tmp)
        fisher_tmp = (beta_t**2) * d2_dbeta2

        scaled_score_tmp =  d_dbeta/torch.sqrt(d2_dbeta2)

        if torch.sum(inds,0)[4] == 1:
            d_dh0 = torch.sum((stp1 - tgh_beta_g))
            d2_dh02_tmp = torch.sum((1 - tgh_beta_g**2))
            d2_dh02 = torch.where(d2_dh02_tmp == 0, torch.ones_like(d2_dh02_tmp), d2_dh02_tmp)
            score = torch.cat((score_tmp, d_dh0.view(1)))
            fisher = torch.cat((fisher_tmp, d2_dh02.view(1)))
            ss_h0 = torch.div(d_dh0, torch.sqrt(d2_dh02))
            scaled_score = torch.cat((scaled_score_tmp, ss_h0.view(1)))
        else:
            score = score_tmp
            fisher = fisher_tmp
            scaled_score = scaled_score_tmp

        return score,  fisher, scaled_score

    def sd_update(self, st, stp1, w, B, A, J, f_t_in, extfields, covariates_t ,
                  covcouplings):
        f_t = f_t_in.clone()
        
        score_t ,fisher_t,resc_score_t = self.score(st_in=st, stp1_in=stp1, f_t = f_t, J=J, extfields=extfields, \
                                                    covariates_t=covariates_t , covcouplings=covcouplings)
        
        f_tp1 = w + B*f_t + A * resc_score_t

        return f_tp1, resc_score_t

    def filter_dyn_par(self,s_T, J, extfields, sdd_pars_vec, covariates_T, covcouplings, T=None, \
                       likeFlag =False):
        """ Each version of the Dynamical parameters approach requires a different number of
        static parameters.
        sdd_pars_vec is assumed to be the vector of restricted parameters
        Can be used to simulate the dgp by setting T > 0 and s_T = starting vector of spins
        """
        (w,B,A) = self.sdd_pars_from_vec(sdd_pars_vec)
        logl_T = torch.tensor(0,dtype=dtype)
        inds = self.beta_scal_inds
        if T is None:
            dgp = False
            T = s_T.shape[0]
            st = torch.ones(1) * s_T[0,:]
        else:
            dgp = True
            st =  s_T.clone()
            s_T = torch.zeros((T,st.shape[0]))
            s_T[0,:] = st.clone()
            score_T = torch.zeros((1,inds.shape[0]))
            score_t =  torch.zeros((1,inds.shape[0]))
            
        f_t = torch.div(w,(1-B)) # initialize to the unconditional mean
      
        #f_T = torch.zeros( 1)
        covariates_tt = None
        for t in range(0,T-1):
            beta_t = torch.exp(f_t)[torch.where(inds[:,4] == 0)[0]]
            if torch.sum(inds,0)[4] == 1:
                extfields_eff = extfields.clone() + f_t[torch.where(inds[:,4] == 1)[0]]
            else:
                extfields_eff = extfields.clone()
            if covcouplings is not None:
                covariates_tt = covariates_T[t, :]
            if dgp:
                stp1_ = self.sample_stp1_scaled(beta_t,st=st, J= J, extfields= extfields_eff,
                                                covariates_t= covariates_tt ,
                                                covcouplings=covcouplings)
                stp1 = torch.tensor(stp1_.detach().numpy() ) #useless

                s_T[t+1,:] = stp1.clone()
                score_T = torch.cat((score_T,score_t.view(1,-1)),0)
            else:
                stp1 = s_T[t+1,:]

            f_tp1,score_t = self.sd_update(st=st.clone(), stp1=stp1.clone(), f_t_in = f_t, J=J, w=w, B=B, A=A,\
                                           extfields=extfields_eff, covariates_t=covariates_tt ,\
                                           covcouplings=covcouplings)
           # print(score_t)

            if likeFlag:
                logl_t = self.loglik_t_beta_scal(beta_t,st=st,stp1=stp1,J=J, extfields=extfields_eff,\
                                         covariates_t=covariates_tt , covcouplings=covcouplings)
                logl_T += logl_t
            
            if t == 0:
                f_T = f_t.clone().view(1,-1)
            else:
                f_T = torch.cat((f_T,f_t.clone().view(1,-1)), 0)
            st = stp1.clone() 
            f_t = f_tp1.clone() 
        f_T = torch.cat((f_T, f_tp1.clone().view(1,-1)), 0)
        if dgp:
            return f_T,s_T,  score_T
        elif likeFlag:
            return f_T, logl_T
        else:
            return f_T

    def sdd_pars_from_vec(self,re_par_vec):
        """ appropriately reshape the vector of static parameters for the
        score driven dynamics (ssd).
        the parameters in vec form are supposed to be restricted
        """
        return (re_par_vec[:,0], re_par_vec[:,1], re_par_vec[:,2]) # (w, B, A)


    def un2re_parVec(self, un_par_vec):
        """
        get the vector of unrestricted sdd parameters and return their restricted version
        """
        re_par_vec = un_par_vec.clone()
        x =  torch.exp(un_par_vec[:,1])
        re_par_vec[:,1] = torch.div(x,1 + x) # restrict B in (0,1)
        x =  torch.exp(un_par_vec[:,2])
        re_par_vec[:,2] = torch.div(x,1 + x) # restrict B in (0,1)


        return re_par_vec


    def re2un_parVec(self, re_par_vec):
        """
        get the vector of unrestricted sdd parameters and return their restricted version
        """
        un_par_vec = re_par_vec.clone()
        un_par_vec[:,1] = torch.log( torch.div(re_par_vec[:,1],1 - re_par_vec[:,1]) ) # map (0,1) to R
        un_par_vec[:,2] = torch.log( torch.div(re_par_vec[:,2],1 - re_par_vec[:,2]) ) # map (0,1) to R

        return un_par_vec


    def estimate_const_beta(self,s_T,J,extfields, covariates_T , covcouplings,
                            opt_n = 1 ,Steps = 200, lr=0.01):


        def obj_fun(unPar):
            rePar = torch.cat((unPar.view(-1,1), torch.zeros((unPar.shape[0], 2))),1)
            
            f_T_filtered,logl_T = self.filter_dyn_par(s_T ,J,extfields,rePar,\
                                                      covariates_T=covariates_T, \
                                                      covcouplings=covcouplings, likeFlag=True)
            #print(logl_T)
            #print(unPar)
            return -logl_T

        unPar0 = self.re2un_parVec(self.reasonable_pars).clone().detach()[:,0]*torch.ones(self.reasonable_pars.shape[0])
        unPar_est, diag = self.optim_torch(obj_fun, unPar0, lRate=lr,
                                           opt_n = opt_n, opt_steps = Steps)
        par_est = unPar_est.clone().detach()
        return par_est


    def estimate_targeted(self,unc_mean,s_T,J,extfields, covariates_T , \
                          covcouplings, opt_n = 1, Steps = 200, lr=0.01, rel_improv_tol = 5e-7):


        unc_mean = unc_mean * torch.ones(self.reasonable_pars.shape[0])
        unPar0 = self.re2un_parVec(self.reasonable_pars).clone().detach()[:,1:].reshape(-1)
        unPar0.requires_grad_(True)


        def obj_fun(unPar):
            unPar = unPar.reshape((self.reasonable_pars.shape[0],2))
            rePar = self.un2re_parVec(torch.cat((torch.zeros((unPar.shape[0], 1)),unPar), 1) )
            rePar[:,0] = unc_mean * (1 - rePar[:,1]) #- rePar[2].clone()**2 / (2 * (1 - rePar[1]))
            # rePar[0] = rePar[0].clone() 
            f_T_filtered,logl_T = self.filter_dyn_par(s_T ,J,extfields,rePar,\
                                                      covariates_T = covariates_T, \
                                                      covcouplings=covcouplings,likeFlag = True)
            return -logl_T
        
        unPar_est, diag = self.optim_torch(obj_fun, unPar0, lRate=lr, rel_improv_tol = rel_improv_tol,
                                           opt_n = opt_n, opt_steps = Steps)
        
        par_est = self.un2re_parVec(torch.cat((unc_mean.reshape(self.reasonable_pars.shape[0],1), unPar_est.reshape((self.reasonable_pars.shape[0],2))), 1)).clone()
        par_est[:,0] = par_est[:,0] * (1 - par_est[:,1]) #- par_est[2]**2 / (2 * (1 - par_est[1]))

        return par_est

    
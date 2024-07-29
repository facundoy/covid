#/usr/bin/env python3

import numpy as np
import scipy.stats as st
import operator
from functools import reduce
from constants import *
import pdb

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
import torch.optim as optim

from qpth.qp import QPFunction

###


import torch.nn.functional as F
import copy
###
'''
class Net(nn.Module):
    def __init__(self, X, Y, hidden_layer_sizes):
        super(Net, self).__init__()

        self.n_in = X.shape[1]
        self.n_out = 2*Y.shape[1]
        #self.prior = prior_instance

        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.1, 0.1))
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-3, -2))

        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-3, -2))

        self.sig = Parameter(torch.ones(1, Y.shape[1], device=DEVICE))
        
    def forward(self, X):

        eps_W = Variable(self.W_mu.data.new(self.W_mu.size()).normal_())
        eps_b = Variable(self.b_mu.data.new(self.b_mu.size()).normal_())

        # sample parameters
        std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
        std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)

        W = self.W_mu + 1 * std_w * eps_W
        b = self.b_mu + 1 * std_b * eps_b

        output = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)
        output = output[:,:int(self.n_out/2)]
        return output, \
            self.sig.expand(X.size(0), self.sig.size(1))
    
    def set_sig(self, X, Y):
        eps_W = Variable(self.W_mu.data.new(self.W_mu.size()).normal_())
        eps_b = Variable(self.b_mu.data.new(self.b_mu.size()).normal_())

        # sample parameters
        std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
        std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)

        W = self.W_mu + 1 * std_w * eps_W
        b = self.b_mu + 1 * std_b * eps_b

        var = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)
        var = output[:,int(self.n_out/2):]

        var = 1/X.shape[0]*torch.sum(var,dim=0)
        self.sig.data = torch.sqrt(torch.abs(var)).data.unsqueeze(0)


class Net(nn.Module):
    def __init__(self, X, Y, hidden_layer_sizes):
        super(Net, self).__init__()

        # Initialize linear layer with least squares solution
        X_ = np.hstack([X, np.ones((X.shape[0],1))])
        print (X_.shape)
        Theta = np.linalg.solve(X_.T.dot(X_), X_.T.dot(Y))
        
        self.lin = nn.Linear(X.shape[1], Y.shape[1])
        W,b = self.lin.parameters()
        W.data = torch.Tensor(Theta[:-1,:].T)
        b.data = torch.Tensor(Theta[-1,:])
        
        # Set up non-linear network of 
        # Linear -> BatchNorm -> ReLU -> Dropout layers
        layer_sizes = [X.shape[1]] + hidden_layer_sizes
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], Y.shape[1])]
        self.net = nn.Sequential(*layers)
        self.sig = Parameter(torch.ones(1, Y.shape[1], device=DEVICE))
        print (self.sig)
        
    def forward(self, x):
        print ((self.lin(x) + self.net(x)).shape)
        return self.lin(x) + self.net(x), \
            self.sig.expand(x.size(0), self.sig.size(1))
    
    def set_sig(self, X, Y):
        Y_pred = self.lin(X) + self.net(X)
        var = torch.mean((Y_pred-Y)**2, 0)
        self.sig.data = torch.sqrt(var).data.unsqueeze(0)
'''

class Net(nn.Module):
    def __init__(self, X, Y, hidden_layer_sizes):
        super(Net, self).__init__()

        layer_sizes = [X.shape[1]] + hidden_layer_sizes
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers = layers + [nn.Linear(layer_sizes[-1], Y.shape[1]*2)]
        self.net = nn.Sequential(*layers)
        
        self.sig = Parameter(torch.ones(1, Y.shape[1], device=DEVICE))
        
        
    def forward(self, X):

        prediction = self.net(X)
        mu = prediction[0,:]
        sigma = prediction[1:]
        
        sigma = F.softplus(sigma)+1e-6
        
        return mu, sigma
    
    def set_sig(self, X, Y):
        
        prediction = self.net(X)
        mu = prediction[:,0:1]
        var = prediction[:,1:]

        var = 1/x.shape[0]*torch.sum(var,dim=0)
        self.sig.data = torch.sqrt(torch.abs(var)).data.unsqueeze(0)
'''
def GLinearApprox(gamma_under, gamma_over):
    """ Linear (gradient) approximation of G function at z"""
    class GLinearApproxFn(Function):
        @staticmethod    
        def forward(ctx, z, mu, sig):
            ctx.save_for_backward(z, mu, sig)
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            res = torch.DoubleTensor((gamma_under + gamma_over) * p.cdf(
                z.cpu().numpy()) - gamma_under)
            if USE_GPU:
                res = res.cuda()
            return res
        
        @staticmethod
        def backward(ctx, grad_output):
            z, mu, sig = ctx.saved_tensors
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            pz = torch.tensor(p.pdf(z.cpu().numpy()), dtype=torch.double, device=DEVICE)
            
            dz = (gamma_under + gamma_over) * pz
            dmu = -dz
            dsig = -(gamma_under + gamma_over)*(z-mu) / sig * pz
            return grad_output * dz, grad_output * dmu, grad_output * dsig

    return GLinearApproxFn.apply


def GQuadraticApprox(gamma_under, gamma_over):
    """ Quadratic (gradient) approximation of G function at z"""
    class GQuadraticApproxFn(Function):
        @staticmethod
        def forward(ctx, z, mu, sig):
            ctx.save_for_backward(z, mu, sig)
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            res = torch.DoubleTensor((gamma_under + gamma_over) * p.pdf(
                z.cpu().numpy()))
            if USE_GPU:
                res = res.cuda()
            return res
        
        @staticmethod
        def backward(ctx, grad_output):
            z, mu, sig = ctx.saved_tensors
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            pz = torch.tensor(p.pdf(z.cpu().numpy()), dtype=torch.double, device=DEVICE)
            
            dz = -(gamma_under + gamma_over) * (z-mu) / (sig**2) * pz
            dmu = -dz
            dsig = (gamma_under + gamma_over) * ((z-mu)**2 - sig**2) / \
                (sig**3) * pz
            
            return grad_output * dz, grad_output * dmu, grad_output * dsig

    return GQuadraticApproxFn.apply


class SolveSchedulingQP(nn.Module):
    """ Solve a single SQP iteration of the scheduling problem"""
    def __init__(self, params):
        super(SolveSchedulingQP, self).__init__()
        self.c_ramp = params["c_ramp"]
        self.n = params["n"]
        D = np.eye(self.n - 1, self.n) - np.eye(self.n - 1, self.n, 1)
        self.G = torch.tensor(np.vstack([D,-D]), dtype=torch.double, device=DEVICE)
        self.h = (self.c_ramp * torch.ones((self.n - 1) * 2, device=DEVICE)).double()
        self.e = torch.DoubleTensor()
        if USE_GPU:
            self.e = self.e.cuda()
        
    def forward(self, z0, mu, dg, d2g):
        nBatch, n = z0.size()
        
        Q = torch.cat([torch.diag(d2g[i] + 1).unsqueeze(0) 
            for i in range(nBatch)], 0).double()
        p = (dg - d2g*z0 - mu).double()
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        
        out = QPFunction(verbose=False)(Q, p, G, h, self.e, self.e)
        return out
'''

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc,self).__init__()

        # self.N = nn.Parameter(torch.tensor(1938000.0,device=DEVICE))
        # self.Ca = nn.Parameter(torch.tensor(0.425,device=DEVICE))
        # self.Cp = nn.Parameter(torch.tensor(1.0,device=DEVICE))
        # self.Cm = nn.Parameter(torch.tensor(1.0,device=DEVICE))
        # self.Cs = nn.Parameter(torch.tensor(1.0,device=DEVICE))
        # self.alpha = nn.Parameter(torch.tensor(0.4875,device=DEVICE))
        # self.delta = nn.Parameter(torch.tensor(0.1375,device=DEVICE))
        # self.mu = nn.Parameter(torch.tensor(0.928125,device=DEVICE))
        # self.gamma = nn.Parameter(torch.tensor(1/3.5,device=DEVICE))
        # self.lambdaa = nn.Parameter(torch.tensor(1/7,device=DEVICE))
        # self.lambdap = nn.Parameter(torch.tensor(1/1.5,device=DEVICE))
        # self.lambdam = nn.Parameter(torch.tensor(1/5.5,device=DEVICE))
        # self.lambdas = nn.Parameter(torch.tensor(1/5.5,device=DEVICE))
        # self.rhor = nn.Parameter(torch.tensor(1/15,device=DEVICE))
        # self.rhod = nn.Parameter(torch.tensor(1/13.3,device=DEVICE))

        # self.gamma = nn.Parameter(torch.tensor(3.5,device=DEVICE))
        # self.lambdaa = nn.Parameter(torch.tensor(7.0,device=DEVICE))
        # self.lambdap = nn.Parameter(torch.tensor(1.5,device=DEVICE))
        # self.lambdam = nn.Parameter(torch.tensor(5.5,device=DEVICE))
        # self.lambdas = nn.Parameter(torch.tensor(5.5,device=DEVICE))
        # self.rhor = nn.Parameter(torch.tensor(15.0,device=DEVICE))
        # self.rhod = nn.Parameter(torch.tensor(13.3,device=DEVICE))
        self.N = 1938000.0
        self.Ca = 0.425
        self.Cp = 1.0
        self.Cm = 1.0
        self.Cs = 1.0
        self.alpha = 0.4875
        self.delta = 0.1375
        self.mu = 0.928125
        self.gamma = 3.5
        self.lambdaa = 7.0
        self.lambdap = 2.0
        self.lambdam = 5.0
        self.lambdas = 5.0
        self.rhor = 13.3
        self.rhod = 1/13.3
        #Don't need nn.parameter anymore
        self.beta = 0.4
        self.t = 0

        self.E = 0
        self.Ia = 0
        self.I = 0

    def set_params(self,params):
        self.beta = params[0]
        self.Ca = params[1]
        self.alpha = params[2]
        self.delta = params[3]
        self.rhod = params[4]
        self.mu = params[5]
        self.E = params[6]
        self.Ia = params[7]
        self.Ip = params[8]
        self.Im = params[9]
        self.Is = params[10]
        self.Hr = params[11]
        self.Hd = params[12]

    def forward(self,t,y):
        """defining y0 etc, y is a vector 10, one dimension is 10, another dimension is time
        extract which y you have to compare against the ground truth data whisch is hopstializations
        HR and HD (maybe HR + HD) --> need to verify - y is 10 dimension but data.csv is one deimsion, need 
        to extract relevant infomration maybe sum it, adn then compare it against he real wordl data
        And then do task los"""

        """MLE notes w/ regard to optimization --> MLE does not consider gamma under predicted
        and gamma over predicted whereas task loss consdiers both of those. MLE is just trying to
        fit the data.  Defining a very simple task loss --> not just fit data and consider some penalty
        voer and under prediction (task loss), want to balance that"""
        
        """To mimic CMU method, MLE + Task Loss"""
        S, E, Ia, Ip, Im, Is, Hr, Hd, R, D = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]
        # dt = 1
        dt = t - self.t
        self.t = t
        dSE = S * (1-torch.exp(-self.beta*(self.Ca*Ia+self.Cp*Ip+self.Cm*Im+self.Cs*Is)*dt/self.N))
        dEIa = E * self.alpha * (1-torch.exp(-self.gamma*dt))
        dEIp = E * (1-self.alpha) * (1-torch.exp(-self.gamma*dt))
        dIaR = Ia * (1-torch.exp(-self.lambdaa*dt))
        dIpIm = Ip * self.mu * (1-torch.exp(-self.lambdap*dt))
        dIpIs = Ip * (1-self.mu) * (1-torch.exp(-self.lambdap*dt))
        dImR = Im * (1-torch.exp(-self.lambdam*dt))
        dIsHr = Is * self.delta * (1-torch.exp(-self.lambdas*dt))
        dIsHd = Is * (1-self.delta) * (1-torch.exp(-self.lambdas*dt))
        dHrR = Hr * (1-torch.exp(-self.rhor*dt))
        dHdD = Hd * (1-torch.exp(-self.rhod*dt))

        dS = -dSE
        dE = dSE - dEIa - dEIp
        dIa = dEIa - dIaR
        dIp = dEIp - dIpIs - dIpIm
        dIm = dIpIm - dImR
        dIs = dIpIs - dIsHr - dIsHd
        dHr = dIsHr - dHrR
        dHd = dIsHd - dHdD
        dR = dHrR
        dD = dHdD
      
        dy = torch.hstack([dS,dE,dIa,dIp,dIm,dIs,dHr,dHd,dR,dD])
        
        # y = y + dy
        # y = torch.concatenate([y, dH])
        # import pdb
        # pdb.set_trace()

        # y[0] = S + dS
        # y[1] = E + dE
        # y[2] = Ia + dIa
        # y[3] = Ip + dIp
        # y[4] = Im + dIm
        # y[5] = Is + dIs
        # y[6] = Hr + dHr
        # y[7] = Hd + dHd
        # y[8] = R + dR
        # y[9] = D + dD

        return dy
    
    def reset_t(self):
        self.t = 0

class CalibrationNN(nn.Module):
    def __init__(self):
        super(CalibrationNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 13)
        self.min_value = torch.tensor([0.0, 0.0, 0.0, 0.1, 1/20.0, 0.925, 0, 0, 0, 0, 0, 0, 0], device=DEVICE)
        self.max_value = torch.tensor([1.0, 1.0, 1.0, 0.3, 1/13.0, 0.975, 5000, 30000, 10, 10, 10, 10, 5], device=DEVICE)
        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()


    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        beta = self.fc3(x)
        out = self.min_value + (self.max_value-self.min_value)*self.sigmoid(beta)
        return out


import numpy as np
import torch
import pdb 
import pandas as pd
# import yaml
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = torch.device('cuda')
dtype = torch.float

class ODE(nn.Module):
    def __init__(self, params, device):
        super(ODE, self).__init__()
    
        county_id = params['county_id']
        abm_params = f'Data/{county_id}_generated_params.yaml'
        # print("ABM params file: ", abm_params)

        #Reading params
        # with open(abm_params, 'r') as stream:
        #     try:
        #         abm_params = yaml.safe_load(stream)
        #     except yaml.YAMLError as exc:
        #         print('Error in reading parameters file')
        #         print(exc)
        
        # params.update(abm_params)
        self.params = params
        self.device = device
        # self.num_agents = self.params['num_agents'] # Population
        self.num_agents = 1938000 # Population
        self.t = 0

    def reset_t(self):
        self.t = 0


class SEIRM(ODE):
    def __init__(self, params, learnable_params, device):
        super().__init__(params,device)
        self.beta = learnable_params['beta']
        self.alpha = learnable_params['alpha']
        self.gamma = learnable_params['gamma']
        self.mu = learnable_params['mu']

        self.new_infections = torch.zeros(100, device=device)
        self.new_deaths = torch.zeros(100, device=device)
    
    def init_compartments(self,learnable_params):
        ''' let's get initial conditions '''
        initial_infections_percentage = learnable_params['initial_infections_percentage']
        initial_conditions = torch.empty((5)).to(self.device)
        no_infected = (initial_infections_percentage / 100) * self.num_agents # 1.0 is ILI
        initial_conditions[2] = no_infected
        initial_conditions[0] = self.num_agents - no_infected
        print('initial infected',no_infected)

        return initial_conditions

    def forward(self, t, state):
        """
        Computes ODE states via equations       
            state is the array of state value (S,E,I,R,M)
        """
        
        # to make the NN predict lower numbers, we can make its prediction to be N-Susceptible
        dSE = self.beta * state[0] * state[2] / self.num_agents 
        dEI = self.alpha * state[1] 
        dIR = self.gamma * state[2] 
        dIM = self.mu * state[2] 

        dS  = -1.0 * dSE
        dE  = dSE - dEI
        dI = dEI - dIR - dIM
        dR  = dIR
        dM  = dIM

        # concat and reshape to make it rows as obs, cols as states
        dstate = torch.stack([dS, dE, dI, dR, dM], 0)

        # for integer values, save
        if not t - torch.round(t) == 0:
            t_int = int(t.item())
            self.new_infections[t_int] = dEI
            self.new_deaths[t_int] = dIM
            # print(state)
            # print(dstate)
            # pdb.set_trace()

        # update state
        state = state + dstate
        self.t = t
        # print(t)
        return dstate

'''
class SolveScheduling(nn.Module):
    """ Solve the entire scheduling problem, using sequential quadratic 
        programming. """
    def __init__(self, params):
        super(SolveScheduling, self).__init__()
        self.params = params
        self.c_ramp = params["c_ramp"]
        self.n = params["n"]
        
        D = np.eye(self.n - 1, self.n) - np.eye(self.n - 1, self.n, 1)
        self.G = torch.tensor(np.vstack([D,-D]), dtype=torch.double, device=DEVICE)
        self.h = (self.c_ramp * torch.ones((self.n - 1) * 2, device=DEVICE)).double()
        self.e = torch.DoubleTensor()
        if USE_GPU:
            self.e = self.e.cuda()
        
    def forward(self, mu, sig):
        nBatch, n = mu.size()
        
        # Find the solution via sequential quadratic programming, 
        # not preserving gradients
        z0 = mu.detach() # Variable(1. * mu.data, requires_grad=False)
        mu0 = mu.detach() # Variable(1. * mu.data, requires_grad=False)
        sig0 = sig.detach() # Variable(1. * sig.data, requires_grad=False)
        for i in range(20):
            dg = GLinearApprox(self.params["gamma_under"], 
                self.params["gamma_over"])(z0, mu0, sig0)
            d2g = GQuadraticApprox(self.params["gamma_under"], 
                self.params["gamma_over"])(z0, mu0, sig0)
            z0_new = SolveSchedulingQP(self.params)(z0, mu0, dg, d2g)
            solution_diff = (z0-z0_new).norm().item()
            print("+ SQP Iter: {}, Solution diff = {}".format(i, solution_diff))
            z0 = z0_new
            if solution_diff < 1e-10:
                break
                  
        # Now that we found the solution, compute the gradient-propagating 
        # version at the solution
        dg = GLinearApprox(self.params["gamma_under"], 
            self.params["gamma_over"])(z0, mu, sig)
        d2g = GQuadraticApprox(self.params["gamma_under"], 
            self.params["gamma_over"])(z0, mu, sig)
        return SolveSchedulingQP(self.params)(z0, mu, dg, d2g)
'''

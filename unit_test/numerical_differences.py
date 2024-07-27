## This file contains numerical differences functions for various SE3 to euclidian groups
## Author : Avadesh Meduri
## Date : 02/05/2024
import torch
import numpy as np
import pinocchio as pin

tol = 1e-4

# Classical numdiff
def numdiff(f,x0,h=1e-6):
    f0 = f(x0)[0].copy()
    x = x0.copy()
    Fx = []
    for ix in range(len(x)):
        x[ix] += h
        Fx.append((f(x)[0]-f0)/h)
        x[ix] = x0[ix]
    
    return np.array(Fx).T

# Numerical difference function on SE3 variable and Euclidian output
def numdiffSE3toEuclidian(f,x0, rmodel, h=1e-6):
    f0 = f(x0)[0]
    Fx = []
    nq, nv = rmodel.nq, rmodel.nv
    dx = np.zeros(2*nv)
    for ix in range(len(dx)):
        dx[ix] += h
        x_prime = np.hstack((pin.integrate(rmodel, x0[:nq], dx[:nv]), x0[nq:] + dx[nv:]))
        f_prime = f(x_prime)[0].copy()
        tmp = (f_prime - f0)/h
        Fx.append(tmp)
        dx[ix] = 0.0

    return np.array(Fx).T

# Numerical difference function with euclidian input to SE3 output
def numdiffEuclidiantoSE3(f,x0, rmodel, h=1e-6):
    f0 = f(x0)[0]
    Fx = []
    nq, nv = rmodel.nq, rmodel.nv
    dx = np.zeros_like(x0)
    for ix in range(len(dx)):
        dx[ix] += h
        f_prime = f(x0 + dx)[0].copy()
        tmp = np.hstack((pin.difference(rmodel, f0[:nq], f_prime[:nq]), f_prime[nq:] - f0[nq:]))
        Fx.append(tmp/h)
        dx[ix] = 0.0

    return np.array(Fx).T

# Numerical difference function on SE3
def numdiffSE3toSE3(f,x0, rmodel, h=1e-6):
    f0 = f(x0)[0]
    Fx = []
    nq, nv = rmodel.nq, rmodel.nv
    dx = np.zeros(2*nv)
    for ix in range(len(dx)):
        dx[ix] += h
        x_prime = np.hstack((pin.integrate(rmodel, x0[:nq], dx[:nv]), x0[nq:] + dx[nv:]))
        f_prime = f(x_prime)[0].copy()
        tmp = np.hstack((pin.difference(rmodel,f0[:nq], f_prime[:nq]), f_prime[nq:] - f0[nq:]))
        # print(f_prime[0:3], f0[0:3], x_prime[0:3], tmp[0:3])
        Fx.append(tmp/h)
        dx[ix] = 0.0
    return np.array(Fx).T

# Numerical difference function on SE3 variable and Euclidian output
def TorchnumdiffSE3toEuclidian(f,x0, rmodel, h=1e-6):
    f0 = f(x0)
    x0_numpy = x0.detach().numpy()
    Fx = []
    nq, nv = rmodel.nq, rmodel.nv
    dx = np.zeros(2*nv)
    for ix in range(len(dx)):
        dx[ix] += h
        x_prime = np.hstack((pin.integrate(rmodel, x0_numpy[:nq], dx[:nv]), x0_numpy[nq:] + dx[nv:]))
        f_prime = f(torch.tensor(x_prime))
        tmp = (f_prime - f0)/h
        Fx.append(tmp.detach().numpy())
        dx[ix] = 0.0

    return np.array(Fx).T

# Numerical difference function on SE3 variable and Euclidian output
def Torchnumdiff(f,x0,h=1e-6):
    f0 = f(x0).detach().numpy()
    x0_numpy = x0.detach().numpy().copy()
    x = x0_numpy
    Fx = []
    for ix in range(len(x)):
        x[ix] += h
        x_prime = torch.tensor(x.copy())
        f_prime = f(x_prime).detach().numpy()
        Fx.append((f_prime-f0)/h)
        x[ix] -= h
    return np.array(Fx).T
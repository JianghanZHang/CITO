# this file wraps all functions with torch

import torch
import numpy as np
import pinocchio as pin
from cube_model import *

from dynamics import TorchDiscreteDynamics
from frame_derivatives import TorchFrameTranslation
from force_derivatives import TorchLocalWorldAlignedForce
from numerical_differences import TorchnumdiffSE3toEuclidian, Torchnumdiff


dt = 1e-2
force = 100.0*torch.rand(3,  dtype=torch.float64)
force.requires_grad = True
p0 = 0.25*torch.rand(3, dtype=torch.float64)
p0[0] = 0.0
# print(force)
rot = np.pi*np.random.rand(3)
quat = pin.Quaternion(pin.utils.rpyToMatrix(rot[0], rot[1], rot[2]))

q0 = np.hstack((np.random.rand(3), np.array([quat[0], quat[1], quat[2], quat[3]])))
v0 = np.random.rand(rmodel.nv)
x0 = torch.hstack((torch.tensor(q0), torch.tensor(v0)))
x0.requires_grad = True
grf = np.array([0, 0, mass*9.81, 0, 0, 0]) # ground reaction forces

TorchDiscreteDynamics_ = TorchDiscreteDynamics.apply

def chain_func(x, force, fids, rmodel, rdata, dt, grf):
    force = TorchLocalWorldAlignedForce.apply(force, x, fids[0], rmodel, rdata)
    xout = TorchDiscreteDynamics_(x, force, fids, rmodel, rdata, dt, grf)
    loc = TorchFrameTranslation.apply(xout, fids[0], rmodel, rdata)
    loss = torch.linalg.norm(loc - torch.ones(3))
    return loss

arg1_func = lambda x : chain_func(x, force, fids, rmodel, rdata, dt, grf)
arg2_func = lambda F : chain_func(x0, F, fids, rmodel, rdata, dt, grf)

Jnumdiff_arg1 = TorchnumdiffSE3toEuclidian(arg1_func, x0, rmodel)
Jnumdiff_arg2 = Torchnumdiff(arg2_func, force)

Localforce = TorchLocalWorldAlignedForce.apply(force, x0, fids[0], rmodel, rdata)
xout = TorchDiscreteDynamics_(x0, Localforce, fids, rmodel, rdata, dt, grf)
loc = TorchFrameTranslation.apply(xout, fids[0], rmodel, rdata)
loss = torch.linalg.norm(loc - torch.ones(3))
loss.backward()

d1 = x0.grad.detach().numpy()
d2 = force.grad.detach().numpy()

assert np.linalg.norm(Jnumdiff_arg1[:rmodel.nv] - d1[:rmodel.nv]) < 1e-4
assert np.linalg.norm(Jnumdiff_arg1[rmodel.nv:] - d1[rmodel.nq:]) < 1e-4

print("unit test passed ...")
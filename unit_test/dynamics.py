## This class enforces soft robot dynamics constraints assuming the tau is in the joint space
## uses euler integration
## Author : Avadesh Meduri
## Date : 23/11/2022

import torch
from torch.autograd import Function
import numpy as np
import pinocchio as pin


def ContinousDynamicsDerivatives(x, localFrameForces, fids, rmodel, rdata, grf = np.zeros(6)):
    """
    This function computes the acceleration and their derivatives of a rigid body system
    Note : The forces are assumed to be in the local contact frame (frame defined with fid)
    Input:
        x : current state of the system (q,v)
        localFrameForces : forces applied in the local frame
        fids : frames ids where forces are applied    
        rmodel : pinocchio model
        rdata : pinocchio data
        grf : forces applied directly to the CoM
    """
    assert len(x) == rmodel.nq + rmodel.nv
    assert len(localFrameForces) == len(fids)
    q, v = x[:rmodel.nq], x[rmodel.nq:]
    JointFrameForces = [pin.Force(np.zeros(6)) for _ in range(rmodel.njoints)]
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateFramePlacements(rmodel, rdata)

    for idx, LocalFrameforce in enumerate(localFrameForces):
        jid = rmodel.frames[fids[idx]].parent # joint id of the parent frame
        jMf =  (rdata.oMi[jid].inverse()*rdata.oMf[fids[idx]]) # SE3 matrix from contact frame to parent joint frame
        LocalFrameforce = pin.Force(LocalFrameforce, np.zeros(3))
        JointFrameForces[jid] += jMf.act(LocalFrameforce) 

    a = pin.aba(rmodel, rdata, q, v, grf, JointFrameForces)

    #computing derivatives
    pin.updateGlobalPlacements(rmodel, rdata)
    pin.framesForwardKinematics(rmodel, rdata, q)
    da_dq, da_dv, _ = pin.computeABADerivatives(rmodel, rdata, q, v, grf, JointFrameForces)
    
    da_df = np.zeros((rmodel.nv,3*len(fids))) #derivatives wrt forces
    for idx, LocalFrameforce in enumerate(localFrameForces):
        ljac = pin.computeFrameJacobian(rmodel, rdata, q, fids[idx], pin.LOCAL)
        da_df[:,3*idx:3*(idx+1)] = (rdata.Minv @ ljac.T)[:,:3]

    da_dx = np.hstack((da_dq, da_dv))
    return a, da_dx, da_df


def DiscreteDynamicsDerivatives(x, localFrameForces, fids, rmodel, rdata, dt, grf = np.zeros(6)):
    """
    Function integrates a discretized system and provides its derivatives. (Semi Implicit Euler used)
    Note : The forces are assumed to be in the local contact frame (frame defined with fid)
    Input:
        x : current state of the system (q,v)
        localFrameForces : forces applied in the local frame
        fids : frames ids where forces are applied    
        rmodel : pinocchio model
        rdata : pinocchio data
        dt : discretization time step
        grf : forces applied directly to the CoM
    """
    nq, nv = rmodel.nq, rmodel.nv
    q, v = x[:nq], x[nq:nq+nv]
    
    a_next, da_dx, da_df = ContinousDynamicsDerivatives(x, localFrameForces, fids, rmodel, rdata, grf)

    # Forward Dynamics
    v_next = v + a_next * dt
    q_next = pin.integrate(rmodel, q, v_next * dt)

    # Discrete dynamics derivatives
    Fx = np.block([[da_dx*dt*dt],[da_dx*dt]])
    Fx[0:nv,nv:] += np.eye(nv)*dt
    Fu = np.block([[da_df *dt * dt],
                       [da_df *dt]])

    Fx[0:nv] = pin.dIntegrateTransport(rmodel, q, v_next * dt, Fx[0:nv], pin.ARG1)
    Fx[0:nv,0:nv] += pin.dIntegrate(rmodel, q, v_next * dt, pin.ARG0)
    Fx[nv:, nv:] += np.eye(nv)
    Fu[0:nv] = pin.dIntegrateTransport(rmodel, q, v_next * dt, Fu[0:nv], pin.ARG1)

    return np.hstack((q_next, v_next)), Fx, Fu


class TorchDiscreteDynamics(Function):
    
    @staticmethod
    def forward(ctx, x, localFrameForces, fids, rmodel, rdata, dt, grf = np.zeros(6)):
        
        localFrameForces = localFrameForces.detach().numpy()
        localFrameForcesList = [localFrameForces[3*i:3*i+3] for i in range(len(fids))]
        x =  x.detach().numpy()
        nq, nv = rmodel.nq, rmodel.nv
        xout, Fx, Fu = DiscreteDynamicsDerivatives(x, localFrameForcesList, fids, rmodel, rdata, dt, grf)

        ctx.Fx = torch.tensor(Fx)
        ctx.Fu = torch.tensor(Fu)

        ctx.nq, ctx.nv = nq, nv
        
        return torch.tensor(xout)
    
    @staticmethod
    def backward(ctx, grad):
                    
        Fu, Fx = ctx.Fu, ctx.Fx
        nq, nv = ctx.nq, ctx.nv
        nx = nq + nv
        if len(grad) == nx and nq != nv:
            grad = torch.hstack((grad[0:nv], grad[nq:]))
        
        Fu_grad = Fu.T@grad

        Fx_grad = Fx.T@grad
        tmp = torch.zeros((nx))
        tmp[0:nv] = Fx_grad[:nv]
        tmp[nq:] = Fx_grad[nv:]
        Fx_grad = tmp
        
        return Fx_grad, Fu_grad, None, None, None, None, None
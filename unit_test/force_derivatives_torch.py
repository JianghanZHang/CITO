## This files contains the forces and their derivatives when applied in particular frames. 

import torch
from torch.autograd import Function
import numpy as np
import pinocchio as pin

def FrameTranslationForce(Fb, aPb):
    """
    This function transfers a force applied at frame B to Frame A. 
    Frame A & B are assumed to have the same orientation but have a position offset    
    Input:
        Fb : Force applied in frame B
        aPb : 3d Vector of translation of frame B wrt frame A
    """
    assert len(Fb) == 3
    aMb = pin.SE3.Identity()
    aMb.translation = aPb
    Fb = pin.Force(Fb, np.zeros(3))
    Fa = aMb.act(Fb)
    JFb = aMb.actionInverse.T[:,:3] # derivative of Fa/Fb (action Inverse transpose)
    JaPb = np.zeros((6,3))
    Fx, Fy, Fz = Fb.linear[0],Fb.linear[1], Fb.linear[2]
    JaPb[3:] = np.array([[0.0, Fz, -Fy], 
                         [-Fz, 0.0, Fx],
                         [Fy, -Fx, 0.0]]) # derivatives of Fa/aPb
    return np.array(Fa), JFb, JaPb

def LocalWorldAlignedForce(Flw, x, fid, rmodel, rdata):
    """
    This function rotates a force in the local world aligned frame to the contact frame (local frame)
    Input:
        Flw : force in the local world aligned frame
        x : state of the system (q,v)
        fid : frame id of the local frame
        rmodel : pinocchio model
        rdata : pinocchio data
    """
    assert len(Flw)
    nq, nv = rmodel.nq, rmodel.nv

    q, v = x[:nq], x[nq:nq+nv]
    
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.computeAllTerms(rmodel, rdata, q, v)
    pin.updateFramePlacements(rmodel, rdata)

    # rotating the force into the contact frame
    oRl = rdata.oMf[fid].rotation # orientation of the local frame wrt world frame
    fc = pin.Force(oRl.T @ Flw.copy(), np.zeros(3))

    # derivatives
    ljac = pin.computeFrameJacobian(rmodel, rdata, q, fid, pin.LOCAL)
    dfc_dq = pin.skew(oRl.T @ Flw) @ ljac[3:]
    dfc_dx = np.hstack((dfc_dq, np.zeros((3, nv))))
    dfc_df = oRl.T

    return np.array(fc.linear), dfc_df, dfc_dx


class TorchFrameTranslationForce(Function):

    @staticmethod
    def forward(ctx, Fb, aPb):
        Fb = Fb.detach().numpy()
        aPb = aPb.detach().numpy()
        Fa, JFb, JaPb = FrameTranslationForce(Fb, aPb)

        ctx.JFb = torch.tensor(JFb)
        ctx.JaPb = torch.tensor(JaPb)

        return torch.tensor(Fa)
    
    @staticmethod
    def backward(ctx, grad):
        
        JFb, JaPb = ctx.JFb, ctx.JaPb

        dl_dFb = JFb.T @ grad    
        dl_dPb = JaPb.T @ grad
        
        return dl_dFb, dl_dPb


class TorchLocalWorldAlignedForce(Function):

    @staticmethod
    def forward(ctx, Flw, x, fid, rmodel, rdata):
        Flw = Flw.detach().numpy()
        x = x.detach().numpy()
        Fc, dfc_df, dfc_dx = LocalWorldAlignedForce(Flw, x, fid, rmodel, rdata)

        ctx.dfc_df = torch.tensor(dfc_df)
        ctx.dfc_dx = torch.tensor(dfc_dx)
        ctx.nq, ctx.nv = rmodel.nq, rmodel.nv


        return torch.tensor(Fc)
    
    @staticmethod
    def backward(ctx, grad):
        
        dfc_df, dfc_dx = ctx.dfc_df, ctx.dfc_dx
        nq, nv = ctx.nq, ctx.nv

        dl_df = dfc_df.T @ grad #dloss / df   
        dl_dx = dfc_dx.T @ grad #d loss / dx
        if nq != nv:
            dl_dx = torch.cat([dl_dx[:nv],  torch.zeros(nq - nv), dl_dx[nv:]])


        return dl_df, dl_dx, None, None, None
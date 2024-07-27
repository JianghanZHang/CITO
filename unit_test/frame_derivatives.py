## This file contains frame related derivatives
## Author : Avadesh Meduri
## Date : 27/05/2024
import pinocchio as pin
import numpy as np
import torch
from torch.autograd import Function

def FrameTranslation(x, fid, rmodel, rdata):
    """
    This function returns the location of the frame and its derivatives 
    Input:
        x : state of the system (q,v)
        fid : frame id         
        rmodel : pinocchio model
        rdata : pinocchio data
    """
    nq, nv = rmodel.nq, rmodel.nv
    q, v = x[:rmodel.nq], x[rmodel.nq:]
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateFramePlacements(rmodel, rdata)
    loc = np.array(rdata.oMf[fid].translation) #location of the frame

    # computing derivatives
    J = pin.computeFrameJacobian(rmodel, rdata, x[0:nq], fid, pin.ReferenceFrame.LOCAL)
    J_rot = np.matmul(rdata.oMf[fid].rotation,J[0:3])
    dloc_dx = np.hstack((J_rot, np.zeros((3, nv))))

    return loc, dloc_dx


def FramePlacementError(x, oMd, fid, rmodel, rdata):
    """
    This function returns the SE3 error of the frame wrt its desired and its derivatives 
    Input:
        x : state of the system (q,v)
        oMd : desired frame location
        fid : frame id         
        rmodel : pinocchio model
        rdata : pinocchio data
    """
    nq, nv = rmodel.nq, rmodel.nv
    q, v = x[:rmodel.nq], x[rmodel.nq:]
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateFramePlacements(rmodel, rdata)

    se3_error = pin.SE3(oMd).inverse() * rdata.oMf[fid]
    error = np.array(pin.log6(se3_error)) # log error between desired and actual
        
    # computing derivatives
    J_error = pin.Jlog6(se3_error)       
    J = pin.computeFrameJacobian(rmodel, rdata, x[0:nq], fid, pin.ReferenceFrame.LOCAL)
    derror_dq = J_error @ J
    derror_dx = np.hstack((derror_dq, np.zeros((6, nv))))

    return error, derror_dx

def JointPlacementError(x, oMd, jid, rmodel, rdata):
    """
    This function returns the SE3 error of the frame wrt its desired and its derivatives 
    Input:
        x : state of the system (q,v)
        oMd : desired frame location
        jid : joint id         
        rmodel : pinocchio model
        rdata : pinocchio data
    """
    nq, nv = rmodel.nq, rmodel.nv
    q, v = x[:rmodel.nq], x[rmodel.nq:]
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateFramePlacements(rmodel, rdata)

    se3_error = pin.SE3(oMd).inverse() * rdata.oMi[jid]
    error = np.array(pin.log6(se3_error)) # log error between desired and actual
    # computing derivatives
    J_error = pin.Jlog6(se3_error)       
    J = pin.computeJointJacobian(rmodel, rdata, x[0:nq], jid)
    derror_dq = J_error @ J
    derror_dx = np.hstack((derror_dq, np.zeros((6, nv))))

    return error, derror_dx

def FrameVelocity(x, fid, rmodel, rdata):
    """
    This function returns the SE3 error of the frame wrt its desired and its derivatives 
    Input:
        x : state of the system (q,v)
        oMd : desired frame location
        fid : frame id         
        rmodel : pinocchio model
        rdata : pinocchio data
    """
    nq, nv = rmodel.nq, rmodel.nv
    q, v = x[:rmodel.nq], x[rmodel.nq:]
    pin.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, np.zeros(nv))
    pin.updateFramePlacements(rmodel, rdata)
    
    vel = pin.getFrameVelocity(rmodel, rdata, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    dvel_dq, dvel_dv = pin.getFrameVelocityDerivatives(rmodel,rdata,fid,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)        
    # computing derivatives

    return vel, dvel_dq, dvel_dv


class TorchFrameTranslation(Function):

    @staticmethod
    def forward(ctx, x, fid, rmodel, rdata):
        x = x.detach().numpy()
        loc, dloc_dx = FrameTranslation(x, fid, rmodel, rdata)

        ctx.dloc_dx = torch.tensor(dloc_dx)
        ctx.nq, ctx.nv = rmodel.nq, rmodel.nv

        return torch.tensor(loc)

    @staticmethod
    def backward(ctx, grad):

        dloc_dx = ctx.dloc_dx
        nq, nv = ctx.nq, ctx.nv
        dl_dx = dloc_dx.T @ grad
        if nq != nv:
            dl_dx = torch.cat([dl_dx[:nv], torch.zeros(nq - nv), dl_dx[nv:]])
        return dl_dx, None, None, None
    
class TorchFramePlacementError(Function):

    @staticmethod
    def forward(ctx, x, oMd, fid, rmodel, rdata):
        x = x.detach().numpy()
        error, derror_dx = FramePlacementError(x, oMd, fid, rmodel, rdata)

        ctx.derror_dx = torch.tensor(derror_dx)
        ctx.nq, ctx.nv = rmodel.nq, rmodel.nv

        return torch.tensor(error)

    @staticmethod
    def backward(ctx, grad):

        derror_dx = ctx.derror_dx
        nq, nv = ctx.nq, ctx.nv
        dl_dx = derror_dx.T @ grad
        if nq != nv:
            dl_dx = torch.cat([dl_dx[:nv],  torch.zeros(nq - nv), dl_dx[nv:]])
            
        return dl_dx, None, None, None, None
    
class TorchJointPlacementError(Function):

    @staticmethod
    def forward(ctx, x, oMd, jid, rmodel, rdata):
        x = x.detach().numpy()
        error, derror_dx = JointPlacementError(x, oMd, jid, rmodel, rdata)

        ctx.derror_dx = torch.tensor(derror_dx)
        ctx.nq, ctx.nv = rmodel.nq, rmodel.nv

        return torch.tensor(error)

    @staticmethod
    def backward(ctx, grad):

        derror_dx = ctx.derror_dx
        nq, nv = ctx.nq, ctx.nv
        dl_dx = derror_dx.T @ grad
        if nq != nv:
            dl_dx = torch.cat([dl_dx[:nv],  torch.zeros(nq - nv), dl_dx[nv:]])
            
        return dl_dx, None, None, None, None

class TorchFrameVelocity(Function):

    @staticmethod
    def forward(ctx, x, fid, rmodel, rdata):
        x = x.detach().numpy()
        vel, dvel_dq, dvel_dq = FrameVelocity(x, fid, rmodel, rdata)

        ctx.der = torch.hstack((torch.tensor(dvel_dq), torch.tensor(dvel_dq)))

        ctx.nq, ctx.nv = rmodel.nq, rmodel.nv

        return torch.tensor(vel.np)

    @staticmethod
    def backward(ctx, grad):

        dvel_dx = ctx.der
        nq, nv = ctx.nq, ctx.nv
        dl_dx = dvel_dx.T @ grad
        if nq != nv:
            dl_dx = torch.cat([dl_dx[:nv], torch.zeros(nq - nv), dl_dx[nv:]])
        return dl_dx, None, None, None
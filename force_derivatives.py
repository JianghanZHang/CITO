## Author : Avadesh Meduri
## This files contains the forces and their derivatives when applied in particular frames. 

import numpy as np
import pinocchio as pin

def FrameTranslationForceDerivatives(Fb, aPb):
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

def LocalWorldAlignedForceDerivatives(Flw, x, fid, rmodel, rdata):
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

def ForceDerivatives(Flw, x, fid, rmodel, rdata):
    """
    This function computes the force and derivatives in the local world aligned frame
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

    flw = pin.Force(Flw.copy(), np.zeros(3))

    # derivatives
    dflw_dx = np.zeros((3, 2*nv))

    return np.array(flw.linear), dflw_dx


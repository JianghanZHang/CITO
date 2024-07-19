## This class computes the Joint Kinematic Hessian in the centered frame swith Autograd
## Author : Jianghan Zhang


import autograd.numpy as np
from autograd import jacobian
import pinocchio as pin


class JointKinematicHessian:
    def __init__(self, rmodel, rdata, fid):
        self.rmodel = rmodel
        self.rdata = rdata
        self.fid = fid

    def compute_centered_jacobian(self, q):
        pin.framesForwardKinematics(self.rmodel, self.rdata, q)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        J1 = pin.computeFrameJacobian(self.rmodel, self.rdata, q, self.fid, pin.ReferenceFrame.LOCAL)
        J1_rot = np.matmul(self.rdata.oMf[self.fid].rotation, J1[0:3, :])
        return J1_rot
    
    def getJointKinematicHessian(self, q):
        Hessian = jacobian(self.compute_centered_jacobian)
        return Hessian(q)

import crocoddyl
import numpy as np
import pinocchio as pin
from robot_env import GO2_FOOT_RADIUS
    
class ResidualModelFrameTranslationNormal(crocoddyl.ResidualModelAbstract):

    def __init__(self, state, nu, fid):
        crocoddyl.ResidualModelAbstract.__init__(
            self, state, 1, nu, True, False, False
        )
        self.fid = fid

    def calc(self, data, x, u = None):
        nq, nv = self.state.nq, self.state.nv
        ndx = self.state.ndx
        q, v = x[:nq], x[-nv:]
        rmodel, rdata = self.state.pinocchio, data.pinocchio

        pin.forwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        
        Dist = np.array([rdata.oMf[self.fid].translation[2]])
    
        # Update the residual
        data.r = Dist        
   
    def calcDiff(self, data, x, u=None):
        # self.calc(data, x, u)
        
        nq, nv = self.state.nq, self.state.nv
        q, v = x[:nq], x[-nv:]
        rmodel, rdata = self.state.pinocchio, data.pinocchio
        
        J = pin.computeFrameJacobian(rmodel, rdata, q, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED) # Extract the Jacobian
        # J1 = rdata.oMf[self.fid].rotation @ J[:3,: ]
        
        dDist_dx = np.hstack((J[2, :], np.zeros((nv)))) # Extract the Z components
        # # Compute the derivative and zero out the x and y components
        data.Rx = dDist_dx
        # import pdb; pdb.set_trace()
    def createData(self, data):
        data = crocoddyl.ResidualDataAbstract(self, data)
        data.pinocchio = pin.Data(self.state.pinocchio)
        return data
    
class ResidualModelFrameVelocityTangential(crocoddyl.ResidualModelAbstract):
     
    def __init__(self, state, nu, fid):
        crocoddyl.ResidualModelAbstract.__init__(
            self, state, 2, nu, True, True, False
        )
        self.fid = fid

    def calc(self, data, x, u = None):
        nq, nv = self.state.nq, self.state.nv
        q, v = x[: self.state.nq], x[-self.state.nv :]
        rmodel, rdata = self.state.pinocchio, data.pinocchio
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)
        velocity = pin.getFrameVelocity(rmodel, rdata, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        data.r = velocity[:2]
           
    def calcDiff(self, data, x, u=None):
        nq, nv = self.state.nq, self.state.nv
        q, v = x[: self.state.nq], x[-self.state.nv :]
        rmodel, rdata = self.state.pinocchio, data.pinocchio
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)

        pin.computeJointJacobians(rmodel, rdata, q)
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, np.zeros(nv))

        dV_dx_full = pin.getFrameVelocityDerivatives(rmodel, rdata, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # print(f'dV_dx_full:\n {dV_dx_full}')
        # import pdb; pdb.set_trace()
        dV_dq = np.array(dV_dx_full[0][0:3, :])
        dV_dv = np.array(dV_dx_full[1][0:3, :])
        dV_dx = np.hstack((dV_dq, dV_dv))

        data.Rx = dV_dx[:2, :]


    def createData(self, data):
        data = crocoddyl.ResidualDataAbstract(self, data)
        data.pinocchio = pin.Data(self.state.pinocchio)
        return data

'''
This class compute the foot clearance residual:
    r = 1 / (1 + exp(-sigmoid_steepness * dist))* velocity_tan^T @ velocity_tan 
'''    
class ResidualModelFootClearance(crocoddyl.ResidualModelAbstract):

    def __init__(self, state, nu, fid, sigmoid_steepness=-30):
        crocoddyl.ResidualModelAbstract.__init__(
            self, state, 1, nu, True, True, False
        )
        self.sigmoid_steepness = sigmoid_steepness
        self.fid = fid

    def calc(self, data, x, u = None):
        nq, nv = self.state.nq, self.state.nv
        q, v = x[: nq], x[-nv :]
        rmodel, rdata = self.state.pinocchio, data.pinocchio

        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)
        velocity = pin.getFrameVelocity(rmodel, rdata, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        velocity_tan = velocity[:2]
        term2 = velocity_tan.T @ velocity_tan        
        dist = rdata.oMf[self.fid].translation[2] - GO2_FOOT_RADIUS
        term1 = 1 / (1 + np.exp(-self.sigmoid_steepness * dist))
        # import pdb; pdb.set_trace()
        data.r = np.array([term1 * term2])
        # print(f'data.r: {data.r}')
    def calcDiff(self, data, x, u=None):
        nq, nv = self.state.nq, self.state.nv
        q, v = x[: nq], x[-nv :]
        rmodel, rdata = self.state.pinocchio, data.pinocchio
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)
        velocity = pin.getFrameVelocity(rmodel, rdata, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        velocity_tan = velocity[:2]
        term2 = velocity_tan.T @ velocity_tan        
        
        dist = rdata.oMf[self.fid].translation[2] - GO2_FOOT_RADIUS
        term1 = 1 / (1 + np.exp(-self.sigmoid_steepness * dist))
        dterm1_dDist = self.sigmoid_steepness * term1 * (1 - term1) 
        J = pin.computeFrameJacobian(rmodel, rdata, q, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED) # Extract the Jacobian        
        dDist_dx = np.hstack((J[2, :], np.zeros((nv)))) # Extract the Z components

        # import pdb; pdb.set_trace() 
        dterm1_dx = dterm1_dDist * dDist_dx

        pin.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, np.zeros(nv))
        dV_dx_full = pin.getFrameVelocityDerivatives(rmodel, rdata, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        dVtan_dq = np.array(dV_dx_full[0][0:2, :])
        dVtan_dv = np.array(dV_dx_full[1][0:2, :])
        dVtan_dx = np.hstack((dVtan_dq, dVtan_dv))

        dterm2_dVtan = 2 * velocity_tan
        dterm2_dx = dterm2_dVtan @ dVtan_dx

        data.Rx = np.array([dterm1_dx * term2 + term1 * dterm2_dx])

    def createData(self, data):
        data = crocoddyl.ResidualDataAbstract(self, data)
        data.pinocchio = pin.Data(self.state.pinocchio)
        return data
    
class ResidualModelFootSlipping(crocoddyl.ResidualModelAbstract):

    def __init__(self, state, nu, fid, eps=1e-8):
        crocoddyl.ResidualModelAbstract.__init__(
            self, state, 1, nu, True, True, False
        )
        self.fid = fid
        self.eps = eps

    def calc(self, data, x, u = None):
        nq, nv = self.state.nq, self.state.nv
        q, v = x[: nq], x[-nv :]
        rmodel, rdata = self.state.pinocchio, data.pinocchio

        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)
        velocity = pin.getFrameVelocity(rmodel, rdata, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        velocity_tan = velocity[:2]
        # Vt_norm = np.linalg.norm(velocity_tan, 2)
        Vt_norm = np.sqrt(np.sum(velocity_tan**2) + self.eps**2)  # Smoothed norm
        Dist = rdata.oMf[self.fid].translation[2] - GO2_FOOT_RADIUS
        data.r = np.array([Dist * Vt_norm])
        
    def calcDiff(self, data, x, u=None):
        nq, nv = self.state.nq, self.state.nv
        q, v = x[: nq], x[-nv :]
        rmodel, rdata = self.state.pinocchio, data.pinocchio
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)
        velocity = pin.getFrameVelocity(rmodel, rdata, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        velocity_tan = velocity[:2]
        # Vt_norm = np.linalg.norm(velocity_tan, 2)
        Vt_norm = np.sqrt(np.sum(velocity_tan**2) + self.eps**2)  # Smoothed norm
        Dist = rdata.oMf[self.fid].translation[2] - GO2_FOOT_RADIUS

        
        J = pin.computeFrameJacobian(rmodel, rdata, q, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED) # Extract the Jacobian        
        dDist_dx = np.hstack((J[2, :], np.zeros((nv)))) # Extract the Z components

        pin.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, np.zeros(nv))
        dV_dx_full = pin.getFrameVelocityDerivatives(rmodel, rdata, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        dVt_dq = np.array(dV_dx_full[0][0:2, :])
        dVt_dv = np.array(dV_dx_full[1][0:2, :])
        dVt_dx = np.hstack((dVt_dq, dVt_dv))

        dVt_norm_dVt = velocity_tan / Vt_norm

        data.Rx = np.array([dDist_dx * Vt_norm + Dist * dVt_norm_dVt @ dVt_dx])



    def createData(self, data):
        data = crocoddyl.ResidualDataAbstract(self, data)
        data.pinocchio = pin.Data(self.state.pinocchio)
        return data
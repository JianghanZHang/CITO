
import crocoddyl
import numpy as np
import pinocchio as pin

class ResidualModelFrameTranslationError(crocoddyl.ResidualModelAbstract):

    def __init__(self, state, nu, fid1, fid2):

        crocoddyl.ResidualModelAbstract.__init__(
            self, state, 3, nu, True, False, False
        )

        self.fid1 = fid1
        self.fid2 = fid2

    def calc(self, data, x, u):

        nq, nv = self.state.nq, self.state.nv
        q, v = x[: nq], x[-nv :]
        rmodel, rdata = self.state.pinocchio, data.pinocchio

        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        data.r = np.array(rdata.oMf[self.fid1].translation) - np.array(rdata.oMf[self.fid2].translation) #location of the frame
        # computing derivatives
        J = pin.computeFrameJacobian(rmodel, rdata, x[0:nq], self.fid1, pin.ReferenceFrame.LOCAL)
        J_rot = np.matmul(rdata.oMf[self.fid1].rotation,J[0:3])
        dloc1_dx = np.hstack((J_rot, np.zeros((3, nv))))

        J = pin.computeFrameJacobian(rmodel, rdata, x[0:nq], self.fid2, pin.ReferenceFrame.LOCAL)
        J_rot = np.matmul(rdata.oMf[self.fid2].rotation,J[0:3])
        dloc2_dx = np.hstack((J_rot, np.zeros((3, nv))))

        self.der = dloc1_dx - dloc2_dx

    def calcDiff(self, data, x, u):
        self.calc(data, x, u)

        data.Rx = self.der

    def createData(self, data):
        data = crocoddyl.ResidualDataAbstract(self, data)
        data.pinocchio = pin.Data(self.state.pinocchio)

        return data
    
class ResidualModelFrameTranslationNormal(crocoddyl.ResidualModelAbstract):

    def __init__(self, state, nu, fid1):
        crocoddyl.ResidualModelAbstract.__init__(
            self, state, 1, nu, True, False, False
        )
        self.fid1 = fid1

    def calc(self, data, x, u):
        nq, nv = self.state.nq, self.state.nv
        ndx = self.state.ndx
        q, v = x[:nq], x[-nv:]
        rmodel, rdata = self.state.pinocchio, data.pinocchio

        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        
        # Compute the translation error
        Dist = np.array(rdata.oMf[self.fid1].translation)[2]
    
        # Update the residual
        data.r = Dist

        # Computing derivatives
        J1 = pin.computeFrameJacobian(rmodel, rdata, q, self.fid1, pin.ReferenceFrame.LOCAL)

        J1_rot = np.matmul(rdata.oMf[self.fid1].rotation, J1[0:3, :]) # Extract the linear part of the Jacobian and rotate it to the world frame
        
        dDist_dx = np.hstack((J1_rot[2], np.zeros((1, nv)))) # Extract the Z components

        # Compute the derivative and zero out the x and y components
        der = dDist_dx
        self.der = der

    def calcDiff(self, data, x, u):
        self.calc(data, x, u)
        data.Rx = self.der

    def createData(self, data):
        data = crocoddyl.ResidualDataAbstract(self, data)
        data.pinocchio = pin.Data(self.state.pinocchio)
        return data

class ResidualModelComplementarityErrorNormal(crocoddyl.ResidualModelAbstract):

    def __init__(self, state, nu, fid, idx, nq_j):
        """
        Creates complementarity constraints between a contact frame and end effector frame of robot
        Input:
            state : crocodyl multibody state
            nu : size of control vector
            robotFid : FrameId of the robot end effector/ contact frame
            cntFid : FrameId of the contact frame
            idx : index of contact frame (needed to segment the control)
            rnv : number of velocity for robot 
        """

        crocoddyl.ResidualModelAbstract.__init__(
            self, state, 1, nu, True, False, True
        )

        self.fid = fid
        self.idx = idx
        self.nq_j = nq_j

    def calc(self, data, x, u):

        nq, nv = self.state.nq, self.state.nv
        q, v = x[: nq], x[-nv :]
        fid, idx = self.fid, self.idx
        rmodel, rdata = self.state.pinocchio, data.pinocchio

        # Extracting 3d contact force on idx end-effector
        force = u[self.nq_j+3*idx : self.nq_j+3*(idx+1)]
        force_normal = force[2]

        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        
        Dist = np.array(rdata.oMf[fid].translation)[2]
        data.r = Dist * force_normal
        # print(data.r, force)
        # print(data.r, np.linalg.norm(dist), np.linalg.norm(force))
        # computing derivatives
        # Computing derivatives
        J1 = pin.computeFrameJacobian(rmodel, rdata, q, self.fid1, pin.ReferenceFrame.LOCAL)

        J1_rot = np.matmul(rdata.oMf[self.fid1].rotation, J1[0:3, :]) # Extract the linear part of the Jacobian and rotate it to the world frame
        
        dDist_dx = np.hstack((J1_rot[2], np.zeros((1, nv)))) # Extract the Z components

        dr_dDist = force_normal

        self.dr_dx = dr_dDist @ dDist_dx
        self.dr_du = np.zeros((1,len(u)))
        self.dr_du[self.nq_j+3*idx+2] = Dist

    def calcDiff(self, data, x, u):
        self.calc(data, x, u)
        data.Rx = self.dr_dx
        data.Ru = self.dr_du

    def createData(self, data):
        data = crocoddyl.ResidualDataAbstract(self, data)
        data.pinocchio = pin.Data(self.state.pinocchio)

        return data
    
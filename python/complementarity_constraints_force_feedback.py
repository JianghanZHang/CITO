
import crocoddyl
import numpy as np
import pinocchio as pin
from robots.robot_env import create_solo12_env_force_feedback

# Create the robot
env = create_solo12_env_force_feedback()
rmodel = env['rmodel']
rdata = rmodel.createData()
    
class ResidualModelFrameTranslationNormal(crocoddyl.ResidualModelAbstract):

    def __init__(self, state, nu, fid, nf):
        crocoddyl.ResidualModelAbstract.__init__(
            self, state, 1, nu, True, False, False
        )
        self.fid = fid
        self.nf = nf

    def calc(self, data, x, u = None):
        nq, nv = self.state.nq, self.state.nv
        nf = self.nf
        ndx = self.state.ndx
        q, v, f = x[:nq], x[nq:nq + nv], x[-nf:]
        rmodel, rdata = self.state.pinocchio, data.pinocchio

        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        
        # Compute the translation error
        # Dist = np.array(rdata.oMf[self.fid].translation)[2]
        Dist = np.array([rdata.oMf[self.fid].translation[2]])
    
        # Update the residual
        data.r = Dist

        # Computing derivatives
        J1 = pin.computeFrameJacobian(rmodel, rdata, q, self.fid, pin.ReferenceFrame.LOCAL)

        J1_rot = np.matmul(rdata.oMf[self.fid].rotation, J1[0:3, :]) # Extract the linear part of the Jacobian and rotate it to the world frame
        
        dDist_dx = np.hstack((J1_rot[2], np.zeros((nv), np.zeros(nf)))) # Extract the Z components

        # Compute the derivative and zero out the x and y components
        der = dDist_dx
        self.der = der
   
    def calcDiff(self, data, x, u=None):
        self.calc(data, x, u)
        data.Rx = self.der

    def createData(self, data):
        data = crocoddyl.ResidualDataAbstract(self, data)
        data.pinocchio = pin.Data(self.state.pinocchio)
        return data
    
class ResidualModelContactForceNormal(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu, fid, idx, nq_j, nf, K, B):
        crocoddyl.ResidualModelAbstract.__init__(
            self, state, 1, nu, True, False, False
        )
        self.fid = fid
        self.idx = idx
        self.nq_j = nq_j
        self.nf = nf
        self.stiffness = K
        self.damping = B
    
    def calc(self, data, x, u):
        nq, nv = self.state.nq, self.state.nv
        nf = self.nf
        q, v, f = x[:nq], x[nq:nq + nv], x[-nf:]

        rmodel, rdata = self.state.pinocchio, data.pinocchio

        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        
        data.r = np.array([x[nq+nv+3*self.idx+2]])  # Ensure this is a 1D array
        self.dr_dx = np.zeros((len(x),))

        J1 = pin.computeFrameJacobian(rmodel, rdata, q, self.fid, pin.ReferenceFrame.LOCAL)
        J1_rot = np.matmul(rdata.oMf[self.fid].rotation, J1[0:3, :]) # Extract the linear part of the Jacobian and rotate it to the world frame
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, np.zeros(nv))
        dV_dx_full = pin.getFrameVelocityDerivatives(rmodel, rdata, self.fid, pin.ReferenceFrame.WORLD)       

        dV_dq = np.array(dV_dx_full[0][2, :])
        dV_dv = np.array(dV_dx_full[1][2, :])

        self.dr_dx[:nq] = -self.stiffness * J1_rot[2] - self.damping * dV_dq                        # dFz/dq
        self.dr_dx[nq:nq+nv] = -self.damping * dV_dv                                                # dFz/dv
        self.dr_dx[nq+nv+3*self.idx+2] = 1                                                          # dFz/df
     
    def calcDiff(self, data, x, u):
        self.calc(data, x, u)
        data.Rx = self.dr_dx

    def createData(self, data):
        data = crocoddyl.ResidualDataAbstract(self, data)
        data.pinocchio = pin.Data(self.state.pinocchio)
        return data


class ResidualModelComplementarityErrorNormal(crocoddyl.ResidualModelAbstract):

    def __init__(self, state, nu, fid, idx, nq_j, nf, K, B):
        """
        Creates complementarity constraints between a contact frame and end effector frame of robot
        Input:
            state : crocodyl multibody state
            nu : size of control vector
            idx : index of contact frame (needed to segment the control)
            rnv : number of velocity for robot 
        """

        crocoddyl.ResidualModelAbstract.__init__(
            self, state, 1, nu, True, False, False
        )

        self.fid = fid
        self.idx = idx
        self.nq_j = nq_j
        self.nf = nf
        self.stiffness = K
        self.damping = B
    
    def calc(self, data, x, u):
        nq, nv = self.state.nq, self.state.nv
        nf = self.nf
        q, v, f = x[:nq], x[nq:nq + nv], x[-nf:]

        fid, idx = self.fid, self.idx
        rmodel, rdata = self.state.pinocchio, data.pinocchio

        # Extracting 3d contact force on idx end-effector
        force = f[3 * idx + 2]
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        
        Dist = np.array(rdata.oMf[fid].translation)[2]
        # import pdb; pdb.set_trace()
        data.r = np.array([Dist * force])
        
        # computing derivatives
        J1 = pin.computeFrameJacobian(rmodel, rdata, q, self.fid, pin.ReferenceFrame.LOCAL)
        J1_rot = np.matmul(rdata.oMf[self.fid].rotation, J1[0:3, :]) # Extract the linear part of the Jacobian and rotate it to the world frame
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, np.zeros(nv))
        dV_dx_full = pin.getFrameVelocityDerivatives(rmodel, rdata, self.fid, pin.ReferenceFrame.WORLD)       

        dV_dq = np.array(dV_dx_full[0][2, :])
        dV_dv = np.array(dV_dx_full[1][2, :])

        # import pdb; pdb.set_trace()

        self.dr_dx[:nq] = J1_rot[2] * force + Dist * (-self.stiffness * J1_rot[2] - self.damping * dV_dq) # dr/dq
        self.dr_dx[nq:nq+nv] = Dist * (-self.damping * dV_dv)                                             # dr/dv
        self.dr_dx[nq+nv+3*idx+2] = Dist                                                                  # dr/df
        

    def calcDiff(self, data, x, u):
        self.calc(data, x, u)
        data.Rx = self.dr_dx

    def createData(self, data):
        data = crocoddyl.ResidualDataAbstract(self, data)
        data.pinocchio = pin.Data(self.state.pinocchio)

        return data
    

class ResidualModelComplementarityErrorTangential(crocoddyl.ResidualModelAbstract):

    def __init__(self, state, nu, fid, idx, nq_j):
        """
        Creates complementarity constraints between a contact frame and end effector frame of robot
        Input:
            state : crocodyl multibody state
            nu : size of control vector
            idx : index of contact frame (needed to segment the control)
            rnv : number of velocity for robot 
        """

        crocoddyl.ResidualModelAbstract.__init__(
            self, state, 4, nu, True, False, True
        )

        self.fid = fid
        self.idx = idx
        self.nq_j = nq_j
        # self.numDiff = JointKinematicHessian(rmodel, rdata, fid)

    def calc(self, data, x, u):
        nq, nv = self.state.nq, self.state.nv
        q, v = x[: nq], x[-nv :]
        fid, idx = self.fid, self.idx
        rmodel, rdata = self.state.pinocchio, data.pinocchio

        # Extracting 3d contact force on idx end-effector
        force = u[self.nq_j+3*idx : self.nq_j+3*(idx+1)]
        force_x, force_y = force[0], force[1]

        # pin.framesForwardKinematics(rmodel, rdata, q)
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)

        velocity = pin.getFrameVelocity(rmodel, rdata, fid, pin.ReferenceFrame.WORLD).linear[:2]

        data.r = np.array([velocity[0]* force_x, velocity[1]*force_x, velocity[0]*force_y, velocity[1]*force_y])
        
        # computing derivatives
        # Jl = pin.computeFrameJacobian(rmodel, rdata, q, self.fid, pin.ReferenceFrame.LOCAL)

        # J = np.matmul(rdata.oMf[fid].rotation, Jl[0:3, :]) # Extract the linear part of the Jacobian and rotate it to the world frame
        
        # tmp1 = np.array([dJ_dq[0]@v.T, dJ_dq[1]@v.T, dJ_dq[0]@v.T, dJ_dq[1]@v.T])

        # ddq_dx = np.array([np.zeros((nv, nv)), np.eye(nv)])
        # tmp2 = np.array([J[0:2, :] @ ddq_dx, J[0:2, :] @ ddq_dx]) 

        # dV_dx = tmp1 + tmp2
        pin.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, np.zeros(nv))
        dV_dx_full = pin.getFrameVelocityDerivatives(rmodel, rdata, fid, pin.ReferenceFrame.WORLD)       

        dV_dq = np.array(dV_dx_full[0][0:2, :])
        dV_dv = np.array(dV_dx_full[1][0:2, :])

        dV_dx = np.hstack((dV_dq, dV_dv))
        dr_dV = np.vstack((force_x*np.eye(2), force_y*np.eye(2)))
        # import pdb; pdb.set_trace()

        self.dr_dx = dr_dV @ dV_dx

        self.dr_du = np.zeros((4, len(u)))
        self.dr_du[0, self.nq_j+3*idx] = velocity[0]
        self.dr_du[1, self.nq_j+3*idx] = velocity[1]
        self.dr_du[2, self.nq_j+3*idx+1] = velocity[0]
        self.dr_du[3, self.nq_j+3*idx+1] = velocity[1]        

        # print(f"Complementarity Constraints tangential:")
        # print(f"idx:{idx}")
        # print(f'confact frame ID:{fid}')
        # print(f'force:{force}' )
        # print(f"velocity:{pin.getFrameVelocity(rmodel, rdata, fid, pin.ReferenceFrame.WORLD)}")
        # print(f"Residual:{data.r}")
        # print(f'drdu:{self.dr_du}')
        # print(f'drdV:{dr_dV}')
        # print(f"dVdx:{dV_dx_full}")
        # print(f'drdx:{self.dr_dx}')
        # print(f'confact frame:{rdata.oMf[fid]}')

    def calcDiff(self, data, x, u):
        self.calc(data, x, u)
        data.Rx = self.dr_dx
        data.Ru = self.dr_du

    def createData(self, data):
        data = crocoddyl.ResidualDataAbstract(self, data)
        data.pinocchio = pin.Data(self.state.pinocchio)

        return data
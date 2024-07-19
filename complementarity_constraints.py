
import crocoddyl
import numpy as np
import pinocchio as pin

    
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
        
        dDist_dx = np.hstack((J1_rot[2], np.zeros((nv)))) # Extract the Z components

        # Compute the derivative and zero out the x and y components
        der = dDist_dx
        self.der = der
        # print('Z constraint:')
        # print(f'confact frame ID:{self.fid}')
        # print(f'forces:{u[12:]}' )
        # print(f"Dist:{Dist}")
        # print(f"Residual:{data.r}")
        # print(f"dr_dx:{self.der}")
    def calcDiff(self, data, x, u=None):
        self.calc(data, x, u)
        data.Rx = self.der

    def createData(self, data):
        data = crocoddyl.ResidualDataAbstract(self, data)
        data.pinocchio = pin.Data(self.state.pinocchio)
        return data
    
# class ResidualModelContactForceNormal(crocoddyl.ResidualModelAbstract):
#     def __init__(self, state, nu, fid, idx, nq_j):
#         crocoddyl.ResidualModelAbstract.__init__(
#             self, state, 1, nu, False, False, True
#         )
#         self.fid = fid
#         self.idx = idx
#         self.nq_j = nq_j
    
#     def calc(self, data, x, u):
#         nq, nv = self.state.nq, self.state.nv
#         q, v = x[:nq], x[-nv:]
#         rmodel, rdata = self.state.pinocchio, data.pinocchio

#         pin.framesForwardKinematics(rmodel, rdata, q)
#         pin.updateFramePlacements(rmodel, rdata)
        
#         data.r = np.array([u[self.nq_j + 3*self.idx + 2]])  # Ensure this is a 1D array
#         self.dr_du = np.zeros((len(u),))
#         self.dr_du[self.nq_j + 3*self.idx + 2] = 1
#         # print(f"Force Constraints:")
#         # print(f"idx:{self.idx}")
#         # print(f'confact frame ID:{self.fid}')
#         # print(f'force:{u[self.nq_j+3*self.idx : self.nq_j+3*(self.idx+1)]}' )
#         # print(f"Dist:{np.array(rdata.oMf[self.fid].translation)[2]}")
#         # print(f"Residual:{data.r}")
#         # print(f"dr_du:{self.dr_du}")

#     def calcDiff(self, data, x, u):
#         self.calc(data, x, u)
#         data.Ru = self.dr_du

#     def createData(self, data):
#         data = crocoddyl.ResidualDataAbstract(self, data)
#         data.pinocchio = pin.Data(self.state.pinocchio)
#         return data


class ResidualModelComplementarityErrorNormal(crocoddyl.ResidualModelAbstract):

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
            self, state, 3, nu, True, False, True
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

        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        
        Dist = np.array(rdata.oMf[fid].translation)[2]
        # import pdb; pdb.set_trace()
        data.r = np.array(Dist * force)
        
        # computing derivatives
        J1 = pin.computeFrameJacobian(rmodel, rdata, q, self.fid, pin.ReferenceFrame.LOCAL)

        J1_rot = np.matmul(rdata.oMf[fid].rotation, J1[0:3, :]) # Extract the linear part of the Jacobian and rotate it to the world frame
        
        dDist_dx = np.hstack((J1_rot, np.zeros((3, nv)))) 

        dr_dDist = np.array([force])
        self.dr_dx = dr_dDist @ dDist_dx
        self.dr_du = np.zeros((3, len(u)))
        self.dr_du[:, self.nq_j+3*idx : self.nq_j + 3*(idx+1)] = Dist
        

        # print(f"Complementarity Constraints:")
        # print(f"idx:{idx}")
        # print(f'confact frame ID:{fid}')
        # print(f'force:{force}' )
        # print(f"Dist:{Dist}")
        # print(f"Residual:{np.array([Dist * force])}")
        # print(f'drdu:{self.dr_du}')
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

    def calc(self, data, x, u):
        nq, nv = self.state.nq, self.state.nv
        q, v = x[: nq], x[-nv :]
        fid, idx = self.fid, self.idx
        rmodel, rdata = self.state.pinocchio, data.pinocchio

        # Extracting 3d contact force on idx end-effector
        force = u[self.nq_j+3*idx : self.nq_j+3*(idx+1)]
        force_x, force_y = force[0], force[1]

        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        
        pin.forwardKinematics(rmodel, rdata, q, v)
        velocity = pin.getFrameVelocity(rmodel, rdata, fid, pin.ReferenceFrame.WORLD).linear[:2]

        
        data.r = np.array([velocity*force_x, velocity*force_y])
        
        # computing derivatives
        J1 = pin.computeFrameJacobian(rmodel, rdata, q, self.fid, pin.ReferenceFrame.LOCAL)

        J1_rot = np.matmul(rdata.oMf[fid].rotation, J1[0:3, :]) # Extract the linear part of the Jacobian and rotate it to the world frame
        
        dDist_dx = np.hstack((J1_rot, np.zeros((3, nv)))) 

        ddq_dx = np.array([np.zeros((nv, nv)), np.eye(nv)])
        dV_dx = np.array([J1_rot[0:2, :] @ ddq_dx, J1_rot[0:2, :] @ ddq_dx]) + 
        # dr_dDist = np.array([force])
        # self.dr_dx = dr_dDist @ dDist_dx

        self.dr_du = np.zeros((4, len(u)))
        self.dr_du[0, self.nq_j+3*idx] = velocity[0]
        self.dr_du[1, self.nq_j+3*idx] = velocity[1]
        self.dr_du[2, self.nq_j+3*idx+1] = velocity[0]
        self.dr_du[3, self.nq_j+3*idx+1] = velocity[1]        

        # print(f"Complementarity Constraints:")
        # print(f"idx:{idx}")
        # print(f'confact frame ID:{fid}')
        # print(f'force:{force}' )
        # print(f"Dist:{Dist}")
        # print(f"Residual:{np.array([Dist * force])}")
        # print(f'drdu:{self.dr_du}')
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
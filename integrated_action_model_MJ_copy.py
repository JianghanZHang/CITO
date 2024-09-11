import crocoddyl
import numpy as np
# import pinocchio as pin
from pycontact import CCPADMMSolver, ContactProblem, ContactSolverSettings
import mujoco
import copy
import pinocchio as pin
from utils import change_convention_pin2mj, change_convention_mj2pin
CONVENTION = 'pin'
# This is for Mujoco collision checking
geom2_to_index = {20: 0, 32: 1, 44: 2, 56: 3}

class IntegratedActionModelForceMJ(crocoddyl.IntegratedActionModelAbstract):
    def __init__(self, DAM: crocoddyl.DifferentialActionModelAbstract, dt: float, with_cost_residuals: bool = True):
        """
        Forward Model with contact forces as explicit variables
        Input:
            nu : nu
            state : crocoddyl state multibody
            costModel : cost model
            constraintModel : constraints 
        """
        crocoddyl.IntegratedActionModelAbstract.__init__(
            self, DAM, dt, with_cost_residuals
        )
        self.fids = self.differential.fids
        self.nu = self.differential.nu
        self.mj_model = self.differential.mj_model
        self.mj_data = self.differential.mj_data
        self.with_cost_residuals = with_cost_residuals
        self.forces = np.zeros((4,6))
        self.contacts = self.mj_data.contact
        # print(f'constraints:\n {DAM.constraints}')
        # print(f'costs: \n{DAM.costs}')


    def calc(self, data, x, u=None):
        if CONVENTION == 'pin':
            x_mj = change_convention_pin2mj(x)
        else:
            x_mj = x.copy()
        # x_mj = x.copy()
        self.mj_data.time = 0.
        self.mj_data.qacc = np.zeros(self.state.nv)
        self.forces = np.zeros((4,6))
        self.mj_data.sensordata = np.zeros_like(self.mj_data.sensordata)
        self.contacts = 0
        if u is not None and not self.dt == 0.: # u not None and not terminal node

            q_mj, v_mj = x_mj[: self.state.nq], x_mj[-self.state.nv :]
            self.mj_data.qpos = q_mj.copy()
            self.mj_data.qvel = v_mj.copy()

            q, v = x[: self.state.nq].copy(), x[-self.state.nv :].copy()
            rmodel, rdata = self.state.pinocchio, data.differential.pinocchio

            pin.forwardKinematics(rmodel, rdata, q, v)
            pin.updateFramePlacements(rmodel, rdata)

            
            self.differential.costs.calc(data.differential.costs, x, u)
                
            if self.differential.constraints:
                data.differential.constraints.resize(self, data, True)
                self.differential.constraints.calc(data.differential.constraints, x, u)
                self.g_lb = self.differential.constraints.g_lb
                self.g_ub = self.differential.constraints.g_ub

                data.r = data.differential.r
                data.g = data.differential.g
                data.h = data.differential.h
                
            # forward one step with mujoco 
            self.mj_data.ctrl = u.copy()

            mujoco.mj_forward(self.mj_model, self.mj_data)
            data.differential.xout = self.mj_data.qacc.copy()
            mujoco.mj_Euler(self.mj_model, self.mj_data)
            data.cost = data.differential.costs.cost * self.dt

            # data.cost = data.differential.costs.cost * self.dt

            q = self.mj_data.qpos.copy()
            v = self.mj_data.qvel.copy()

        else:     
                           
                q_mj, v_mj = x_mj[: self.state.nq], x_mj[-self.state.nv :]
                self.mj_data.qpos = q_mj.copy()
                self.mj_data.qvel = v_mj.copy()
                
                q, v = x[: self.state.nq], x[-self.state.nv :]
                rmodel, rdata = self.state.pinocchio, data.differential.pinocchio

                pin.forwardKinematics(rmodel, rdata, q, v)
                pin.updateFramePlacements(rmodel, rdata)

            
                self.differential.costs.calc(data.differential.costs, x)
                if self.differential.constraints:
                    data.differential.constraints.resize(self, data, True)
                    self.differential.constraints.calc(data.differential.constraints, x)
                    self.g_lb = self.differential.constraints.g_lb
                    self.g_ub = self.differential.constraints.g_ub

                    data.r = data.differential.r
                    data.g = data.differential.g
                    data.h = data.differential.h

                self.mj_data.ctrl = np.zeros(self.mj_model.nu).copy()

                mujoco.mj_forward(self.mj_model, self.mj_data)
                data.differential.xout = self.mj_data.qacc.copy()
                # mujoco.mj_Euler(self.mj_model, self.mj_data)
                data.cost = data.differential.costs.cost

                q = self.mj_data.qpos.copy()
                v = self.mj_data.qvel.copy()
        if CONVENTION == 'pin':       
            xnext = change_convention_mj2pin(np.hstack([q, v]))
        else:
            xnext = np.hstack([q, v])
        data.xnext = xnext.copy()
        
        for i in range(self.mj_data.ncon):
            geom2 = self.mj_data.contact[i].geom2
            if geom2 in geom2_to_index:
                index = geom2_to_index[geom2]

                # Get the contact force
                mujoco.mj_contactForce(self.mj_model, self.mj_data, i, self.forces[index])

                # Transform the force according to the contact frame
                R = self.mj_data.contact[i].frame.reshape((3, 3))
                self.forces[index, :3] = R.T @ self.forces[index, :3]
            else:
                raise Exception("Bad contact geom2")
        
        self.contacts = self.mj_data.contact

    def calcDiff(self, data, x, u=None):
        if CONVENTION == 'pin':
            x_mj = change_convention_pin2mj(x)
        else:
            x_mj = x.copy()
        self.mj_data.time = 0.
        self.mj_data.qacc = np.zeros(self.state.nv)
        if u is not None and not self.dt == 0.: # u not None and not terminal node
            q_mj, v_mj = x_mj[: self.state.nq], x_mj[-self.state.nv :]
            self.mj_data.qpos = q_mj.copy()
            self.mj_data.qvel = v_mj.copy()

            q, v = x[: self.state.nq], x[-self.state.nv :]
            rmodel, rdata = self.state.pinocchio, data.differential.pinocchio

            pin.computeAllTerms(rmodel, rdata, q, v)
            pin.updateGlobalPlacements(rmodel, rdata)
            
            self.mj_data.ctrl = u.copy()

            A = np.zeros((2*self.state.nv, 2*self.state.nv))
            B = np.zeros((2*self.state.nv, self.nu))
        
            mujoco.mjd_transitionFD(self.mj_model, self.mj_data, eps = 1e-8, flg_centered = 1, A = A, B = B, C = None, D = None)        
            data.Fx = A.copy()
            data.Fu = B.copy()            

            if self.with_cost_residuals:
                self.differential.costs.calcDiff(data.differential.costs, x, u)
                data.Lxu = self.dt * data.differential.costs.Lxu
                data.Luu = self.dt * data.differential.costs.Luu
                data.Lu = self.dt * data.differential.costs.Lu
                data.Lx = self.dt * data.differential.costs.Lx 
                data.Lxx = self.dt * data.differential.costs.Lxx 
                    
            if self.differential.constraints:
                self.differential.constraints.calcDiff(data.differential.constraints, x, u)
                data.Hx = data.differential.constraints.Hx
                data.Hu = data.differential.constraints.Hu
                data.Gx = data.differential.constraints.Gx
                data.Gu = data.differential.constraints.Gu
            
        else:
            q_mj, v_mj = x_mj[: self.state.nq], x_mj[-self.state.nv :]
            self.mj_data.qpos = q_mj.copy()
            self.mj_data.qvel = v_mj.copy()

            q, v = x[: self.state.nq], x[-self.state.nv :]
            rmodel, rdata = self.state.pinocchio, data.differential.pinocchio

            pin.computeAllTerms(rmodel, rdata, q, v)
            pin.updateGlobalPlacements(rmodel, rdata)
            
            self.mj_data.ctrl = np.zeros(self.mj_model.nu).copy()


            A = np.zeros((2*self.state.nv, 2*self.state.nv))
            mujoco.mjd_transitionFD(self.mj_model, self.mj_data, eps = 1e-8, flg_centered = 1, A = A, B = None, C = None, D = None)
            data.Fx = A.copy()
            
            if self.with_cost_residuals:
                self.differential.costs.calcDiff(data.differential.costs, x)
                data.Lx = data.differential.costs.Lx 
                data.Lxx = data.differential.costs.Lxx

            if self.differential.constraints:
                self.differential.constraints.calcDiff(data.differential.constraints, x)
                data.Hx = data.differential.constraints.Hx
                data.Gx = data.differential.constraints.Gx

    def createData(self):
        data = crocoddyl.IntegratedActionModelAbstract.createData(self)
        data.differential = self.differential.createData()

        return data
    


            # mujoco.mj_step1(self.mj_model, self.mj_data)
            # print(f'*******************************************************************************************************')
            # q, v = x[: self.state.nq], x[-self.state.nv :]
            # rmodel, rdata = self.state.pinocchio, data.differential.pinocchio
            # print(f'Base Orientation:\n From pinocchio:\n {rdata.oMf[rmodel.getFrameId("base_link")].rotation}\n')
            # pin.forwardKinematics(rmodel, rdata, q, v)
            # pin.updateFramePlacements(rmodel, rdata)

            # print(f'_______________________________________________________________________________________________________')
            # n = 0
            # FL_foot_velocity = pin.getFrameVelocity(rmodel, rdata, rmodel.getFrameId("FL_foot"), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
            # FL_foot_position = rdata.oMf[rmodel.getFrameId("FL_foot")].translation
            # print(f'FL_foot_normal_force:\n From pinocchio: Not available\n From Mujoco: {self.mj_data.sensordata[n]}\n')
            # print(f'FL_foot_velocity:\n From pinocchio:{FL_foot_velocity}\n From Mujoco: {self.mj_data.sensordata[n+4:n+7]}\n')
            # print(f'FL_foot_position:\n From pinocchio:{FL_foot_position}\n From Mujoco: {self.mj_data.sensordata[n+1:n+4]}\n')

            # print(f'_______________________________________________________________________________________________________')
            # n += 7
            # FR_foot_velocity = pin.getFrameVelocity(rmodel, rdata, rmodel.getFrameId("FR_foot"), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
            # FR_foot_position = rdata.oMf[rmodel.getFrameId("FR_foot")].translation
            # print(f'FR_foot_normal_force:\n From pinocchio: Not available\n From Mujoco: {self.mj_data.sensordata[n]}\n')
            # print(f'FR_foot_velocity:\n From pinocchio:{FR_foot_velocity}\n From Mujoco: {self.mj_data.sensordata[n+4:n+7]}\n')
            # print(f'FR_foot_position:\n From pinocchio:{FR_foot_position}\n From Mujoco: {self.mj_data.sensordata[n+1:n+4]}\n')
            
            # print(f'_______________________________________________________________________________________________________')
            # n += 7
            # RL_foot_velocity = pin.getFrameVelocity(rmodel, rdata, rmodel.getFrameId("RL_foot"), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
            # RL_foot_position = rdata.oMf[rmodel.getFrameId("RL_foot")].translation
            # print(f'RL_foot_normal_force:\n From pinocchio: Not available\n From Mujoco: {self.mj_data.sensordata[n]}\n')
            # print(f'RL_foot_velocity:\n From pinocchio:{RL_foot_velocity}\n From Mujoco: {self.mj_data.sensordata[n+4:n+7]}\n')
            # print(f'RL_foot_position:\n From pinocchio:{RL_foot_position}\n From Mujoco: {self.mj_data.sensordata[n+1:n+4]}\n')

            # print(f'_______________________________________________________________________________________________________')
            # n += 7
            # RR_foot_velocity = pin.getFrameVelocity(rmodel, rdata, rmodel.getFrameId("RR_foot"), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
            # RR_foot_position = rdata.oMf[rmodel.getFrameId("RR_foot")].translation
            # print(f'RR_foot_normal_force:\n From pinocchio: Not available\n From Mujoco: {self.mj_data.sensordata[n]}\n')
            # print(f'RR_foot_velocity:\n From pinocchio:{RR_foot_velocity}\n From Mujoco: {self.mj_data.sensordata[n+4:n+7]}\n')
            # print(f'RR_foot_position:\n From pinocchio:{RR_foot_position}\n From Mujoco: {self.mj_data.sensordata[n+1:n+4]}\n')
            
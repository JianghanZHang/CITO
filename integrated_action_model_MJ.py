import crocoddyl
import numpy as np
# import pinocchio as pin
from pycontact import CCPADMMSolver, ContactProblem, ContactSolverSettings
import mujoco
import pinocchio as pin

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
        self.nu = self.differential.nu
        self.mj_model = self.differential.mj_model
        self.mj_data = self.differential.mj_data
        self.with_cost_residuals = with_cost_residuals
        self.forces = np.zeros((4,6))
        print(f'constraints:\n {DAM.constraints}')
        print(f'costs: \n{DAM.costs}')


    def calc(self, data, x, u=None):
        self.mj_data.time = 0.
        self.mj_data.qacc = np.zeros(self.state.nv)
        if u is not None and not self.dt == 0.: # u not None and not terminal node
            q, v = x[: self.state.nq], x[-self.state.nv :]
            self.mj_data.qpos = q.copy()
            self.mj_data.qvel = v.copy()
        
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
            # self.mj_data.actuator_force = u.copy()
            
            mujoco.mj_forward(self.mj_model, self.mj_data)
            data.differential.xout = self.mj_data.qacc.copy()
            mujoco.mj_Euler(self.mj_model, self.mj_data)
            data.cost = data.differential.costs.cost * self.dt

            q = self.mj_data.qpos.copy()
            v = self.mj_data.qvel.copy()
            xnext = np.hstack([q, v])
            data.xnext = xnext.copy()
        
        else:
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
                data.cost = data.differential.costs.cost * self.dt

                q = self.mj_data.qpos.copy()
                v = self.mj_data.qvel.copy()
                xnext = np.hstack([q, v])
                data.xnext = xnext.copy()
        
        for i in range(min(self.mj_data.ncon, 4)):

            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, self.forces[i])
            R = self.mj_data.contact[i].frame.reshape((3,3))
            self.forces[i, :3] = R.T @ self.forces[i, :3]

    def calcDiff(self, data, x, u=None):
        self.mj_data.time = 0.
        self.mj_data.qacc = np.zeros(self.state.nv)
        if u is not None and not self.dt == 0.: # u not None and not terminal node
            q, v = x[: self.state.nq], x[-self.state.nv :]
            self.mj_data.qpos = q.copy()
            self.mj_data.qvel = v.copy()

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
                # import pdb; pdb.set_trace()
                    
            if self.differential.constraints:
                self.differential.constraints.calcDiff(data.differential.constraints, x, u)
                data.Hx = data.differential.constraints.Hx
                data.Hu = data.differential.constraints.Hu
                data.Gx = data.differential.constraints.Gx
                data.Gu = data.differential.constraints.Gu
            
        else:
            q, v = x[: self.state.nq], x[-self.state.nv :]
            self.mj_data.qpos = q.copy()
            self.mj_data.qvel = v.copy()

            rmodel, rdata = self.state.pinocchio, data.differential.pinocchio

            pin.computeAllTerms(rmodel, rdata, q, v)
            pin.updateGlobalPlacements(rmodel, rdata)
            
            self.mj_data.ctrl = np.zeros(self.mj_model.nu).copy()
            # self.mj_data.actuator_force = np.zeros(self.mj_model.nu)


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
        
        # print(f'contact: \n {self.mj_data.contact}')

        # print(f'In IntegratedActionModelForceMJ calcDiff')
        # print(f'ctrl: \n {self.mj_data.ctrl}')
        # print(f'actuator_force: \n {self.mj_data.actuator_force}')
        # print(f'qfrc_actuator: \n {self.mj_data.qfrc_actuator}')
        # print(f'qfrc_applied: \n {self.mj_data.qfrc_applied}')
        # print(f'qfrc_bias: \n {self.mj_data.qfrc_bias}')
        # print(f'qfrc_inverse: \n {self.mj_data.qfrc_inverse}')
        # print(f'_________________________________________________________')
                   
    def createData(self):
        data = crocoddyl.IntegratedActionModelAbstract.createData(self)
        data.differential = self.differential.createData()

        # data.pinocchio = pin.Data(self.state.pinocchio)
        # data.multibody = crocoddyl.DataCollectorMultibody(data.pinocchio)
        # data.costs = self.costs.createData(data.multibody)
        # data.costs.shareMemory(
        #     data
        # )  # this allows us to share the memory of cost-terms of action model
        # if self.constraints:
        #     data.constraints = self.constraints.createData(data.multibody)
        #     data.constraints.shareMemory(data)

        return data
    

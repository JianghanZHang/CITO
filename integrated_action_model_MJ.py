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
        self.time_step = dt
        self.mj_model = self.differential.mj_model
        self.mj_data = self.differential.mj_data
        self.with_cost_residuals = with_cost_residuals

    def calc(self, data, x, u=None):
        q, v = x[: self.state.nq], x[-self.state.nv :]
        self.mj_data.qpos = q.copy()
        self.mj_data.qvel = v.copy()
    
        rmodel, rdata = self.state.pinocchio, data.differential.pinocchio

        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)

        if u is None:
            self.differential.costs.calc(data.differential.costs, x)
        else:
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

        mujoco.mj_step1(self.mj_model, self.mj_data)
        if u is not None:
            self.mj_data.ctrl = u.copy()
            # self.mj_data.actuator_force = u.copy()
        else:
            self.mj_data.ctrl = np.zeros(self.mj_model.nu)
            # self.mj_data.actuator_force = np.zeros(self.mj_model.nu)

        mujoco.mj_step2(self.mj_model, self.mj_data)
        mujoco.mj_rnePostConstraint(self.mj_model, self.mj_data)
        # mujoco.mj_forward(self.mj_model, self.mj_data)
        data.differential.xout = self.mj_data.qacc.copy()
        # mujoco.mj_Euler(self.mj_model, self.mj_data)
        data.cost = data.differential.costs.cost * self.time_step

        q = self.mj_data.qpos.copy()
        v = self.mj_data.qvel.copy()
        xnext = np.hstack([q, v])
        data.xnext = xnext

        print(f'contact: \n {self.mj_data.contact}')
        print(f'In IntegratedActionModelForceMJ calc')
        print(f'ctrl: \n {self.mj_data.ctrl}')
        print(f'actuator_force: \n {self.mj_data.actuator_force}')
        print(f'qfrc_actuator: \n {self.mj_data.qfrc_actuator}')
        print(f'qfrc_applied: \n {self.mj_data.qfrc_applied}')
        print(f'qfrc_bias: \n {self.mj_data.qfrc_bias}')
        print(f'qfrc_inverse: \n {self.mj_data.qfrc_inverse}')
        print(f'cfrc_ext : \n {self.mj_data.cfrc_ext}')
        print(f'_________________________________________________________')

    def calcDiff(self, data, x, u=None):
        if u is not None:
            q, v = x[: self.state.nq], x[-self.state.nv :]
            self.mj_data.qpos = q.copy()
            self.mj_data.qvel = v.copy()

            rmodel, rdata = self.state.pinocchio, data.differential.pinocchio

            pin.forwardKinematics(rmodel, rdata, q, v)
            pin.updateGlobalPlacements(rmodel, rdata)
            
            self.mj_data.ctrl = u.copy()
            # self.mj_data.actuator_force = u.copy()

            A = np.zeros((2*self.state.nv, 2*self.state.nv))
            B = np.zeros((2*self.state.nv, self.nu))
        
            mujoco.mjd_transitionFD(self.mj_model, self.mj_data, eps = 1e-8, flg_centered = 1, A = A, B = B, C = None, D = None)        
            data.Fx = A.copy()
            data.Fu = B.copy()            

            if self.with_cost_residuals:
                self.differential.costs.calcDiff(data.differential.costs, x, u)
                data.Lxu = self.time_step * data.differential.costs.Lxu
                data.Luu = self.time_step * data.differential.costs.Luu
                data.Lu = self.time_step * data.differential.costs.Lu
                data.Lx = self.time_step * data.differential.costs.Lx 
                data.Lxx = self.time_step * data.differential.costs.Lxx 
                # import pdb; pdb.set_trace()
                    
            if self.differential.constraints:
                self.differential.constraints.calcDiff(data.differential.constraints, x, u)
                data.Hx = self.time_step * data.differential.constraints.Hx
                data.Hu = self.time_step * data.differential.constraints.Hu
                data.Gx = self.time_step * data.differential.constraints.Gx
                data.Gu = self.time_step * data.differential.constraints.Gu
            
        else:
            q, v = x[: self.state.nq], x[-self.state.nv :]
            self.mj_data.qpos = q.copy()
            self.mj_data.qvel = v.copy()

            rmodel, rdata = self.state.pinocchio, data.differential.pinocchio

            pin.forwardKinematics(rmodel, rdata, q, v)
            pin.updateGlobalPlacements(rmodel, rdata)
            
            self.mj_data.ctrl = np.zeros(self.mj_model.nu)
            # self.mj_data.actuator_force = np.zeros(self.mj_model.nu)


            A = np.zeros((2*self.state.nv, 2*self.state.nv))
            mujoco.mjd_transitionFD(self.mj_model, self.mj_data, eps = 1e-8, flg_centered = 1, A = A, B = None, C = None, D = None)
            data.Fx = A.copy()
            data.Fu = None
            
            if self.with_cost_residuals:
                self.differential.costs.calcDiff(data.differential.costs, x)
                data.Lx = self.time_step * data.differential.costs.Lx 
                data.Lxx = self.time_step * data.differential.costs.Lxx

            if self.differential.constraints:
                
                self.differential.constraints.calcDiff(data.differential.constraints, x)
                data.Hx = self.time_step * data.differential.constraints.Hx
                data.Gx = self.time_step * data.differential.constraints.Gx
        
        print(f'contact: \n {self.mj_data.contact}')

        print(f'In IntegratedActionModelForceMJ calcDiff')
        print(f'ctrl: \n {self.mj_data.ctrl}')
        print(f'actuator_force: \n {self.mj_data.actuator_force}')
        print(f'qfrc_actuator: \n {self.mj_data.qfrc_actuator}')
        print(f'qfrc_applied: \n {self.mj_data.qfrc_applied}')
        print(f'qfrc_bias: \n {self.mj_data.qfrc_bias}')
        print(f'qfrc_inverse: \n {self.mj_data.qfrc_inverse}')
        print(f'_________________________________________________________')
                   
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
    

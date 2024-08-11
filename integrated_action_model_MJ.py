import crocoddyl
import numpy as np
# import pinocchio as pin
from pycontact import CCPADMMSolver, ContactProblem, ContactSolverSettings
import mujoco
import pinocchio as pin

class DummyDifferentialActionModelMJ(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, mj_model, mj_data, state, nu, nq_j, fids, costModel, constraintModel = None):
        """
        Forward Model with contact forces as explicit variables
        Input:
            nu : nu
            state : crocoddyl state multibody
            costModel : cost model
            constraintModel : constraints 
        """
        ng = nh = 0
        if constraintModel:
            ng = constraintModel.ng
            nh = constraintModel.nh
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, nu, costModel.nr, ng, nh
        )
        self.costs = costModel
        self.constraints = constraintModel
        self.fids = fids
        self.nq_j = nq_j
        self.tau = np.zeros(self.state.nv)
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.mj_model.opt.tolerance = 0.

    def calc(self, data, x, u=None):
        q, v = x[: self.state.nq], x[-self.state.nv :]
        self.mj_data.qpos = q.copy()
        self.mj_data.qvel = v.copy()
        
        if u is not None:
            self.mj_data.ctrl = u.copy()

        rmodel, rdata = self.state.pinocchio, data.pinocchio
        fids = self.fids
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        # Forward dynamics with mujoco 
        mujoco.mj_forward(self.mj_model, self.mj_data)
        a = self.mj_data.qacc
        data.xout = a
        if u is None:
            self.costs.calc(data.costs, x)
        else:
            self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost
        if self.constraints:
            data.constraints.resize(self, data)
            self.constraints.calc(data.constraints, x, u)
            self.g_lb = self.constraints.g_lb
            self.g_ub = self.constraints.g_ub

    def calcDiff(self, data, x, u=None):
        pass

    def createData(self):
        data = crocoddyl.DifferentialActionModelAbstract.createData(self)
        data.pinocchio = pin.Data(self.state.pinocchio)
        data.multibody = crocoddyl.DataCollectorMultibody(data.pinocchio)
        data.costs = self.costs.createData(data.multibody)
        data.costs.shareMemory(
            data
        )  # this allows us to share the memory of cost-terms of action model
        if self.constraints:
            data.constraints = self.constraints.createData(data.multibody)
            data.constraints.shareMemory(data)

        return data

class IntegratedActionModelForceMJ(crocoddyl.IntegratedActionDataAbstract):
    def __init__(self, DAM: crocoddyl.DifferentialActionModel, dt: float, with_cost_residuals: bool = True):
        """
        Forward Model with contact forces as explicit variables
        Input:
            nu : nu
            state : crocoddyl state multibody
            costModel : cost model
            constraintModel : constraints 
        """
        ng = nh = 0

        crocoddyl.IntegratedActionModelAbstract.__init__(
            self, DAM, dt, with_cost_residuals
        )
        self.time_step = dt
        self.mj_model = self.differential.mj_model
        self.mj_data = self.differential.mj_data
        self.with_cost_residuals = with_cost_residuals

    def calc(self, data, x, u=None):
        q, v = x[: self.state.nq], x[-self.state.nv :]
        self.mj_data.qpos = q.copy()
        self.mj_data.qvel = v.copy()
    
        rmodel, rdata = self.state.pinocchio, data.pinocchio
        self.differential.calc(data.differential, x, u)

        data.r = data.differential.r
        data.g = data.differential.g
        data.h = data.differential.h

        # forward one step with mujoco 
        mujoco.mj_step1(self.mj_model, self.mj_data)

        if u is not None:
            self.mj_data.ctrl = u.copy()
        mujoco.mj_step2(self.mj_model, self.mj_data)

        self.differen
        if u is None:
            self.costs.calc(data.costs, x)
        else:
            self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost
        if self.differential.constraints:
            data.constraints.resize(self, data)
            self.constraints.calc(data.constraints, x, u)
            self.g_lb = self.constraints.g_lb
            self.g_ub = self.constraints.g_ub

    def calcDiff(self, data, x, u=None):
        q, v = x[: self.state.nq], x[-self.state.nv :]
        self.mj_data.qpos = q.copy()
        self.mj_data.qvel = v.copy()

        if u is not None:
            self.mj_data.ctrl = u.copy()
        
        A = np.zeros((2*self.state.nv, 2*self.state.nv))
        B = np.zeros((2*self.state.nv, self.state.nv))
        mujoco.mjd_transitionFD(self.mj_model, self.mj_data, eps = 1e-6, flag_centered = 1, A = A, B = B, C = None, D = None)
        data.Fx = A
        data.Fu = B
    
        if self.with_cost_residuals:
            if u is not None:
                self.differential.costs.calcDiff(data.differential.costs, x, u)
                data.Lxu = self.time_step * data.differential.costs.Lxu
                data.Luu = self.time_step * data.differential.costs.Luu
                data.Lu = self.time_step * data.differential.costs.Lu
                data.Lx = self.time_step * data.differential.costs.Lx 
                data.Lxx = self.time_step * data.differential.costs.Lxx 
            else:
                self.differential.costs.calcDiff(data.differential.costs, x)

                data.Lx = self.time_step * data.differential.costs.Lx 
                data.Lxx = self.time_step * data.differential.costs.Lxx
                
        if self.differernial.constraints:
            if u is not None:
                self.differential.constraints.calcDiff(data.differential.constraints, x, u)
                data.Hx = self.time_step * data.differential.constraints.Hx
                data.Hu = self.time_step * data.differential.constraints.Hu
                data.Gx = self.time_step * data.differential.constraints.Gx
                data.Gu = self.time_step * data.differential.constraints.Gu
            else:
                self.differential.constraints.calcDiff(data.differential.constraints, x)
                data.Hx = self.time_step * data.differential.constraints.Hx
                data.Gx = self.time_step * data.differential.constraints.Gx

    def createData(self):
        data = crocoddyl.IntegratedActionModelAbstract.createData(self)
        data.pinocchio = pin.Data(self.state.pinocchio)
        data.multibody = crocoddyl.DataCollectorMultibody(data.pinocchio)
        data.costs = self.costs.createData(data.multibody)
        data.costs.shareMemory(
            data
        )  # this allows us to share the memory of cost-terms of action model
        if self.constraints:
            data.constraints = self.constraints.createData(data.multibody)
            data.constraints.shareMemory(data)

        return data
    

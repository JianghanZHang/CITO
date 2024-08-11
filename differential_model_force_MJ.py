import crocoddyl
import numpy as np
# import pinocchio as pin
from pycontact import CCPADMMSolver, ContactProblem, ContactSolverSettings
import mujoco
import pinocchio as pin

class DifferentialActionModelForceMJ(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, mj_model, mj_data, state, nu, nq_j, fids, costModel, constraintModel = None, forceRecorder = None):
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
    

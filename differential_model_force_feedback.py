import crocoddyl
import numpy as np
import pinocchio as pin


class DifferentialActionModelForceFeedback(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, nu, nq_j, fids, costModel, constraintModel = None):
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
        self.tau = np.zeros(self.state.nv) # Have to use this dimension due to pin.aba() arguments signature
    def calc(data, x, u):
        pass 

    def calcDiff(data, x, u):
        pass
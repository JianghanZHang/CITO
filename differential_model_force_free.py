import crocoddyl
import numpy as np
import pinocchio as pin

class DifferentialActionModelForceExplicit(crocoddyl.DifferentialActionModelAbstract):
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

    def calc(self, data, x, u=None):
        q, v = x[: self.state.nq], x[-self.state.nv :]
        rmodel, rdata = self.state.pinocchio, data.pinocchio
        fids = self.fids
        if u is None:
            self.tau = np.zeros(self.state.nv) # Have to use this dimension due to pin.aba() arguments signature
            forces = np.zeros(3 * len(fids)) # Extract contact forces

        else:
            self.tau[6:] = u[:self.nq_j]
            forces = u[self.nq_j:] # Extract contact forces

        localWorldAlignedForces = [forces[3*i:3*i+3] for i in range(len(fids))] # reshape the forces into #contacts x 3.

        JointFrameForces = [pin.Force(np.zeros(6)) for _ in range(rmodel.njoints)]
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)

        for idx, localWorldAlignedForce in enumerate(localWorldAlignedForces):
            oRl = rdata.oMf[fids[idx]].rotation # orientation of the local frame wrt world frame
            jid = rmodel.frames[fids[idx]].parent # joint id of the parent frame
            jMf =  (rdata.oMi[jid].inverse()*rdata.oMf[fids[idx]]) # SE3 matrix from contact frame to parent joint frame
            LocalFrameforce = pin.Force(oRl.T @ localWorldAlignedForce.copy(), np.zeros(3)) # 6d force in the local frame
            JointFrameForces[jid] += jMf.act(LocalFrameforce)  # 6d forces in the parent joint frame

        # import pdb; pdb.set_trace() 
        a = pin.aba(rmodel, rdata, q, v, self.tau, JointFrameForces)

        #computing derivatives
        pin.updateGlobalPlacements(rmodel, rdata)
        pin.framesForwardKinematics(rmodel, rdata, q)
        da_dq, da_dv, da_dtau = pin.computeABADerivatives(rmodel, rdata, q, v, self.tau, JointFrameForces)

        da_df = np.zeros((rmodel.nv,3*len(fids))) #derivatives wrt forces
        for idx, localWorldAlignedForce in enumerate(localWorldAlignedForces):
            oRl = rdata.oMf[fids[idx]].rotation # orientation of the local frame wrt world frame
            ljac = pin.computeFrameJacobian(rmodel, rdata, q, fids[idx], pin.LOCAL) # local jacobian

            #dF_local/dq = d(R_ol@F_local)/dq = (R_lo @ F_world) X Jacobian_local (using Lie algebra)
            dfc_dq = pin.skew(oRl.T @ localWorldAlignedForce.copy()) @ ljac[3:, :] 

            dfc_df = oRl.T # dF_world/dt = d(R_ol@F_local)/dt
            da_dfc = ((rdata.Minv @ ljac.T)[:,:3]) # compute the local force derivatives through the EoM
            da_df[:,3*idx:3*(idx+1)] =  da_dfc @ dfc_df # transform the local force derivatives to the global frame
            da_dq += da_dfc @ dfc_dq

        self.da_dx = np.hstack((da_dq, da_dv))        
        self.da_du = np.hstack((da_dtau[:,-self.nq_j:], da_df))
        # import pdb; pdb.set_trace() 
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
        self.calc(data, x, u)
        # Computing the dynamics derivatives
        data.Fx = self.da_dx
        data.Fu = self.da_du

        # Computing the cost derivatives
        if u is None:
            self.costs.calcDiff(data.costs, x)
        else:
            self.costs.calcDiff(data.costs, x, u)
            
        if self.constraints:
            self.constraints.calcDiff(data.constraints, x, u)

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
    

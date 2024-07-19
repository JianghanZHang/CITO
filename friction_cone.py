

import pinocchio as pin
import numpy as np
import crocoddyl
from force_derivatives import LocalWorldAlignedForceDerivatives, ForceDerivatives


class ResidualLinearFrictionCone(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, njoints, nContants, nu, fid, idx, mu = 0.8):
        """
        Ensures the normal force is positive
        Input:
            state : state multibody
            scene : Scene
            fid : contact frameId
            idx : segment of force
            mu : friction coeff
        """
        self.nc = 5
        crocoddyl.ResidualModelAbstract.__init__(self, state, self.nc, nu, True, False, True)
        self.idx = idx
        self.fid = fid
        self.mu = mu
        self.nq_j = njoints
        self.nb_contacts = nContants
        self.dcone_df = np.zeros((self.nc, 3*self.nb_contacts + njoints))
        self.dcone_dx = np.zeros((self.nc, state.pinocchio.nv + state.pinocchio.nv))
        self.rdata = pin.Data(state.pinocchio)
        self.rmodel = state.pinocchio
        self.eps = 1e-2 #offset to avoid division by zero with derivatives

    def calc(self, data, x, u): 

        idx = self.idx
        rmodel, rdata = self.rmodel, self.rdata
        forces = u[self.nq_j: ]
        localWorldAlignedForces = [forces[3*i:3*i+3] for i in range(self.nb_contacts)]

        f, df_dx = ForceDerivatives(localWorldAlignedForces[idx], x, self.fid, rmodel, rdata)


        data.r = np.array([self.mu*f[2] + f[0], 
                           self.mu*f[2] - f[0],
                           self.mu*f[2] + f[1],
                           self.mu*f[2] - f[1],
                           f[2]])
        
        dcone_df = np.array([[1.0, 0.0, self.mu], 
                              [-1.0, 0.0, self.mu],
                              [0.0, 1.0, self.mu,],
                              [0.0, -1.0, self.mu,],
                              [0.0, 0.0,  1.0]])
        
        self.dcone_df[:, self.nq_j + 3*idx : self.nq_j + 3*(idx+1)] =  dcone_df
        self.dcone_dx =  dcone_df @ df_dx

        # print(f"Friction Constraints:")
        # print(f"idx:{idx}")
        # print(f'confact frame ID:{self.fid}')
        # print(f'force:{f}' )
        # print(f"Residual:{data.r}")
        # print(f'drdu:{self.dcone_df}')
        # print(f'drdx:{self.dcone_dx}')

        # print(f'confact frame:{rdata.oMf[self.fid]}')

    def calcDiff(self, data, x, u):
        self.calc(data, x, u)

        data.Ru = self.dcone_df
        data.Rx = self.dcone_dx



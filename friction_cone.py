

import pinocchio as pin
import numpy as np
import crocoddyl
from force_derivatives import LocalWorldAlignedForceDerivatives


class ResidualLinearFrictionCone(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, scene, fid, idx, mu = 0.8):
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
        crocoddyl.ResidualModelAbstract.__init__(self, state, self.nc, scene["nu"], True, False, True)
        self.idx = idx
        self.fid = fid
        self.mu = mu
        self.rnv = scene["rnv"]
        self.nb_contacts = scene["nbContactFrames"]
        self.dcone_df = np.zeros((self.nc, 3*self.nb_contacts + scene["rnv"]))
        self.dcone_dx = np.zeros((self.nc, state.pinocchio.nv + state.pinocchio.nv))
        self.rdata = pin.Data(state.pinocchio)
        self.rmodel = state.pinocchio
        self.eps = 1e-2 #offset to avoid division by zero with derivatives

    def calc(self, data, x, u): 

        idx = self.idx
        rmodel, rdata = self.rmodel, self.rdata
        forces = u[self.rnv: ]
        localWorldAlignedForces = [forces[3*i:3*i+3] for i in range(self.nb_contacts)]

        fc, dfc_df, dfc_dx = LocalWorldAlignedForceDerivatives(localWorldAlignedForces[idx], x, self.fid, rmodel, rdata)
        data.r = np.array([self.mu*fc[0] + fc[1], 
                           self.mu*fc[0] - fc[1],
                           self.mu*fc[0] + fc[2],
                           self.mu*fc[0] - fc[2],
                           fc[0]])
        dcone_dfc = np.array([[self.mu,  1.0,  0.0], 
                              [self.mu, -1.0,  0.0],
                              [self.mu,  0.0,  1.0],
                              [self.mu,  0.0, -1.0],
                              [1.0,      0.0,  0.0]])
        
        self.dcone_df[:, self.rnv + 3*idx : self.rnv + 3*(idx+1)] =  dcone_dfc @ dfc_df
        self.dcone_dx =  dcone_dfc @ dfc_dx

    def calcDiff(self, data, x, u):
        self.calc(data, x, u)

        data.Ru = self.dcone_df
        data.Rx = self.dcone_dx



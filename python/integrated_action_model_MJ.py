import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../robots/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'python/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'robots/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils/')))

import crocoddyl
import numpy as np
# import pinocchio as pin
import mujoco
import copy
import pinocchio as pin
from utils import stateMapping_mj2pin, stateMapping_pin2mj
from numerical_difference import stateMappingDerivative_mj2pin_numDiff, stateMappingDerivative_pin2mj_numDiff
CONVENTION = 'mujoco'
# This is for Mujoco collision checkings
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


    def calc(self, data, x, u=None):
        rmodel, rdata = self.state.pinocchio, data.differential.pinocchio

        if CONVENTION == 'mujoco':
            x_mj, _, _ = stateMapping_pin2mj(x, rmodel)
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

            pin.forwardKinematics(rmodel, rdata, q, v)
            pin.updateFramePlacements(rmodel, rdata)

            
            self.differential.costs.calc(data.differential.costs, x, u)
                
            if self.differential.constraints:
                data.differential.constraints.resize(self, data)
                self.differential.constraints.calc(data.differential.constraints, x, u)
                self.differential.g_lb = self.differential.constraints.g_lb.copy()
                self.differential.g_ub = self.differential.constraints.g_ub.copy()


                data.r = data.differential.r.copy()
                data.differential.g = data.differential.constraints.g.copy()
                data.g = data.differential.g.copy()

            # forward one step with mujoco 
            self.mj_data.ctrl = u.copy()

            mujoco.mj_forward(self.mj_model, self.mj_data)
            data.differential.xout = self.mj_data.qacc.copy()
            mujoco.mj_Euler(self.mj_model, self.mj_data)
            data.cost = data.differential.costs.cost * self.dt

            q = self.mj_data.qpos.copy()
            v = self.mj_data.qvel.copy()

        else:     
                           
                q_mj, v_mj = x_mj[: self.state.nq], x_mj[-self.state.nv :]
                self.mj_data.qpos = q_mj.copy()
                self.mj_data.qvel = v_mj.copy()
                
                q, v = x[: self.state.nq], x[-self.state.nv :]

                pin.forwardKinematics(rmodel, rdata, q, v)
                pin.updateFramePlacements(rmodel, rdata)

            
                self.differential.costs.calc(data.differential.costs, x)
                if self.differential.constraints:
                    data.differential.constraints.resize(self, data)
                    self.differential.constraints.calc(data.differential.constraints, x)
                    self.differential.g_lb = self.differential.constraints.g_lb.copy()
                    self.differential.g_ub = self.differential.constraints.g_ub.copy()

                    data.r = data.differential.r.copy()
                    data.differential.g = data.differential.constraints.g.copy()
                    data.g = data.differential.g.copy()

                self.mj_data.ctrl = np.zeros(self.mj_model.nu).copy()

                mujoco.mj_forward(self.mj_model, self.mj_data)
                data.differential.xout = self.mj_data.qacc.copy()
                mujoco.mj_Euler(self.mj_model, self.mj_data)
                data.cost = data.differential.costs.cost

                q = self.mj_data.qpos.copy()
                v = self.mj_data.qvel.copy()
        
        if CONVENTION == 'mujoco':       
            xnext, _, _ = stateMapping_mj2pin(np.hstack([q, v]), rmodel)
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
      
        
        self.contacts = self.mj_data.contact

    def calcDiff(self, data, x, u=None):
        rmodel, rdata = self.state.pinocchio, data.differential.pinocchio

        if CONVENTION == 'mujoco':
            x_mj, _, _ = stateMapping_pin2mj(x, rmodel)
            dx_mj_dx_pin = stateMappingDerivative_pin2mj_numDiff(x, rmodel, self.mj_model)
        else:
            x_mj = x.copy()
        self.mj_data.time = 0.
        self.mj_data.qacc = np.zeros(self.state.nv)
        if u is not None and not self.dt == 0.: # u not None and not terminal node
            q_mj, v_mj = x_mj[: self.state.nq], x_mj[-self.state.nv :]
            self.mj_data.qpos = q_mj.copy()
            self.mj_data.qvel = v_mj.copy()

            q, v = x[: self.state.nq], x[-self.state.nv :]

            pin.computeAllTerms(rmodel, rdata, q, v)
            pin.updateGlobalPlacements(rmodel, rdata)
            
            self.mj_data.ctrl = u.copy()

            A = np.zeros((2*self.state.nv, 2*self.state.nv))
            B = np.zeros((2*self.state.nv, self.nu))
        
            mujoco.mjd_transitionFD(self.mj_model, self.mj_data, eps = 1e-8, flg_centered = 1, A = A, B = B, C = None, D = None)        
            Fx_mj = A.copy()
            Fu_mj = B.copy()     
            if CONVENTION == 'mujoco':
                x_pin_next = data.xnext.copy()
                x_mj_next, _, _ = stateMapping_pin2mj(x_pin_next, rmodel)
                dx_next_pin_dx_next_mj = stateMappingDerivative_mj2pin_numDiff(x_mj_next, rmodel, self.mj_model)

                Fx = dx_next_pin_dx_next_mj @ Fx_mj @ dx_mj_dx_pin # TODO: validate the correctness
                Fu = dx_next_pin_dx_next_mj @ Fu_mj
            else:
                Fx = A.copy()
                Fu = B.copy()

            data.Fx = Fx.copy()
            data.Fu = Fu.copy()            
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

            pin.computeAllTerms(rmodel, rdata, q, v)
            pin.updateGlobalPlacements(rmodel, rdata)
            
            self.mj_data.ctrl = np.zeros(self.mj_model.nu).copy()


            A = np.zeros((2*self.state.nv, 2*self.state.nv))
            mujoco.mjd_transitionFD(self.mj_model, self.mj_data, eps = 1e-8, flg_centered = 1, A = A, B = None, C = None, D = None)
            Fx_mj = A.copy()
            if CONVENTION == 'mujoco':
                x_pin_next = data.xnext.copy()
                x_mj_next, _, _ = stateMapping_pin2mj(x_pin_next, rmodel)
                # import pdb; pdb.set_trace()

                dx_next_pin_dx_next_mj = stateMappingDerivative_mj2pin_numDiff(x_mj_next, rmodel, self.mj_model)

                Fx = dx_next_pin_dx_next_mj @ Fx_mj @ dx_mj_dx_pin # TODO: validate the correctness
            else:
                Fx = A.copy()

            data.Fx = Fx.copy()
                      
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
    
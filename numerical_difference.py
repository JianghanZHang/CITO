## This class draws arrows to depic the force in meshcat
## Author : Huaijiang Zhu

import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf
import pinocchio as pin
import meshcat
import mujoco
from robot_env import ROOT_JOINT_INDEX
from utils import *

def _numdiffSE3toEuclidian_stateMapping_pin2mj(f, x0_pin, rmodel, mj_model, h=1e-6):
    q0_pin, v0_pin = x0_pin[:rmodel.nq].copy(), x0_pin[-rmodel.nv:].copy()
    x0_mj = f(x0_pin).copy() # x0 in Mujoco
    Fx = []
    nq, nv = rmodel.nq, rmodel.nv
    q0_mj, v0_mj = x0_mj[:nq].copy(), x0_mj[-nv:].copy()
    dx = np.zeros(2 * nv)
    for ix in range(len(dx)):
       
        dx[ix] += h
        hinv = 1.0 / h
        x_prime_pin = np.hstack((pin.integrate(rmodel, x0_pin[:nq].copy(), dx[:nv]), x0_pin[nq:].copy() + dx[nv:]))
        x_prime_mj = f(x_prime_pin).copy() # x_prime in Mujoco
        
        q_prime_mj, v_prime_mj = x_prime_mj[:nq].copy(), x_prime_mj[-nv:].copy()
        q_prime_pin, v_prime_pin = x_prime_pin[:nq].copy(), x_prime_pin[-nv:].copy()
        diff_q = np.zeros((nv,))

        #Use mujoco differentiation for consistent lie group computation
        mujoco.mj_differentiatePos(mj_model, diff_q, 1., q0_mj, q_prime_mj) 
        
        diff_v = (v_prime_mj - v0_mj) 
        diff = np.hstack((diff_q, diff_v)) / h 
        Fx.append(diff)
        dx[ix] = 0.0

    Fx = np.array(Fx).T
    return Fx


# This function computes dx_mj_dx_pin using numerical differentiation
def stateMappingDerivative_pin2mj_numDiff(x_pin, rmodel, mj_model):

    def f(x):
        _x ,_ ,_ = stateMapping_pin2mj(x, rmodel)
        return _x

    dx_mj_dx_pin = _numdiffSE3toEuclidian_stateMapping_pin2mj(f, x_pin, rmodel, mj_model, 1e-8)

    return dx_mj_dx_pin

# This function computes dx_pin_dx_mj using numerical differentiation
def stateMappingDerivative_mj2pin_numDiff(x_mj, rmodel, mj_model):
    # print(f'entered stateMappingDerivative_mj2pin_numDiff')
    def f(x):
        # print(f'before mj2pin mapping x:\n {x}')
        _x ,_ ,_ = stateMapping_mj2pin(x, rmodel)
        # print(f'after mj2pin mapping x: {_x}')
        return _x

    dx_pin_dx_mj = _numdiffSE3toEuclidian_stateMapping_mj2pin(f, x_mj, rmodel, mj_model, 1e-8)

    return dx_pin_dx_mj

def _numdiffSE3toEuclidian_stateMapping_mj2pin(f, x0_mj, rmodel, mj_model, h=1e-6):
    nq, nv = rmodel.nq, rmodel.nv

    q0_mj, v0_mj = x0_mj[:nq].copy(), x0_mj[-nv:].copy()
    x0_pin = f(x0_mj).copy() # x0 in Mujoco
    Fx = []
    q0_pin, v0_pin = x0_pin[:nq].copy(), x0_pin[-nv:].copy()
    dx = np.zeros(2 * nv)
    for ix in range(len(dx)):
        # print(f'ix: {ix}')
        dx[ix] += h
        q_prime_mj = q0_mj.copy()
        mujoco.mj_integratePos(mj_model, q_prime_mj, dx[:nv], 1.)
        x_prime_mj = np.hstack((q_prime_mj, x0_mj[nq:].copy() + dx[nv:]))

        x_prime_pin = f(x_prime_mj).copy() # x_prime in Pinocchio    
        q_prime_pin, v_prime_pin = x_prime_pin[:nq].copy(), x_prime_pin[-nv:].copy()
        
        #Use pinocchio differentiation for consistent lie group computation
        diff_q = pin.difference(rmodel, q0_pin, q_prime_pin)
        diff_v = (v_prime_pin - v0_pin) 
        diff = np.hstack((diff_q, diff_v)) / h 
        Fx.append(diff)
        dx[ix] = 0.0

    Fx = np.array(Fx).T
    return Fx

# Test calcDiff function in ResidualModels with numerical differentiation results for floaing base robots.
# Author: Jianghan Zhang
import sys
import os

# Add the outer directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from numerical_differences import numdiffSE3toEuclidian
from friction_cone import ResidualLinearFrictionCone
from ResidualModels import ResidualModelFrameTranslationNormal, ResidualModelFrameVelocityTangential
import crocoddyl
import pinocchio as pin
import numpy as np
from robot_env import create_go2_env, create_go2_env_force_MJ
import mujoco
from utils import change_convention_pin2mj, change_convention_mj2pin, random_go2_x_u, stateMapping_mj2pin, stateMapping_pin2mj
pin_env = create_go2_env()
rmodel = pin_env["rmodel"]

mj_env = create_go2_env_force_MJ()
q0 = mj_env["q0"]
v0 = mj_env["v0"]
nu = mj_env["nu"]
fids = pin_env["contactFids"]
njoints = mj_env["njoints"]
ncontacts = mj_env["ncontacts"]
nq = mj_env["nq"]
nv = mj_env["nv"]
mj_model = mj_env["mj_model"]
mj_data = mujoco.MjData(mj_model)
ControlLimit = np.array(4 * [23.7, 23.7, 45.43])

print(pin.SE3ToXYZQUAT(pin.SE3().Identity()))

formatter = {'float_kind': lambda x: "{:.4f}".format(x)} 
np.set_printoptions(threshold=np.inf, linewidth=400, precision=5, suppress=False, formatter=formatter)

eps = 1e-8

def numdiffSE3toEuclidian(f, x0, rmodel, h=1e-6):
    f0 = f(x0).copy() # x0 in Mujoco
    Fx = []
    nq, nv = rmodel.nq, rmodel.nv
    q0_mj, v0_mj = f0[:nq].copy(), f0[-nv:].copy()
    dx = np.zeros(2 * nv)
    for ix in range(len(dx)):
       
        dx[ix] += h
        x_prime = np.hstack((pin.integrate(rmodel, x0[:nq].copy(), dx[:nv]), x0[nq:].copy() + dx[nv:]))
        f_prime = f(x_prime).copy() # x_prime in Mujoco
        q_prime_mj, v_prime_mj = f_prime[:nq].copy(), f_prime[-nv:].copy()
        # tmp = (f_prime - f0) / h
        diff_q = pin.difference(rmodel, q_prime_mj, q0_mj)
        diff_v = v_prime_mj - v0_mj
        diff = np.hstack((diff_q, diff_v)) / h
        Fx.append(diff)
        dx[ix] = 0.0

    Fx = np.array(Fx).T
    return Fx

def NumDiff_stateMappingDerivative_pin2mj(x_pin, rmodel):
    nq, nv = rmodel.nq, rmodel.nv
    x_mj, M, M_inv = stateMapping_pin2mj(x_pin, rmodel)
    x_mj = np.hstack((x_mj[:nq], x_mj[-nv:]))

    def f(x):
        x_mj, M, _ = stateMapping_pin2mj(x, rmodel)
        return x_mj

    dM_dx = numdiffSE3toEuclidian(f, x_pin, rmodel)
    return dM_dx

x = np.hstack((pin.randomConfiguration(rmodel, -np.pi/2 * np.ones(nq), np.pi/2 * np.ones(nq)), np.random.rand(nv)))
u = np.random.rand(nu) * ControlLimit
x_pin = x.copy()

dMx_dx = NumDiff_stateMappingDerivative_pin2mj(x_pin, rmodel)
x_pin = x_pin.reshape((nq+nv, 1))
print(f'dMx_dx: {dMx_dx.shape}')
print(f'dMx_dx: {dMx_dx}')
# print(f'dM_dx: {dM_dx}')
import pdb; pdb.set_trace()
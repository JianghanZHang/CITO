# Test calcDiff function in DifferentialActionModelForceExplicit with numerical differentiation results for floating base robots.
# Author: Jianghan Zhang
import sys
import os

# Add the outer directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import crocoddyl
import pinocchio as pin
import numpy as np
from differential_model_force_free import DifferentialActionModelForceExplicit
from friction_cone import  ResidualLinearFrictionCone
from complementarity_constraints_force_free import ResidualModelComplementarityErrorNormal, ResidualModelFrameTranslationNormal, ResidualModelComplementarityErrorTangential
from differential_model_NumDiff import DifferentialActionModelForceExplicit_NoCalcDiff
from differential_model_force_MJ import DifferentialActionModelForceMJ
import mujoco
from robot_env import create_go2_env_force_MJ, create_go2_env

pin_env = create_go2_env()
env = create_go2_env_force_MJ()

rmodel = pin_env["rmodel"]
nu = env["nu"]
njoints = env["njoints"]
ncontacts = env["ncontacts"]
contactFids = env["contactFids"]
mj_model = env["mj_model"]
mj_data = env["mj_data"]

dt = mj_model.opt.timestep

def numdiffSE3toEuclidian(f, x0, rmodel, h=1e-6):
    f0 = f(x0).copy()
    Fx = []
    nq, nv = rmodel.nq, rmodel.nv
    dx = np.zeros(2 * nv)
    for ix in range(len(dx)):
        dx[ix] += h
        x_prime = np.hstack((pin.integrate(rmodel, x0[:nq].copy(), dx[:nv]), x0[nq:].copy() + dx[nv:]))
        f_prime = f(x_prime).copy()
        tmp = (f_prime - f0) / h
        Fx.append(tmp)
        dx[ix] = 0.0

    Fx = np.array(Fx).T
    return Fx

def calcDiff_numdiff(x, u, model, rmodel, h=1e-6):
    # Create a fresh data collector for the nominal calculation
    data = model.createData()
    
    # Compute residual at nominal point
    model.calc(data, x, u)
    xout_nominal = data.xout.copy()
    
    # Function to compute residual for given x
    def dynamics_func(x_):
        data_local = model.createData()
        model.calc(data_local, x_, u)
        return data_local.xout.copy()
    
    # Compute numerical derivatives with respect to x
    Fx_num = numdiffSE3toEuclidian(dynamics_func, x, rmodel, h)
    
    # Compute numerical derivatives with respect to u (classical numdiff)
    Fu_num = np.zeros((len(xout_nominal), len(u)))
    for i in range(len(u)):
        u_eps = np.copy(u)
        u_eps[i] += h
        data_local = model.createData()
        model.calc(data_local, x, u_eps)
        xout_eps = data_local.xout.copy()
        Fu_num[:, i] = (xout_eps - xout_nominal) / h
    
    return Fx_num, Fu_num


def mujoco_forward(mj_model, mj_data, q, v, u):
    mj_model.opt.timestep = dt
    mj_model.opt.integrator = 0
    mj_data.qpos[:] = q
    mj_data.qvel[:] = v
    mj_data.ctrl[:] = u
    mujoco.mj_forward(mj_model, mj_data)
    nq = mj_model.nq
    nv = mj_model.nv
    nx = nq + nv
    nu = mj_model.nu
    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    mujoco.mjd_transitionFD(mj_model, mj_data, eps = 1e-6, flag_centered = 1, A = A, B = B, C = None, D = None)
    return mj_data.qacc, A, B

def mujoco_forward_diff(mj_model, mj_data, x, u):
    mj_model.opt.solver = 0 # Using Newton
    mj_model.opt.timestep = dt
    mj_model.opt.integrator = 0 # Using Semi-implicit Euler
    mj_model.opt.tolerance = 0.

    nq = mj_model.nq
    nv = mj_model.nv
    nu = mj_model.nu
    
    mj_data.qpos[:] = x[:nq]
    mj_data.qvel[:] = x[nq:]
    mj_data.ctrl[:] = u
    mujoco.mj_forward(mj_model, mj_data)
    
    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    mujoco.mjd_transitionFD(mj_model, mj_data, eps = 1e-8, flg_centered = 1, A = A, B = B, C = None, D = None)
    da_dq = A[nv:, :nv] / dt
    da_dv = (A[nv:, nv:] - np.eye(nv))/ dt
    da_du = B[nv:, :] / dt
    da_dx = np.hstack((da_dq, da_dv))

    return mj_data.qacc, da_dx, da_du

def test_DAM(numerical_model, rmodel, mj_model, mj_data):
    state = numerical_model.state
    nq, nv = rmodel.nq, rmodel.nv
    numerical_data = numerical_model.createData()

    
    # Random test state and control input
    x = np.hstack((pin.randomConfiguration(rmodel, -np.pi/2 * np.ones(nq), np.pi/2 * np.ones(nq)), np.random.rand(nv)))
    key_qpos = mj_model.key_qpos
    # import pdb; pdb.set_trace() 

    # x = np.hstack((key_qpos, np.random.rand(1, nv))).reshape((37,))
    u = np.random.rand(numerical_model.nu)
    # u = np.zeros_like(u)
    print(f'Test state: {x}')
    print(f'Test control: {u}')
    # Compute analytical derivatives
    a_analytical = np.zeros(2*nv)
    Fx_analytical = np.zeros((numerical_model.state.ndx, numerical_model.state.ndx))
    Fu_analytical = np.zeros((numerical_model.state.ndx, numerical_model.nu))
    a_analytical, Fx_analytical, Fu_analytical = mujoco_forward_diff(mj_model, mj_data, x, u)
    
    
    # Compute numerical derivatives
    # numerical_model.calc(numerical_data, x.copy(), u.copy())
    # numerical_model.calcDiff(numerical_data, x.copy(), u.copy())
    numerical_model.calc(numerical_data, x, u)
    a_numerical = numerical_data.xout
    
    Fx_numerical, Fu_numerical = calcDiff_numdiff(x, u, numerical_model, rmodel)

    
    diff_Fx = Fx_numerical - Fx_analytical
    diff_Fu = Fu_numerical - Fu_analytical
    diff_a = a_numerical - a_analytical
    
    print(f'a_numerical: \n {a_numerical}')
    print(f'a_analytical: \n {a_analytical}')

    # Compare analytical and numerical derivatives
    if not np.allclose(diff_a, np.zeros_like(diff_a), atol=1e-5) or not np.allclose(diff_Fx, np.zeros_like(diff_Fx), atol=1e-6) or not np.allclose(diff_Fu, np.zeros_like(diff_Fu), atol=1e-6):
        
        print(f'a difference: \n {diff_a}')
        print(f'a diff norm:\n {np.linalg.norm(diff_a)}')

        print(f'Fx difference: \n {diff_Fx}')
        print(f'Fx diff norm:\n {np.linalg.norm(diff_Fx)}')
        
        print(f'Fu difference:\n {diff_Fu}')
        print(f'Fu diff norm:\n {np.linalg.norm(diff_Fu)}')


    # assert np.allclose(diff_Fx, np.zeros_like(diff_Fx), atol=1e-3), f"Fx mismatch: \nAnalytical:\n{Fx_analytical}\nNumerical:\n{Fx_numerical}"
    # assert np.allclose(diff_Fu, np.zeros_like(diff_Fu), atol=1e-3), f"Fu mismatch: \nAnalytical:\n{Fu_analytical}\nNumerical:\n{Fu_numerical}"

    # print("Test passed!")

formatter = {'float_kind': lambda x: "{:.4f}".format(x)}
    
np.set_printoptions(linewidth=210, precision=5, suppress=False, formatter=formatter)

# Example usage with the DifferentialActionModelForceExplicit
state = crocoddyl.StateMultibody(rmodel)

cost_model = crocoddyl.CostModelSum(state, nu)

constraintModelManager = crocoddyl.ConstraintModelManager(state, nu)

# # # Friction cone constraints
# for idx, fid in enumerate(env["contactFids"]):
#     FrictionConeResidual = ResidualLinearFrictionCone(state, njoints, ncontacts, nu, fid, idx, mu = 0.9)
#     FrictionConeConstraint = crocoddyl.ConstraintModelResidual(state, FrictionConeResidual, np.zeros(5), np.inf * np.ones(5))
#     constraintModelManager.addConstraint("FrictionConeConstraint_"+ env["contactFnames"][idx] + str(idx), FrictionConeConstraint)

# # # Complementarity constraints
# # No penetration constraint
# for idx, fid in enumerate(env["contactFids"]):
#     ZResidual = ResidualModelFrameTranslationNormal(state, nu, fid)
#     ZConstraint = crocoddyl.ConstraintModelResidual(state, ZResidual, -1e-3 * np.ones(1), np.inf*np.ones(1))
#     constraintModelManager.addConstraint("ZConstraint_"+ env["contactFnames"][idx] + str(idx), ZConstraint)

# for idx, fid in enumerate(env["contactFids"]):
#     ComplementarityResidual = ResidualModelComplementarityErrorNormal(state, nu, fid, idx, njoints)
#     ComplementarityConstraint = crocoddyl.ConstraintModelResidual(state, ComplementarityResidual, -np.inf *np.ones(1), 1e-2 * np.ones(1))
#     constraintModelManager.addConstraint("ComplementarityConstraintNormal_"+ env["contactFnames"][idx] + str(idx), ComplementarityConstraint)

# for idx, fid in enumerate(env["contactFids"]):
#     ComplementarityResidual = ResidualModelComplementarityErrorTangential(state, nu, fid, idx, njoints)
#     ComplementarityConstraint = crocoddyl.ConstraintModelResidual(state, ComplementarityResidual, -1e-2 *np.ones(6), 1e-2 * np.ones(6))
#     constraintModelManager.addConstraint("ComplementarityConstraintTangential_"+ env["contactFnames"][idx] + str(idx), ComplementarityConstraint)

print("Testing DifferentialActionModelForceMJ")
DAM_no_calc_diff = DifferentialActionModelForceMJ(mj_model, mj_data, state, nu, njoints, contactFids, cost_model)
                                                 
DAM_numerical = crocoddyl.DifferentialActionModelNumDiff(DAM_no_calc_diff, True)
test_DAM(DAM_numerical, rmodel, mj_model, mj_data)

'''
Test state: [1.0687 -0.3318 0.8894 0.3815 0.8455 0.2892 -0.2366 -0.5177 0.8427 -0.6981 0.1696 -0.0710 0.4049 -0.4248 0.0421 1.4207 1.3075 0.4264 0.6827 0.8630 0.7604 0.5086 0.5175 0.5282 0.5602 0.2512 0.8426 0.1299 0.4830
 0.7772 0.5389 0.6985 0.8436 0.4608 0.3294 0.8526 0.2088]
Test control: [0.3006 0.0846 0.2669 0.7285 0.3089 0.4565 0.6249 0.8999 0.0832 0.2781 0.6626 0.3864]

Test state: [1.0687 -0.3318 0.8894 0.3815 0.8455 0.2892 -0.2366 -0.5177 0.8427 -0.6981 0.1696 -0.0710 0.4049 -0.4248 0.0421 1.4207 1.3075 0.4264 0.6827 0.8603 0.4313 0.5337 0.9937 0.5440 0.6510 0.6066 0.2462 0.6862 0.4548
 0.7979 0.9324 0.9338 0.0182 0.0566 0.8083 0.8853 0.3603]
Test control: [0.9989 0.2540 0.1728 0.7494 0.1320 0.2154 0.3781 0.4849 0.9614 0.8272 0.5919 0.6214]


'''
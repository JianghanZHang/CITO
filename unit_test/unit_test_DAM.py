# Test calcDiff function in DifferentialActionModelForceExplicit with numerical differentiation results for floating base robots.
# Author: Jianghan Zhang
import sys
import os
from robot_model import *

# Add the outer directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import crocoddyl
import pinocchio as pin
import numpy as np
from differential_model_force_free import DifferentialActionModelForceExplicit
from friction_cone import  ResidualLinearFrictionCone
from complementarity_constraints_force_free import ResidualModelComplementarityErrorNormal, ResidualModelFrameTranslationNormal, ResidualModelComplementarityErrorTangential


eps = 1e-6

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

def test_DAM(model: crocoddyl.DifferentialActionModelAbstract, rmodel):
    state = model.state
    nq, nv = rmodel.nq, rmodel.nv
    data = model.createData()
    
    # Random test state and control input
    x = np.hstack((pin.randomConfiguration(rmodel, -np.pi/2 * np.ones(nq), np.pi/2 * np.ones(nq)), np.random.rand(nv)))
    u = np.random.rand(model.nu)
    
    # Compute analytical derivatives
    model.calcDiff(data, x, u)
    Fx_analytical = data.Fx
    Fu_analytical = data.Fu
    
    # Compute numerical derivatives
    Fx_numerical, Fu_numerical = calcDiff_numdiff(x, u, model, rmodel)
    
    relative_error_Fx = np.where(Fx_numerical < 1e-9, Fx_analytical, (Fx_analytical - Fx_numerical) / Fx_numerical)
    relative_error_Fx = np.where(relative_error_Fx == -1.0, 0.0, relative_error_Fx)
    relative_error_Fu = np.where(Fu_numerical < 1e-9, Fu_analytical, (Fu_analytical - Fu_numerical) / Fu_numerical)
    diff_Fx = Fx_numerical - Fx_analytical
    diff_Fu = Fu_numerical - Fu_analytical
    
    
    # Compare analytical and numerical derivatives
    if not np.allclose(diff_Fx, np.zeros_like(diff_Fx), atol=1e-3) or not np.allclose(diff_Fu, np.zeros_like(diff_Fu), atol=1e-3):
        # print(f'Fx relative error: \n {relative_error_Fx}')
        print(f'Fx difference: \n {diff_Fx}')
        print(f'Fx diff norm:\n {np.linalg.norm(diff_Fx)}')
        
        # print(f'Fu relative error: \n {relative_error_Fu}')
        print(f'Fu difference:\n {diff_Fu}')
        print(f'Fu diff norm:\n {np.linalg.norm(diff_Fu)}')

    assert np.allclose(diff_Fx, np.zeros_like(diff_Fx), atol=1e-3), f"Fx mismatch: \nAnalytical:\n{Fx_analytical}\nNumerical:\n{Fx_numerical}"
    assert np.allclose(diff_Fu, np.zeros_like(diff_Fu), atol=1e-3), f"Fu mismatch: \nAnalytical:\n{Fu_analytical}\nNumerical:\n{Fu_numerical}"

    print("Test passed!")

formatter = {'float_kind': lambda x: "{:.4f}".format(x)}
    
np.set_printoptions(linewidth=210, precision=5, suppress=False, formatter=formatter)

# Example usage with the DifferentialActionModelForceExplicit
state = crocoddyl.StateMultibody(rmodel)

cost_model = crocoddyl.CostModelSum(state, nu)

constraintModelManager = crocoddyl.ConstraintModelManager(state, nu)

# # Friction cone constraints
for idx, fid in enumerate(env["contactFids"]):
    FrictionConeResidual = ResidualLinearFrictionCone(state, njoints, ncontacts, nu, fid, idx, mu = 0.9)
    FrictionConeConstraint = crocoddyl.ConstraintModelResidual(state, FrictionConeResidual, np.zeros(5), np.inf * np.ones(5))
    constraintModelManager.addConstraint("FrictionConeConstraint_"+ env["contactFnames"][idx] + str(idx), FrictionConeConstraint)

# # Complementarity constraints
# No penetration constraint
for idx, fid in enumerate(env["contactFids"]):
    ZResidual = ResidualModelFrameTranslationNormal(state, nu, fid)
    ZConstraint = crocoddyl.ConstraintModelResidual(state, ZResidual, -1e-3 * np.ones(1), np.inf*np.ones(1))
    constraintModelManager.addConstraint("ZConstraint_"+ env["contactFnames"][idx] + str(idx), ZConstraint)

for idx, fid in enumerate(env["contactFids"]):
    ComplementarityResidual = ResidualModelComplementarityErrorNormal(state, nu, fid, idx, njoints)
    ComplementarityConstraint = crocoddyl.ConstraintModelResidual(state, ComplementarityResidual, -np.inf *np.ones(1), 1e-2 * np.ones(1))
    constraintModelManager.addConstraint("ComplementarityConstraintNormal_"+ env["contactFnames"][idx] + str(idx), ComplementarityConstraint)

for idx, fid in enumerate(env["contactFids"]):
    ComplementarityResidual = ResidualModelComplementarityErrorTangential(state, nu, fid, idx, njoints)
    ComplementarityConstraint = crocoddyl.ConstraintModelResidual(state, ComplementarityResidual, -1e-2 *np.ones(4), 1e-2 * np.ones(4))
    constraintModelManager.addConstraint("ComplementarityConstraintTangential_"+ env["contactFnames"][idx] + str(idx), ComplementarityConstraint)


print("Testing DifferentialActionModelForceExplicit")
differential_action_model_force_explicit = DifferentialActionModelForceExplicit(state, nu, njoints, contactFids, cost_model, constraintModelManager)
test_DAM(differential_action_model_force_explicit, rmodel)

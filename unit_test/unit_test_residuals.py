# Test calcDiff function in ResidualModels with numerical differentiation results for floaing base robots.
# Author: Jianghan Zhang
import sys
import os
from robot_model import *

# Add the outer directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from numerical_differences import numdiffSE3toEuclidian
from complementarity_constraints_force_free import ResidualModelContactForceNormal, ResidualModelComplementarityErrorNormal, ResidualModelComplementarityErrorTangential, ResidualModelFrameTranslationNormal
from friction_cone import ResidualLinearFrictionCone
import crocoddyl
import pinocchio as pin
import numpy as np

eps = 1e-6

def numdiffSE3toEuclidian(f, x0, rmodel, h=1e-6):
    f0 = f(x0).copy()
    Fx = []
    nq, nv = rmodel.nq, rmodel.nv
    dx = np.zeros(2 * nv)
    # q_prime = np.zeros(nq)
    for ix in range(len(dx)):
       
        dx[ix] += h
        # pin.integrate(rmodel, x0[:nq], dx[:nv], q_prime)
        # x_prime = np.hstack((q_prime, x0[nq:] + dx[nv:]))
        x_prime = np.hstack((pin.integrate(rmodel, x0[:nq].copy(), dx[:nv]), x0[nq:].copy() + dx[nv:]))
        # x_bar = np.hstack((pin.difference(rmodel, x0[:nq], -dx[:nv]), x0[nq:] + dx[nv:]))
        f_prime = f(x_prime).copy()
        tmp = (f_prime - f0) / h

        Fx.append(tmp)
        dx[ix] = 0.0

    Fx = np.array(Fx).T
    return Fx

def calcDiff_numdiff(x, u, ResidualModel: crocoddyl.ResidualModelAbstract, rmodel, h=1e-6):
    # Create a fresh data collector for the nominal calculation
    collector = crocoddyl.DataCollectorMultibody(pin.Data(ResidualModel.state.pinocchio))
    data = ResidualModel.createData(collector)
    
    # Compute residual at nominal point
    ResidualModel.calc(data, x, u)
    r_nominal = data.r.copy()
    # print(f"Residual at nominal point r_nominal: {r_nominal}")
    
    # Function to compute residual for given x
    def residual_func(x_):
        data_local = ResidualModel.createData(collector)
        ResidualModel.calc(data_local, x_, u)
        return data_local.r.copy()
    
    # Compute numerical derivatives with respect to x
    Rx_num = numdiffSE3toEuclidian(residual_func, x, rmodel, h)
    
    # Compute numerical derivatives with respect to u (classical numdiff)
    Ru_num = np.zeros((len(r_nominal), len(u)))
    for i in range(len(u)):
        u_eps = np.copy(u)
        u_eps[i] += h
        data_local = ResidualModel.createData(collector)
        ResidualModel.calc(data_local, x, u_eps)
        r_eps = data_local.r.copy()
        Ru_num[:, i] = (r_eps - r_nominal) / h
    
    # print(f"Numerical derivatives Rx_num: {Rx_num}")
    # print(f"Numerical derivatives Ru_num: {Ru_num}")
    return Rx_num, Ru_num


def test_residual(ResidualModel: crocoddyl.ResidualModelAbstract, rmodel):
    state = ResidualModel.state
    collector = crocoddyl.DataCollectorMultibody(pin.Data(ResidualModel.state.pinocchio))
    data = ResidualModel.createData(collector)
    
    # Random test state and control input
    x = np.hstack((pin.randomConfiguration(rmodel, -np.pi/2 * np.ones(nq), np.pi/2 * np.ones(nq)), np.random.rand(nv)))
    u = np.random.rand(ResidualModel.nu)
    
    # Compute analytical derivatives
    ResidualModel.calcDiff(data, x, u)
    Rx_analytical = data.Rx
    Ru_analytical = data.Ru
    
    # Compute numerical derivatives
    Rx_numerical, Ru_numerical = calcDiff_numdiff(x, u, ResidualModel, rmodel)
    
    # Compare analytical and numerical derivatives

    assert np.allclose(Rx_analytical, Rx_numerical, atol=1e-6), f"Rx mismatch: \nAnalytical:\n{Rx_analytical}\nNumerical:\n{Rx_numerical}" 
    assert np.allclose(Ru_analytical, Ru_numerical, atol=1e-6), f"Ru mismatch: \nAnalytical:\n{Ru_analytical}\nNumerical:\n{Ru_numerical}"

    
    print("Test passed!")


# Example usage with your residual models
state = crocoddyl.StateMultibody(rmodel)
idx = 0
fid = contactFids[idx]

print("Testing ResidualLinearFrictionCone")
residual_model_linear_cone = ResidualLinearFrictionCone(state, njoints, ncontacts, nu, fid, idx, mu = 0.9)
test_residual(residual_model_linear_cone, rmodel)

print("Testing ResidualModelFrameTranslationNormal")
residual_model_translation_normal = ResidualModelFrameTranslationNormal(state, nu, fid)
test_residual(residual_model_translation_normal, rmodel)

print("Testing ResidualModelComplementarityErrorNormal")
residual_model_complementarity_normal = ResidualModelComplementarityErrorNormal(state, nu, fid, idx, njoints)
test_residual(residual_model_complementarity_normal, rmodel)

print("Testing ResidualModelComplementarityErrorTangential")
residual_model_complementarity_tangential = ResidualModelComplementarityErrorTangential(state, nu, fid, idx, njoints)
test_residual(residual_model_complementarity_tangential, rmodel)

print("Testing ResidualModelContactForceNormal")
residual_model_force_normal = ResidualModelContactForceNormal(state, nu, fid, idx, njoints)
test_residual(residual_model_force_normal, rmodel)


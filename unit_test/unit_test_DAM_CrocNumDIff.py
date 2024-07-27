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
from differential_model_NumDiff import DifferentialActionModelForceExplicit_NoCalcDiff

def test_DAM(analytical_model: crocoddyl.DifferentialActionModelAbstract, numerical_model: crocoddyl.DifferentialActionModelAbstract, rmodel):
    state = analytical_model.state
    nq, nv = rmodel.nq, rmodel.nv
    analytical_data = analytical_model.createData()
    numerical_data = numerical_model.createData()

    
    # Random test state and control input
    x = np.hstack((pin.randomConfiguration(rmodel, -np.pi/2 * np.ones(nq), np.pi/2 * np.ones(nq)), np.random.rand(nv)))
    u = np.random.rand(analytical_model.nu)
    
    # Compute analytical derivatives
    analytical_model.calcDiff(analytical_data, x, u)
    analytical_model.calcDiff(analytical_data, x, u)
    Fx_analytical = analytical_data.Fx
    Fu_analytical = analytical_data.Fu
    
    # Compute numerical derivatives
    numerical_model.calc(numerical_data, x, u)
    numerical_model.calcDiff(numerical_data, x, u)
    Fx_numerical = numerical_data.Fx
    Fu_numerical = numerical_data.Fu

    
    diff_Fx = Fx_numerical - Fx_analytical
    diff_Fu = Fu_numerical - Fu_analytical
    
    
    # Compare analytical and numerical derivatives
    if not np.allclose(diff_Fx, np.zeros_like(diff_Fx), atol=1e-8) or not np.allclose(diff_Fu, np.zeros_like(diff_Fu), atol=1e-8):
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
DAM_analytical = DifferentialActionModelForceExplicit(state, nu, njoints, contactFids, cost_model, constraintModelManager)
DAM_no_calc_diff = DifferentialActionModelForceExplicit_NoCalcDiff(state, nu, njoints, contactFids, cost_model, constraintModelManager)
DAM_numerical = crocoddyl.DifferentialActionModelNumDiff(DAM_no_calc_diff, True)
test_DAM(DAM_analytical, DAM_numerical, rmodel)

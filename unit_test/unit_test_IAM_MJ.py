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
from differential_model_force_MJ import DifferentialActionModelForceMJ
from integrated_action_model_MJ import IntegratedActionModelForceMJ #, DummyDifferentialActionModelMJ
from robot_env import create_go2_env_force_MJ, create_go2_env

formatter = {'float_kind': lambda x: "{:.4f}".format(x)} 
np.set_printoptions(threshold=np.inf, linewidth=300, precision=5, suppress=False, formatter=formatter)

abduction_lb = -1.0472; abduction_ub = 1.0472; front_hip_lb = -1.5708; front_hip_ub = 3.4807; knee_lb = -2.7227; knee_ub = -0.83776; back_hip_lb = -1.5708; back_hip_ub = 4.5379

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

StateLimit_ub = np.array([np.inf, np.inf, np.inf, 
            np.inf, np.inf, np.inf, np.inf,
            abduction_ub, front_hip_ub, knee_ub, 
            abduction_ub, front_hip_ub, knee_ub, 
            abduction_ub, back_hip_ub, knee_ub, 
            abduction_ub, back_hip_ub, knee_ub] 
            + 18 * [np.inf])

StateLimit_lb = np.array([-np.inf, -np.inf, -np.inf, 
            -np.inf, -np.inf, -np.inf, -np.inf,
            abduction_lb, front_hip_lb, knee_lb, 
            abduction_lb, front_hip_lb, knee_lb, 
            abduction_lb, back_hip_lb, knee_lb, 
            abduction_lb, back_hip_lb, knee_lb]
            + 18 * [-np.inf])

ControlLimit = np.array(4 * [23.7, 23.7, 45.43])

def test_IAM(analytical_model: crocoddyl.IntegratedActionModelAbstract, numerical_model: crocoddyl.IntegratedActionModelAbstract, rmodel):
    state = analytical_model.state
    nq, nv = rmodel.nq, rmodel.nv
    analytical_data = analytical_model.createData()
    numerical_data = numerical_model.createData()

    
    # Random test state and control input
    # x = np.hstack((pin.randomConfiguration(rmodel, -np.pi/2 * np.ones(nq), np.pi/2 * np.ones(nq)), np.random.rand(nv)))
    x = np.hstack((env["q0"], env["v0"]))   
    u = np.random.rand(analytical_model.nu) * ControlLimit
    
    # Compute analytical derivatives
    analytical_model.calc(analytical_data, x, u)
    analytical_model.calcDiff(analytical_data, x, u)
    Acceleration_analytical = analytical_data.differential.xout
    Fx_analytical = analytical_data.Fx
    Fu_analytical = analytical_data.Fu
    
    # Compute numerical derivatives
    numerical_model.calc(numerical_data, x, u)
    numerical_model.calcDiff(numerical_data, x, u)
    Acceleration_numerical = numerical_data.differential.xout
    Fx_numerical = numerical_data.Fx
    Fu_numerical = numerical_data.Fu

    diff_Acceleration = Acceleration_numerical - Acceleration_analytical
    diff_Fx = Fx_numerical - Fx_analytical
    diff_Fu = Fu_numerical - Fu_analytical
    
    
    # Compare analytical and numerical derivatives
    # if not np.allclose(diff_Fx, np.zeros_like(diff_Fx), atol=1e-8) or not np.allclose(diff_Fu, np.zeros_like(diff_Fu), atol=1e-8):
        # print(f'Fx relative error: \n {relative_error_Fx}')
    # print(f'Fx difference: \n {diff_Fx}')

    # print(f'Fu numerical:\n {Fu_numerical}')
    # print(f'Fu analytical:\n {Fu_analytical}')
    
    print(f'Acceleration difference:\n {diff_Acceleration}')
    print(f'Acceleration diff norm:\n {np.linalg.norm(diff_Acceleration)}')
    
    print(f'Fu difference:\n {diff_Fu}')
    print(f'Fu diff norm:\n {np.linalg.norm(diff_Fu)}')
    
    print(f'X: \n {x}')
    print(f'dq_t+1/dq_t difference: \n {diff_Fx[:nq, :nq]}' )
    print(f'dq_t+1/dq_t difference norm: \n {np.linalg.norm(diff_Fx[:nq, :nq])} \n' )

    print(f'dq_t+1/dv_t difference: \n {diff_Fx[:nq, nq:]}' )
    print(f'dq_t+1/dv_t difference norm: \n {np.linalg.norm(diff_Fx[:nq, nq:])} \n' )

    print(f'dv_t+1/dq_t difference: \n {diff_Fx[nq:, :nq]}' )
    print(f'dv_t+1/dq_t difference norm: \n {np.linalg.norm(diff_Fx[nq:, :nq])} \n' )

    print(f'dv_t+1/dv_t difference: \n {diff_Fx[nq:, nq:]}' )
    print(f'dv_t+1/dv_t difference norm: \n {np.linalg.norm(diff_Fx[nq:, nq:])} \n' )

    print(f'Fx diff norm:\n {np.linalg.norm(diff_Fx)}')

        # print(f'Fu relative error: \n {relative_error_Fu}')
    
    # assert np.allclose(diff_Fx, np.zeros_like(diff_Fx), atol=1e-3), f"Fx mismatch: \nAnalytical:\n{Fx_analytical}\nNumerical:\n{Fx_numerical}"
    # assert np.allclose(diff_Fu, np.zeros_like(diff_Fu), atol=1e-3), f"Fu mismatch: \nAnalytical:\n{Fu_analytical}\nNumerical:\n{Fu_numerical}"

    # print("Test passed!")


# Example usage with the DifferentialActionModelForceExplicit
state = crocoddyl.StateMultibody(rmodel)

cost_model = crocoddyl.CostModelSum(state, nu)

constraintModelManager = crocoddyl.ConstraintModelManager(state, nu)

StateResidual = crocoddyl.ResidualModelState(state, np.zeros(37), nu)
StateLimitConstraint = crocoddyl.ConstraintModelResidual(state, StateResidual, StateLimit_lb, StateLimit_ub)
constraintModelManager.addConstraint("StateLimitConstraint", StateLimitConstraint)

# # Control limits
ControlLimit = np.array(4 * [23.7, 23.7, 45.43])
ControlRedisual = crocoddyl.ResidualModelControl(state, nu)
ControlLimitConstraint = crocoddyl.ConstraintModelResidual(state, ControlRedisual, -ControlLimit, ControlLimit)
constraintModelManager.addConstraint("ControlLimitConstraint", ControlLimitConstraint)

print("Testing DifferentialActionModelForceExplicit")
DAM = DifferentialActionModelForceMJ(mj_model, mj_data, state, nu, njoints, contactFids, cost_model, constraintModelManager)
IAM_analytical = IntegratedActionModelForceMJ(DAM, dt, True)
IAM_numerical = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelNumDiff(DAM, True), dt)
test_IAM(IAM_analytical, IAM_numerical, rmodel)

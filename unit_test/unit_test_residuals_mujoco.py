# Test calcDiff function in ResidualModels with numerical differentiation results for floaing base robots.
# Author: Jianghan Zhang
import sys
import os
import argparse
# Add the outer directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from friction_cone import ResidualLinearFrictionCone
from ResidualModels import ResidualModelFrameTranslationNormal, ResidualModelFrameVelocityTangential
import crocoddyl
import pinocchio as pin
import numpy as np
from robot_env import create_go2_env, create_go2_env_force_MJ
import mujoco
from utils import change_convention_pin2mj, change_convention_mj2pin, stateMapping_mj2pin, stateMapping_pin2mj
from numerical_difference import stateMappingDerivative_pin2mj_numDiff, stateMappingDerivative_mj2pin_numDiff
from ResidualModels import ResidualModelFrameVelocityTangentialNumDiff
from ResidualModels import ResidualModelFootClearance, ResidualModelFootClearanceNumDiff
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
ControlLimit = np.array(4 * [23.7, 23.7, 45.43])

print(pin.SE3ToXYZQUAT(pin.SE3().Identity()))

formatter = {'float_kind': lambda x: "{:.4f}".format(x)} 
np.set_printoptions(threshold=np.inf, linewidth=400, precision=5, suppress=False, formatter=formatter)


def _numdiffSE3toEuclidian(f, x0, rmodel, h=1e-8):
    f0 = f(x0).copy()
    Fx = []
    nq, nv = rmodel.nq, rmodel.nv
    dx = np.zeros(2 * nv)
    # q_prime = np.zeros(nq)
    for ix in range(len(dx)):
       
        dx[ix] += h
        x_prime = np.hstack((pin.integrate(rmodel, x0[:nq].copy(), dx[:nv]), x0[nq:].copy() + dx[nv:]))

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
    Rx_num = _numdiffSE3toEuclidian(residual_func, x, rmodel, h)
    
    # Compute numerical derivatives with respect to u (classical numdiff)
    Ru_num = np.zeros((len(r_nominal), len(u)))
    for i in range(len(u)):
        u_eps = np.copy(u)
        u_eps[i] += h
        data_local = ResidualModel.createData(collector)
        ResidualModel.calc(data_local, x, u_eps)
        r_eps = data_local.r.copy()
        Ru_num[:, i] = (r_eps - r_nominal) / h
    
    return Rx_num, Ru_num


def test_residual_translation(ResidualModel: crocoddyl.ResidualModelAbstract, ResidualModelCroc, rmodel, mj_model):
    mj_data = mujoco.MjData(mj_model)
    mj_data.qacc = np.zeros(nv)

    state = ResidualModel.state
    collector = crocoddyl.DataCollectorMultibody(pin.Data(ResidualModel.state.pinocchio))
    data = ResidualModel.createData(collector)
    
    # Random test state and control input
    x = np.hstack((pin.randomConfiguration(rmodel, -np.pi/2 * np.ones(nq), np.pi/2 * np.ones(nq)), np.random.rand(nv)))
    x[:3] = 50.
    # x[19:22] = 0.0
    u = np.random.rand(nu) * ControlLimit
    x_pin = x.copy()
    x_mj, M, _ = stateMapping_pin2mj(x, rmodel)
    print(f'x_pin: {x}')
    print(f'x_mj: {x_mj}')
    
    # Compute analytical derivatives
    ResidualModel.calc(data, x_pin, u)
    r_analytical = data.r
    ResidualModel.calcDiff(data, x_pin, u)
    Rx_analytical = data.Rx
    Rx_numerical, _ = calcDiff_numdiff(x_pin, u, ResidualModel, rmodel)

    collector = crocoddyl.DataCollectorMultibody(pin.Data(ResidualModelCroc.state.pinocchio))
    data_croc = ResidualModelCroc.createData(collector)
    
    pin.forwardKinematics(rmodel, data_croc.pinocchio, x_pin[:nq])
    pin.framesForwardKinematics(rmodel, data_croc.pinocchio, x_pin[:nq])
    pin.updateFramePlacements(rmodel, data_croc.pinocchio)
    r_croc = ResidualModelCroc.calc(data_croc, x_pin, u)
    pin.computeJointJacobians(rmodel, data_croc.pinocchio, x_pin[:nq])

    r_croc = data_croc.r[2]
    ResidualModelCroc.calcDiff(data_croc, x_pin, u)
    Rx_croc = data_croc.Rx[2, :]


    q_mj, v_mj = x_mj[:nq], x_mj[-nv:]
    mj_data.qpos = q_mj.copy()
    mj_data.qvel = v_mj.copy()
    # mj_data.ctrl = u.copy()
    # Compute numerical derivatives
    mujoco.mj_step1(mj_model, mj_data)
    r_numerical_MJ = mj_data.sensordata[3]
    ds_dq = np.zeros((nv, 28))
    ds_dv = np.zeros((nv, 28))
    mujoco.mjd_inverseFD(mj_model, mj_data, eps = 1e-8, flg_actuation = 1, DfDq=None, DfDv=None, DfDa=None, DsDq=ds_dq, DsDv=ds_dv, DsDa=None, DmDq=None)
    
    ds_dq = ds_dq.T[3, :]
    ds_dv = ds_dv.T[3, :]
    Rx_numerical_MJ_ = np.hstack((ds_dq, ds_dv))
    dx_mj_dx_pin = stateMappingDerivative_pin2mj_numDiff(x_pin, rmodel, mj_model)
    # dx_pin_dx_mj = stateMappingDerivative_mj2pin_numDiff(x_mj, rmodel, mj_model)

    Rx_numerical_MJ = Rx_numerical_MJ_ @ dx_mj_dx_pin

    # if not np.allclose(r_analytical, r_numerical_MJ, atol=1e-3):
    print("Analytical: ", r_analytical)
    print("Mujoco: ", r_numerical_MJ)
    # print("crocoddyl: ", r_croc)
    print("Difference: ", r_analytical - r_numerical_MJ)

    # if not np.allclose(Rx_analytical, Rx_numerical_MJ, atol=1e-3):
    print("Analytical:\n ", Rx_analytical)
    print("Numerical Python: \n", Rx_numerical)
    print("crocoddyl: \n", Rx_croc)
    print("Numerical Mujuco before mapping:\n ", Rx_numerical_MJ_)
    print("Numerical Mujuco after mapping: \n ", Rx_numerical_MJ)

    print(f'Difference before mapping:\n {Rx_numerical - Rx_numerical_MJ_}')
    print(f'Difference after mapping:\n {Rx_numerical - Rx_numerical_MJ}')
    
    # Compare analytical and numerical derivatives

    # print("Test passed!")

def test_residual_velocity(ResidualModel: crocoddyl.ResidualModelAbstract, ResidualModelCroc, rmodel, mj_model):
    mj_data = mujoco.MjData(mj_model)
    mj_data.qacc = np.zeros(nv)

    state = ResidualModel.state
    collector = crocoddyl.DataCollectorMultibody(pin.Data(ResidualModel.state.pinocchio))
    data = ResidualModel.createData(collector)
    
    # Random test state and control input
    x = np.hstack((pin.randomConfiguration(rmodel, -np.pi/2 * np.ones(nq), np.pi/2 * np.ones(nq)), np.random.rand(nv)))
    x[:3] = 50.
    quat = np.array([0, 0, 0, 1])
    quat = quat / np.linalg.norm(quat)
    x[3:7] = quat
    x[19:22] = 0.0
    u = np.random.rand(nu) * ControlLimit
    
    x_pin = x.copy()
    x_mj,M ,_ = stateMapping_pin2mj(x, rmodel)
    print(f'x_pin: {x}')
    print(f'x_mj: {x_mj}')
    
    # Compute analytical derivatives
    ResidualModel.calc(data, x_pin, u)
    r_analytical = data.r
    ResidualModel.calcDiff(data, x_pin, u)
    Rx_analytical = data.Rx

    Rx_numerical, _ = calcDiff_numdiff(x_pin, u, ResidualModel, rmodel)

    collector = crocoddyl.DataCollectorMultibody(pin.Data(ResidualModelCroc.state.pinocchio))
    data_croc = ResidualModelCroc.createData(collector)

    pin.forwardKinematics(rmodel, data_croc.pinocchio, x_pin[:nq], x_pin[nq:])
    pin.updateFramePlacements(rmodel, data_croc.pinocchio)
    r_croc = ResidualModelCroc.calc(data_croc, x_pin, u)
    pin.computeForwardKinematicsDerivatives(rmodel, data_croc.pinocchio, x_pin[:nq], x_pin[nq:], np.zeros(nv))
    r_croc = data_croc.r[0:2]
    ResidualModelCroc.calcDiff(data_croc, x_pin, u)
    Rx_croc = data_croc.Rx[0:2, :]


    q_mj, v_mj = x_mj[:nq], x_mj[-nv:]
    mj_data.qpos = q_mj.copy()
    mj_data.qvel = v_mj.copy()
    mj_data.ctrl = u.copy()
    # Compute numerical derivatives
    mujoco.mj_step1(mj_model, mj_data)
    r_numerical_MJ = mj_data.sensordata[4:6]
    ds_dq = np.zeros((nv, 28))
    ds_dv = np.zeros((nv, 28))
    mujoco.mjd_inverseFD(mj_model, mj_data, eps = 1e-8, flg_actuation = 1, DfDq=None, DfDv=None, DfDa=None, DsDq=ds_dq, DsDv=ds_dv, DsDa=None, DmDq=None)
    
    ds_dq = ds_dq.T[4:6, :]
    ds_dv = ds_dv.T[4:6, :]
    Rx_numerical_MJ_ = np.hstack((ds_dq, ds_dv))
    dx_mj_dx_pin = stateMappingDerivative_pin2mj_numDiff(x_pin, rmodel, mj_model)

    Rx_numerical_MJ = Rx_numerical_MJ_ @ dx_mj_dx_pin

    # if not np.allclose(r_analytical, r_numerical_MJ, atol=1e-3):
    print("Analytical: ", r_analytical)
    print("Mujoco: ", r_numerical_MJ)
    print("Crocoddyl: ", r_croc)
    print("Difference: ", r_analytical - r_numerical_MJ)

    # if not np.allclose(Rx_analytical, Rx_numerical_MJ, atol=1e-3):
    print("Analytical: \n", Rx_analytical)
    print("croccodyl: \n", Rx_croc)
    print("Python Numerical: \n", Rx_numerical)
    print("Numerical Mujuco after mapping: \n", Rx_numerical_MJ)

    print(f'Difference after mapping: \n{Rx_numerical - Rx_numerical_MJ}')
    # import pdb; pdb.set_trace()
    
    assert np.allclose(Rx_analytical, Rx_numerical_MJ, atol=1e-3), f"Rx mismatch: \nPython NumDiff:\n{Rx_numerical}\nMujoco NumDiff:\n{Rx_numerical_MJ}" 
    # Compare analytical and numerical derivatives

    print("Test passed!")


def test_residual_footClearance(ResidualModel: crocoddyl.ResidualModelAbstract, ResidualModelCroc, rmodel):

    state = ResidualModel.state
    collector = crocoddyl.DataCollectorMultibody(pin.Data(ResidualModel.state.pinocchio))
    data = ResidualModel.createData(collector)
    
    # Random test state and control input
    x = np.hstack((pin.randomConfiguration(rmodel, -np.pi/2 * np.ones(nq), np.pi/2 * np.ones(nq)), np.random.rand(nv)))
    x[2] = -1.
    quat = np.array([0, 0, 0, 1])
    quat = quat / np.linalg.norm(quat)
    x[3:7] = quat
    x[19:22] = 0.0
    u = np.random.rand(nu) * ControlLimit
    
    x_pin = x.copy()
    print(f'x_pin: {x}')
    
    # Compute analytical derivatives
    ResidualModel.calc(data, x_pin, u)
    r_analytical = data.r
    ResidualModel.calcDiff(data, x_pin, u)
    Rx_analytical = data.Rx

    Rx_numerical, _ = calcDiff_numdiff(x_pin, u, ResidualModel, rmodel)

    collector = crocoddyl.DataCollectorMultibody(pin.Data(ResidualModelCroc.state.pinocchio))
    data_croc = ResidualModelCroc.createData(collector)

    pin.forwardKinematics(rmodel, data_croc.pinocchio, x_pin[:nq], x_pin[nq:])
    pin.updateFramePlacements(rmodel, data_croc.pinocchio)
    r_croc = ResidualModelCroc.calc(data_croc, x_pin, u)
    r_croc = data_croc.r
    ResidualModelCroc.calcDiff(data_croc, x_pin, u)
    Rx_croc = data_croc.Rx

    # if not np.allclose(r_analytical, r_numerical_MJ, atol=1e-3):
    print("Analytical: ", r_analytical)
    print("Croco: ", r_croc)

    # if not np.allclose(Rx_analytical, Rx_numerical_MJ, atol=1e-3):
    print("Analytical: \n", Rx_analytical)
    print("croccodyl NumDiff: \n", Rx_croc)
    print("Python Numerical: \n", Rx_numerical)

    print(f'Difference: \n{Rx_numerical - Rx_croc}')
    # import pdb; pdb.set_trace()
    
    assert np.allclose(Rx_croc, Rx_numerical, atol=1e-3), f"Rx mismatch: \nPython NumDiff:\n{Rx_numerical}\nMujoco NumDiff:\n{Rx_numerical_MJ}" 
    # Compare analytical and numerical derivatives

    print("Test passed!")


def main(name):
    # Example usage with your residual models
    state = crocoddyl.StateMultibody(rmodel)
    idx = 0
    fid = fids[idx]

    print(f'fid: {fid}')
    print("Testing")
    
        
    if name == "translation":
        x_ref = np.zeros(3)
        residualModelTranslation = crocoddyl.ResidualModelFrameTranslation(state, fid, x_ref, nu)
        residual_model_normal_translation = ResidualModelFrameTranslationNormal(state, nu, fid)
        test_residual_translation(residual_model_normal_translation, residualModelTranslation, rmodel, mj_model)

    elif name == "velocity":
        V_ref = pin.Motion(np.zeros(6))
        residual_model_tangential_velocity_numDiff = ResidualModelFrameVelocityTangentialNumDiff(state, nu, fid)
        # residualModelVelocity = crocoddyl.ResidualModelFrameVelocity(state, fid, V_ref, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        residual_model_tangential_velocity = ResidualModelFrameVelocityTangential(state, nu, fid)
        test_residual_velocity(residual_model_tangential_velocity, residual_model_tangential_velocity_numDiff, rmodel, mj_model)

    elif name == "footClearance":
        residualModelFootClearance = ResidualModelFootClearance(state, nu, fid, sigmoid_steepness=-10)
        residualModelFootClearanceNumDiff = ResidualModelFootClearanceNumDiff(state, nu, fid, sigmoid_steepness=-10)
        test_residual_footClearance(residualModelFootClearance, residualModelFootClearanceNumDiff, rmodel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an input string.")
    parser.add_argument("input_string", type=str, help="The input string")

    args = parser.parse_args()

    main(args.input_string)
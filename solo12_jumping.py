import numpy as np
import crocoddyl
from mim_solvers import SolverSQP, SolverCSQP
from differential_model_force_free import DifferentialActionModelForceExplicit
import pinocchio as pin
import meshcat
import time
from utils import Arrow
from complementarity_constraints_force_free import ResidualModelComplementarityErrorNormal, ResidualModelFrameTranslationNormal, ResidualModelComplementarityErrorTangential
from friction_cone import  ResidualLinearFrictionCone
from force_derivatives import LocalWorldAlignedForceDerivatives
from solo12_env import create_solo12_env_free_force
from trajectory_data import save_arrays, load_arrays

# Create the robot
env = create_solo12_env_free_force()
nq = env["nq"]
nv = env["nv"]
njoints = env['njoints']
nu = env['nu']
nx = nq + nv
ncontacts = env['ncontacts']
q0 = env['q0']
v0 = env['v0']
rmodel = env['rmodel']
rdata = rmodel.createData()

###################
dt = 1e-1         #
T = 20            #
###################

q0[2] -= 0.02705 # To establish contacts with the ground.
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)


runningCostModel0 = crocoddyl.CostModelSum(state, nu)
runningCostModel1 = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)
waypointCostModel = crocoddyl.CostModelSum(state, nu)

uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegActivation = crocoddyl.ActivationModelWeightedQuad(np.array(12 * [1.0] +                        # joint torques
                                                            env["ncontacts"]*[1.0, 1.0, 1.0]))      # contact forces
uRegCost = crocoddyl.CostModelResidual(state, uRegActivation, uResidual)

xRegActivation = crocoddyl.ActivationModelWeightedQuad(np.array(3 * [0.0]+
                                                             3 * [1.0] +
                                                             12 * [10.0]+
                                                             3 * [1.0] +
                                                             3 * [10.0] +
                                                             12 * [10.0]))
xreg = np.array([0.0, 0.0, 0.0,      # base position
                 0.0, 0.0, 0.0, 1.0] # base orientation
                + q0[7:] +               # joint positions
                [0.0, 0.0, 0.0,      # base linear velocity
                 0.0, 0.0, 0.0]      # base angular velocity
                + v0[6:])                # joint velocities

xResidual = crocoddyl.ResidualModelState(state, xreg, nu)
xRegCost = crocoddyl.CostModelResidual(state, xRegActivation, xResidual)

# runningCostModel0.addCost("uReg", uRegCost, 1e-2)
runningCostModel0.addCost("xReg", xRegCost, 1e-2)

# runningCostModel1.addCost("uReg", uRegCost, 1e-2)
runningCostModel1.addCost("xReg", xRegCost, 1e-2)
# Constraints (friction cones + complementarity contraints)
constraintModelManager = crocoddyl.ConstraintModelManager(state, nu)

eps1 = 1e-4
eps2 = 1e-3
# Friction cone constraints
for idx, fid in enumerate(env["contactFids"]):
    FrictionConeResidual = ResidualLinearFrictionCone(state, njoints, ncontacts, nu, fid, idx, mu = 0.9)
    FrictionConeConstraint = crocoddyl.ConstraintModelResidual(state, FrictionConeResidual, -eps1 * np.ones(5), np.inf * np.ones(5))
    constraintModelManager.addConstraint("FrictionConeConstraint_"+ env["contactFnames"][idx] + str(idx), FrictionConeConstraint)

# # Complementarity constraints
# No penetration constraint
for idx, fid in enumerate(env["contactFids"]):
    ZResidual = ResidualModelFrameTranslationNormal(state, nu, fid)
    ZConstraint = crocoddyl.ConstraintModelResidual(state, ZResidual, -eps1 * np.ones(1), np.inf*np.ones(1))
    constraintModelManager.addConstraint("ZConstraint_"+ env["contactFnames"][idx] + str(idx), ZConstraint)

for idx, fid in enumerate(env["contactFids"]):
    ComplementarityResidual = ResidualModelComplementarityErrorNormal(state, nu, fid, idx, njoints)
    ComplementarityConstraint = crocoddyl.ConstraintModelResidual(state, ComplementarityResidual, -np.inf *np.ones(1), eps2 * np.ones(1))
    constraintModelManager.addConstraint("ComplementarityConstraintNormal_"+ env["contactFnames"][idx] + str(idx), ComplementarityConstraint)

for idx, fid in enumerate(env["contactFids"]):
    ComplementarityResidual = ResidualModelComplementarityErrorTangential(state, nu, fid, idx, njoints)
    ComplementarityConstraint = crocoddyl.ConstraintModelResidual(state, ComplementarityResidual, -eps2 *np.ones(6), eps2 * np.ones(6))
    constraintModelManager.addConstraint("ComplementarityConstraintTangential_"+ env["contactFnames"][idx] + str(idx), ComplementarityConstraint)

# State limits
StateLimit = np.array(env['nq'] * [2 * np.inf] + env['nv'] * [20 * np.pi])
StateResidual = crocoddyl.ResidualModelState(state, nu)
StateLimitConstraint = crocoddyl.ConstraintModelResidual(state, StateResidual, -StateLimit, StateLimit)
constraintModelManager.addConstraint("StateLimitConstraint", StateLimitConstraint)

# # Control limits
ControlLimit = np.array(env['njoints'] * [100.0] + env['ncontacts'] * [100.0, 100.0, 100.0])
ControlRedisual = crocoddyl.ResidualModelControl(state, nu)
ControlLimitConstraint = crocoddyl.ConstraintModelResidual(state, ControlRedisual, -ControlLimit, ControlLimit)
constraintModelManager.addConstraint("ControlLimitConstraint", ControlLimitConstraint)

P_des = [0.2, 0.0, 0.5]
O_des = pin.Quaternion(pin.utils.rpyToMatrix(0.0, 0.0, 0.0))
V_des = [0.0, 0.0, 0.0]
W_des = [0.0, 0.0, 0.0]
x_des = np.array(P_des + 
                 [O_des[0], O_des[1], O_des[2], O_des[3]] + 
                 q0[7:] +
                 V_des + 
                 W_des + 
                 v0[6:])
xDesActivation = crocoddyl.ActivationModelWeightedQuad(np.array(2 * [1e0] +  # base x, y position
                                                                1 * [1e6] +  # base z position
                                                                3 * [1e2] +  # base orientation
                                                                12 * [1e1] +  #joint positions
                                                                3 * [1e-1] +  # base linear velocity
                                                                3 * [1e2] +  # base angular velocity
                                                                12 * [1e2]))  # joint velocities

xDesResidual = crocoddyl.ResidualModelState(state, x_des, nu)
xDesCost = crocoddyl.CostModelResidual(state, xDesActivation, xDesResidual)

waypointCostModel.addCost("xDes0", xDesCost, 1e3)
# waypointCostModel.addCost("xReg", xRegCost, 1e-1)
# waypointCostModel.addCost("uReg", uRegCost, 1e-2)

# xDesActivation = crocoddyl.ActivationModelWeightedQuad(np.array(3 * [1.0]+
#                                                              3 * [1000.0] +
#                                                              12 * [1e-1]+
#                                                              3 * [1.0] +
#                                                              3 * [1.0] +
#                                                              12 * [1e-2]))
xDesActivation = crocoddyl.ActivationModelWeightedQuad(np.array(2 * [1e0] +  # base x, y position
                                                                1 * [1e6] +  # base z position
                                                                3 * [1e2] +  # base orientation
                                                                12 * [1e2] +  #joint positions
                                                                3 * [1e2] +  # base linear velocity
                                                                3 * [1e2] +  # base angular velocity
                                                                12 * [1e2]))  # joint velocities

x0 = np.array(q0 + v0)
print(f'x0:{q0 + v0}')
xDesResidual = crocoddyl.ResidualModelState(state, x0, nu)
xDesCost = crocoddyl.CostModelResidual(state, xDesActivation, xDesResidual)

# runningCostModel1.addCost("xDes1", xDesCost, 1e2)
terminalCostModel.addCost("xDes1", xDesCost, 1e3)
terminalCostModel.addCost("xReg", xRegCost, 1e-1)




running_DAM0 = DifferentialActionModelForceExplicit(state, nu, njoints, env["contactFids"], runningCostModel0, constraintModelManager)
running_DAM1 = DifferentialActionModelForceExplicit(state, nu, njoints, env["contactFids"], runningCostModel1, constraintModelManager)

waypoint_DAM = DifferentialActionModelForceExplicit(state, nu, njoints, env["contactFids"], waypointCostModel, constraintModelManager)
terminal_DAM = DifferentialActionModelForceExplicit(state, nu, njoints, env["contactFids"], terminalCostModel)

runningModel0 = crocoddyl.IntegratedActionModelEuler(running_DAM0, dt)
runningModel1 = crocoddyl.IntegratedActionModelEuler(running_DAM1, dt)

waypointModel = crocoddyl.IntegratedActionModelEuler(waypoint_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

x0 = np.array(q0 + v0)
problem = crocoddyl.ShootingProblem(x0, [runningModel0] * int(T/2) + [waypointModel] + [runningModel1] * int(T/2-1), terminalModel)

solver = SolverCSQP(problem)
solver.use_filter_line_search = False
solver.verbose = True
solver.with_callbacks = True
solver.termination_tolerance = 1e-3

solver.max_qp_iters = 1000
solver.remove_reg = False
maxIter = 1000

xs_init = [x0 for i in range(T+1)]
us_init = problem.quasiStatic([x0 for i in range(T)])
# xs_init, us_init = load_arrays("solo12_jumping")
# xs_init =  xs_init + [xs_init[-1] for i in range(T - len(xs_init) + 1)]
# us_init = us_init + [us_init[-1] for i in range(T - len(us_init))]

flag = solver.solve(xs_init, us_init, maxiter=maxIter)
xs, us = solver.xs, solver.us

print(f'Solved: {flag}')
input("Press to display")
formatter = {'float_kind': lambda x: "{:.4f}".format(x)}

save_arrays(xs, us, "solo12_jumping")


np.set_printoptions(linewidth=210, precision=4, suppress=False, formatter=formatter)

viz = pin.visualize.MeshcatVisualizer(rmodel, env["gmodel"], env["vmodel"])
import zmq
try:
    viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
except zmq.ZMQError as e:
    print(f"Failed to connect to Meshcat server: {e}")

viz.initViewer(viewer)
viz.loadViewerModel()
viz.initializeFrames()
viz.display_frames = True
arrows = []
fids = env["contactFids"]
for i in range(len(fids)):
    arrows.append(Arrow(viz.viewer, "force" + str(i), length_scale=0.01))

for i in range(len(xs)-1):
    force_t = us[i][env["njoints"]:]
    x_t = xs[i]
    print(f"\n********************Time:{i*dt}********************\n")
    print(f'Base position:{x_t[:3]}')
    for eff, fid in enumerate(fids):
        q, v = x_t[:rmodel.nq], x_t[rmodel.nq:]
        cntForce, _, _ = LocalWorldAlignedForceDerivatives(force_t[3*eff:3*(eff+1)], x_t, fids[eff], rmodel, rdata)
        # print(cntForce, force_t[3*eff:3*(eff+1)])
        print(f'foot id:{eff}')
        print(f'joint torques:{us[i][:rmodel.nq]}')
        print(f'contact forces:{force_t[3*eff:3*(eff+1)]}')
        print(f'distance:{rdata.oMf[fid].translation[2]}')
        print(f'complementarity constraint normal:{rdata.oMf[fid].translation[2] * force_t[3*eff:3*(eff+1)]}')
        arrows[eff].anchor_as_vector(rdata.oMf[fid].translation, force_t[3*eff:3*(eff+1)].copy())        
    # time.sleep(0.1)
    viz.display(xs[i][:rmodel.nq])
    input()

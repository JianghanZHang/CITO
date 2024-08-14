import mujoco
from robot_env import create_go2_env_force_MJ, create_go2_env
import crocoddyl
import pinocchio as pin
import numpy as np
from differential_model_force_MJ import DifferentialActionModelForceMJ
import mim_solvers
import meshcat
from force_derivatives import LocalWorldAlignedForceDerivatives
from trajectory_data import save_arrays, load_arrays
from utils import Arrow

pin_env = create_go2_env()
rmodel = pin_env["rmodel"]
rdata = rmodel.createData()

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
mj_data = mj_env["mj_data"]


###################
dt = mj_model.opt.timestep         #
T = 50            #
###################

# q0[2] -= 0.02705 # To establish contacts with the ground.
print(f'x0:{q0 + v0}')
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)


runningCostModel = crocoddyl.CostModelSum(state, nu)
waypointCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegActivation = crocoddyl.ActivationModelWeightedQuad(np.array(12 * [1.0]))
uRegCost = crocoddyl.CostModelResidual(state, uRegActivation, uResidual)

xRegActivation = crocoddyl.ActivationModelWeightedQuad(np.array(3 * [0.0]+
                                                             3 * [1.0] +
                                                             12 * [1.0]+
                                                             3 * [0.0] +
                                                             3 * [1.0] +
                                                             12 * [1.0]))
xreg = np.array([0.0, 0.0, 0.0,      # base position
                 1.0, 0.0, 0.0, 0.0] # base orientation
                + q0[7:] +               # joint positions
                [0.0, 0.0, 0.0,      # base linear velocity
                 0.0, 0.0, 0.0]      # base angular velocity
                + v0[6:])                # joint velocities

xResidual = crocoddyl.ResidualModelState(state, xreg, nu)
xRegCost = crocoddyl.CostModelResidual(state, xRegActivation, xResidual)

# runningCostModel.addCost("uReg", uRegCost, 1e-1)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
waypointCostModel.addCost("xReg", xRegCost, 1e-2)
terminalCostModel.addCost("xReg", xRegCost, 1e-2)
# Constraints (friction cones + complementarity contraints)
RunningConstraintModelManager = crocoddyl.ConstraintModelManager(state, nu)
WaypointConstraintModelManager = crocoddyl.ConstraintModelManager(state, nu)

# # Control limits
ControlLimit = np.array(njoints * [20.0])
ControlRedisual = crocoddyl.ResidualModelControl(state, nu)
ControlLimitConstraint = crocoddyl.ConstraintModelResidual(state, ControlRedisual, -ControlLimit, ControlLimit)
RunningConstraintModelManager.addConstraint("ControlLimitConstraint", ControlLimitConstraint)
WaypointConstraintModelManager.addConstraint("ControlLimitConstraint", ControlLimitConstraint)

# BaseFrameId = rmodel.getFrameId("base_link")
# BasePosRef = np.array([0., 0.,  0.8])
# BasePosResidual = crocoddyl.ResidualModelFrameTranslation(state, BaseFrameId, BasePosRef, nu)
# BasePosContraint = crocoddyl.ConstraintModelResidual(state, BasePosResidual, np.array([-np.inf, -np.inf, 0.0]), np.array([np.inf, np.inf, 0.0]))
# WaypointConstraintModelManager.addConstraint("BasePosContraint", BasePosContraint)
StateRef = np.array(q0 + v0).copy()
StateRef[2] = 0.7
StateResidual = crocoddyl.ResidualModelState(state, StateRef, nu)
StateContraint = crocoddyl.ConstraintModelResidual(state, StateResidual, 
                                                   np.array([-np.inf, -np.inf, 0.] + [-np.inf] * (nq + nv - 3)), 
                                                   np.array([np.inf, np.inf, np.inf] + [np.inf] * (nq + nv - 3)))
WaypointConstraintModelManager.addConstraint("StateContraint", StateContraint)

P_des = [0.0, 0.0, 0.7]
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
                                                                1 * [1e2] +  # base z position
                                                                3 * [1e0] +  # base orientation
                                                                12 * [0.0] +  #joint positions
                                                                3 * [1e-1] +  # base linear velocity
                                                                3 * [1e-1] +  # base angular velocity
                                                                12 * [0.0]))  # joint velocities


xDesResidual = crocoddyl.ResidualModelState(state, x_des, nu)
xDesCost = crocoddyl.CostModelResidual(state, xDesActivation, xDesResidual)
# runningCostModel.addCost("xDes", xDesCost, 1e1)
terminalCostModel.addCost("xDes", xDesCost, 50e1)


running_DAM = DifferentialActionModelForceMJ(mj_model, mj_data, state, nu, njoints, fids, runningCostModel, RunningConstraintModelManager)
waypoint_DAM = DifferentialActionModelForceMJ(mj_model, mj_data, state, nu, njoints, fids, runningCostModel, WaypointConstraintModelManager)
terminal_DAM = DifferentialActionModelForceMJ(mj_model, mj_data, state, nu, njoints, fids, terminalCostModel)

running_DAM = crocoddyl.DifferentialActionModelNumDiff(running_DAM, False)
waypoint_DAM = crocoddyl.DifferentialActionModelNumDiff(waypoint_DAM, False)
terminal_DAM = crocoddyl.DifferentialActionModelNumDiff(terminal_DAM, False)

runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
waypointModel = crocoddyl.IntegratedActionModelEuler(waypoint_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
x0 = np.array(q0 + v0)
problem = crocoddyl.ShootingProblem(x0, [runningModel] * (T-1) + [waypointModel], terminalModel)
solver = mim_solvers.SolverCSQP(problem)
solver.mu_constriant = 10.
solver.mu_dynamic = 10.
solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
solver.use_filter_line_search = False
solver.verbose = True
solver.termination_tolerance = 1e-3
solver.remove_reg = False
solver.max_qp_iters = 1000

# solver = crocoddyl.SolverDDP(problem)
# solver.setCallbacks([crocoddyl.CallbackVerbose()])

xs_init = [x0 for i in range(T+1)]
us_init = problem.quasiStatic([x0 for i in range(T)])
# import pdb; pdb.set_trace()
# xs_init, us_init = load_arrays("go2_takeoff_MJ")

maxIter = 50
print('Start solving')
flag = solver.solve(xs_init, us_init, maxiter=maxIter, isFeasible=False, regInit = 1e-8)
xs, us = solver.xs, solver.us
# xs, us = xs_init, us_init
# import pdb; pdb.set_trace()
log = solver.getCallbacks()[-1]
crocoddyl.plotOCSolution(xs, us)
mim_solvers.plotConvergence(log.convergence_data)
print(f'Solved: {flag}')
input("Press to display")
formatter = {'float_kind': lambda x: "{:.4f}".format(x)}

save_arrays(xs, us, "go2_takeoff_MJ_constraint")
np.set_printoptions(linewidth=210, precision=4, suppress=False, formatter=formatter)

from pinocchio.visualize import MeshcatVisualizer
viz = MeshcatVisualizer(rmodel, pin_env["gmodel"], pin_env["vmodel"])
import zmq
try:
    viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
except zmq.ZMQError as e:
    print(f"Failed to connect to Meshcat server: {e}")

print('Connected to meshcat')
viz.initViewer(viewer)
viz.loadViewerModel()
viz.initializeFrames()
viz.display_frames = True
arrows = []
fids = fids
# import pdb; pdb.set_trace
for i in range(len(fids)):
    arrows.append(Arrow(viz.viewer, "force" + str(i), length_scale=0.01))

for i in range(len(xs)-1):
    force_t = us[i][njoints:]
    x_t = xs[i]
    print(f"\n********************Time:{i*dt}********************\n")
    print(f'Base position:{x_t[:3]}')
    for eff, fid in enumerate(fids):
        q, v = x_t[:rmodel.nq], x_t[rmodel.nq:]
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.computeAllTerms(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)
        # cntForce, _, _ = LocalWorldAlignedForceDerivatives(force_t[3*eff:3*(eff+1)], x_t, fids[eff], rmodel, rdata)
        print(f'foot id:{eff}')
        # print(f'contact forces:{force_t[3*eff:3*(eff+1)]}')
        print(f'distance:{rdata.oMf[fid].translation[2]}')
        print(f'complementarity constraint:{rdata.oMf[fid].translation[2] * force_t[3*eff:3*(eff+1)]}')
        # arrows[eff].anchor_as_vector(rdata.oMf[fid].translation, force_t[3*eff:3*(eff+1)].copy())        
    # time.sleep(0.1)
    viz.display(xs[i][:rmodel.nq])
    input()
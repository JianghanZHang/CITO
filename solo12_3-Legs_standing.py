import numpy as np
import crocoddyl
from mim_solvers import SolverSQP, SolverCSQP
from differential_model_force_free import DifferentialActionModelForceExplicit
import pinocchio as pin
import meshcat
import time
from utils import Arrow
from complementarity_constraints_force_free import ResidualModelContactForceNormal, ResidualModelComplementarityErrorNormal, ResidualModelFrameTranslationNormal, ResidualModelComplementarityErrorTangential
from friction_cone import  ResidualLinearFrictionCone
from force_derivatives import LocalWorldAlignedForceDerivatives
from solo12_env import create_solo12_env
from trajectory_data import save_arrays, load_arrays

# Create the robot
env = create_solo12_env()
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
dt = 1e-2         #  
T = 50            #  
###################

q0[2] -= 0.02705
print(f'x0:{q0 + v0}')
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)


runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

# uResidual = crocoddyl.ResidualModelControl(state, nu)
# uRegActivation = crocoddyl.ActivationModelWeightedQuad(np.array(12 * [1.0] +                        # joint torques
#                                                             env["ncontacts"]*[1.0, 1.0, 1.0]))      # contact forces
# uRegCost = crocoddyl.CostModelResidual(state, uRegActivation, uResidual)

uGuideResidual = crocoddyl.ResidualModelControl(state, nu)
uGuideActivation = crocoddyl.ActivationModelWeightedQuad(np.array(12 * [1.0] +                             # joint torques
                                                             [1.0, 1.0, 1.0] +                             # first contact forces
                                                            (env["ncontacts"] -1)*[1.0, 1.0, 1.0]))      # other contact forces
uGuideCost = crocoddyl.CostModelResidual(state, uGuideActivation, uGuideResidual)

xRegActivation = crocoddyl.ActivationModelWeightedQuad(np.array(3 * [0.0]+
                                                             3 * [1.0] +
                                                             12 * [1.0]+
                                                             3 * [0.0] +
                                                             3 * [1.0] +
                                                             12 * [1.0]))
xreg = np.array([0.0, 0.0, 0.0,      # base position
                 0.0, 0.0, 0.0, 1.0] # base orientation
                + q0[7:] +               # joint positions
                [0.0, 0.0, 0.0,      # base linear velocity
                 0.0, 0.0, 0.0]      # base angular velocity
                + v0[6:])                # joint velocities

xResidual = crocoddyl.ResidualModelState(state, xreg, nu)
xRegCost = crocoddyl.CostModelResidual(state, xRegActivation, xResidual)

runningCostModel.addCost("uReg", uGuideCost, 1e-1)
# runningCostModel.addCost("uReg", uRegCost, 1e-1)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
# Constraints (friction cones + complementarity contraints)
constraintModelManager = crocoddyl.ConstraintModelManager(state, nu)


# Friction cone constraints
for idx, fid in enumerate(env["contactFids"]):
    FrictionConeResidual = ResidualLinearFrictionCone(state, njoints, ncontacts, nu, fid, idx, mu = 0.9)
    FrictionConeConstraint = crocoddyl.ConstraintModelResidual(state, FrictionConeResidual, -1e-3 * np.ones(5), np.inf * np.ones(5))
    constraintModelManager.addConstraint("FrictionConeConstraint_"+ env["contactFnames"][idx] + str(idx), FrictionConeConstraint)

# # Complementarity constraints
# No penetration constraint
for idx, fid in enumerate(env["contactFids"]):
    ZResidual = ResidualModelFrameTranslationNormal(state, nu, fid)
    ZConstraint = crocoddyl.ConstraintModelResidual(state, ZResidual, -1e-3 * np.ones(1), np.inf*np.ones(1))
    constraintModelManager.addConstraint("ZConstraint_"+ env["contactFnames"][idx] + str(idx), ZConstraint)

for idx, fid in enumerate(env["contactFids"]):
    ComplementarityResidual = ResidualModelComplementarityErrorNormal(state, nu, fid, idx, njoints)
    ComplementarityConstraint = crocoddyl.ConstraintModelResidual(state, ComplementarityResidual, -np.inf *np.ones(1), 1e-3 * np.ones(1))
    constraintModelManager.addConstraint("ComplementarityConstraintNormal_"+ env["contactFnames"][idx] + str(idx), ComplementarityConstraint)

for idx, fid in enumerate(env["contactFids"]):
    ComplementarityResidual = ResidualModelComplementarityErrorTangential(state, nu, fid, idx, njoints)
    ComplementarityConstraint = crocoddyl.ConstraintModelResidual(state, ComplementarityResidual, -1e-3 *np.ones(4), 1e-3 * np.ones(4))
    constraintModelManager.addConstraint("ComplementarityConstraintTangential_"+ env["contactFnames"][idx] + str(idx), ComplementarityConstraint)

idx = 0
ForceResidual = ResidualModelContactForceNormal(state, nu, env["contactFids"][idx], idx, njoints)
ForceConstraint = crocoddyl.ConstraintModelResidual(state, ForceResidual, -np.inf * np.ones(1), 0 * np.ones(1))
constraintModelManager.addConstraint("ForceConstraintNormal_"+ env["contactFnames"][idx] + str(idx), ForceConstraint)


P_des = [0.2, 0.0, q0[2]]
O_des = pin.Quaternion(pin.utils.rpyToMatrix(0.0, 0.0, 0.0))
V_des = [0.0, 0.0, 0.0]
W_des = [0.0, 0.0, 0.0]
x_des = np.array(P_des + 
                 [O_des[0], O_des[1], O_des[2], O_des[3]] + 

                 [-np.pi*2/5, 0.0, 1.6] + 
                 q0[10:] +

                 V_des + 
                 W_des + 
                 v0[6:])
xDesActivation = crocoddyl.ActivationModelWeightedQuad(np.array(3 * [1.0] +  # base position
                                                                3 * [0.0] +  # base orientation

                                                                3 * [1.0] + # First 3 joints positions
                                                                9 * [0.0] +  # other joint positions

                                                                3 * [0.0] +  # base linear velocity
                                                                3 * [0.0] +  # base angular velocity
                                                                12 * [0.0]))  # joint velocities

xDesResidual = crocoddyl.ResidualModelState(state, x_des, nu)
xDesCost = crocoddyl.CostModelResidual(state, xDesActivation, xDesResidual)
runningCostModel.addCost("xDes", xDesCost, 1e2)
terminalCostModel.addCost("xDes", xDesCost, 5e3)

running_DAM = DifferentialActionModelForceExplicit(state, nu, njoints, env["contactFids"], runningCostModel, constraintModelManager)
terminal_DAM = DifferentialActionModelForceExplicit(state, nu, njoints, env["contactFids"], terminalCostModel)

runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

x0 = np.array(q0 + v0)
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

solver = SolverCSQP(problem)
solver.use_filter_line_search = False
solver.verbose = True
solver.with_callbacks = True
# solver.with_qp_callbacks = True
solver.termination_tolerance = 1e-3
solver.max_qp_iters = 1000
solver.remove_reg = False

xs_init = [x0 for i in range(T+1)]
us_init = [np.zeros(nu) for i in range(T)]

# xs_init, us_init = load_arrays("solo12_3-Legs_standing")
flag = solver.solve(xs_init, us_init, maxiter=1000)
xs, us = solver.xs, solver.us

save_arrays(xs, us, "solo12_3-Legs_standing")
print(f'Solved: {flag}')
input("Press to display")

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
    T = pin.utils.XYZQUATToSE3(x_t[:7])
    print(f"\n********************Time:{i*dt}********************\n")
    print(f'Base position:{T.translation}')
    print(f'Base rpy:{pin.utils.matrixtoRpy(T.rotation)}')
    for eff, fid in enumerate(fids):
        q, v = x_t[:rmodel.nq], x_t[rmodel.nq:]
        cntForce, _, _ = LocalWorldAlignedForceDerivatives(force_t[3*eff:3*(eff+1)], x_t, fids[eff], rmodel, rdata)
        # print(cntForce, force_t[3*eff:3*(eff+1)])
        print(f'foot id:{eff}')
        print(f'contact forces:{force_t[3*eff:3*(eff+1)]}')
        print(f'distance:{rdata.oMf[fid].translation[2]}')
        print(f'complementarity constraint:{rdata.oMf[fid].translation[2] * force_t[3*eff:3*(eff+1)]}')
        arrows[eff].anchor_as_vector(rdata.oMf[fid].translation, force_t[3*eff:3*(eff+1)].copy())        
    time.sleep(0.01)
    viz.display(xs[i][:rmodel.nq])

import mujoco
from robot_env import create_go2_env_force_MJ, create_go2_env
import crocoddyl
import pinocchio as pin
import numpy as np
from differential_model_force_MJ import DifferentialActionModelForceMJ
from integrated_action_model_MJ import IntegratedActionModelForceMJ
from CallbackLogger_force import CallbackLogger
import mim_solvers
import meshcat
from force_derivatives import LocalWorldAlignedForceDerivatives
from trajectory_data import save_arrays, load_arrays
from utils import Arrow
import sys
def main():
    solver_type = sys.argv[1]

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
    terminalCostModel = crocoddyl.CostModelSum(state, nu)

    uResidual = crocoddyl.ResidualModelControl(state, nu)
    uRegActivation = crocoddyl.ActivationModelWeightedQuad(np.array(4 * [0.5, 0.5, 1.0])) 
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

    runningCostModel.addCost("uReg", uRegCost, 1e-1)
    # runningCostModel.addCost("xReg", xRegCost, 1e-2)
    # Constraints (friction cones + complementarity contraints)
    constraintModelManager = crocoddyl.ConstraintModelManager(state, nu)

    abduction_lb = -1.0472
    abduction_ub = 1.0472

    front_hip_lb = -1.5708
    front_hip_ub = 3.4807

    knee_lb = -2.7227
    knee_ub = -0.83776

    back_hip_lb = -1.5708
    back_hip_ub = 4.5379

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
    

    StateResidual = crocoddyl.ResidualModelState(state, np.zeros(37), nu)
    StateLimitConstraint = crocoddyl.ConstraintModelResidual(state, StateResidual, StateLimit_lb, StateLimit_ub)
    constraintModelManager.addConstraint("StateLimitConstraint", StateLimitConstraint)

    # # Control limits
    ControlLimit = np.array(4 * [23.7, 23.7, 45.43])
    ControlRedisual = crocoddyl.ResidualModelControl(state, nu)
    ControlLimitConstraint = crocoddyl.ConstraintModelResidual(state, ControlRedisual, -ControlLimit, ControlLimit)
    constraintModelManager.addConstraint("ControlLimitConstraint", ControlLimitConstraint)

    P_des = [1.0, 0.0, 0.280]
    O_des = pin.Quaternion(pin.utils.rpyToMatrix(0.0, 0.0, 0.0))
    V_des = [2.0, 0.0, 0.0]
    W_des = [0.0, 0.0, 0.0]
    x_des = np.array(P_des + 
                    [O_des[0], O_des[1], O_des[2], O_des[3]] + 
                    q0[7:] +
                    V_des + 
                    W_des + 
                    v0[6:])
    xDesActivationRunning = crocoddyl.ActivationModelWeightedQuad(np.array(1 * [1e1] +  # base x, y position
                                                                    2 * [1e0] +  # base z position
                                                                    3 * [1e0] +  # base orientation
                                                                    12 * [0] +  #joint positions
                                                                    3 * [1e0] +  # base linear velocity
                                                                    3 * [1e0] +  # base angular velocity
                                                                    12 * [1e-2]))  # joint velocities

    xDesActivationTerminal = crocoddyl.ActivationModelWeightedQuad(np.array(1 * [1e1] +  # base x, y position
                                                                    2 * [1e0] +  # base z position
                                                                    3 * [1e0] +  # base orientation
                                                                    12 * [1e0] +  #joint positions
                                                                    3 * [0] +  # base linear velocity
                                                                    3 * [1e0] +  # base angular velocity
                                                                    12 * [1e-1]))  # joint velocities



    xDesResidual = crocoddyl.ResidualModelState(state, x_des, nu)
    xDesCostRunning = crocoddyl.CostModelResidual(state, xDesActivationRunning, xDesResidual)
    xDesCostTerminal = crocoddyl.CostModelResidual(state, xDesActivationTerminal, xDesResidual)

    runningCostModel.addCost("xDes_running", xDesCostRunning, 1e2)
    terminalCostModel.addCost("xDes_terminal", xDesCostTerminal,  1e1)


    running_DAM = DifferentialActionModelForceMJ(mj_model, mj_data, state, nu, njoints, fids, runningCostModel, constraintModelManager)
    terminal_DAM = DifferentialActionModelForceMJ(mj_model, mj_data, state, nu, njoints, fids, terminalCostModel)
    
    # running_DAM = crocoddyl.DifferentialActionModelNumDiff(running_DAM)
    # terminal_DAM = crocoddyl.DifferentialActionModelNumDiff(terminal_DAM)

    # runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
    # terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

    runningModel = IntegratedActionModelForceMJ(running_DAM, dt, True)
    terminalModel = IntegratedActionModelForceMJ(terminal_DAM, 0., True)

    runningModels = [IntegratedActionModelForceMJ(running_DAM, dt, True) for _ in range(T)]
    for runningModel in runningModels:
        runningModel.x_lb = StateLimit_lb
        runningModel.x_ub = StateLimit_ub
        runningModel.u_lb = -ControlLimit  
        runningModel.u_ub = ControlLimit 
    x0 = np.array(q0 + v0)

    # problem = crocoddyl.ShootingProblem(x0, [runningModel] * (T), terminalModel)
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    xs_init = [x0 for i in range(T+1)]
    us_init = problem.quasiStatic([x0 for i in range(T)])

    # xs_init, us_init = load_arrays("go2_walking_MJ_CSQP1")

   
    maxIter = 500

    print('Start solving')

    if solver_type == "CSQP":
        solver = mim_solvers.SolverCSQP(problem)
        solver.mu_constriant = 1.
        solver.mu_dynamic = 10.
        solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
        solver.use_filter_line_search = False
        solver.verbose = True
        solver.termination_tolerance = 1e-3
        solver.remove_reg = False
        solver.max_qp_iters = 1000
        flag = solver.solve(xs_init, us_init, maxiter=maxIter, isFeasible=False)

    elif solver_type == "FDDP":
        solver = crocoddyl.SolverBoxDDP(problem)
        solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
        flag = solver.solve(xs_init, us_init, maxIter, False)
        
    elif solver_type == "IPOPT":
        solver = crocoddyl.SolverIpopt(problem)
        solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
        flag = solver.solve(xs_init, us_init, maxIter, False)

    else:
        exit("Invalid solver type")

    xs, us = solver.xs, solver.us
    log = solver.getCallbacks()[-1]
    print(f'Solved: {flag}')

    print(f'Final state: \n {xs[-1]}')
    # if solver_type == "CSQP":
    #     mim_solvers.plotConvergence(log.convergence_data)
    # elif solver_type == "FDDP":
    #     crocoddyl.plotConvergence(log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2)
    input("Press to display")
    crocoddyl.plotOCSolution(xs, us)

    forces = np.array([runningModel.forces[:, :3] for runningModel in problem.runningModels])
    # import pdb; pdb.set_trace()

    formatter = {'float_kind': lambda x: "{:.4f}".format(x)}
    save = sys.argv[2]
    if save == "save":
        save_arrays(xs, us, "go2_takeoff_MJ_" + solver_type)
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
    import time
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

if __name__   == "__main__":
    main()
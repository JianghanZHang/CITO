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
from utils import xyzw2wxyz, wxyz2xyzw
from ResidualModels import ResidualModelFootClearance, ResidualModelFootSlipping
import sys
from robot_env import GO2_FOOT_RADIUS
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

    ###################
    dt = mj_model.opt.timestep         #
    T = 40            #
    ###################

    # q0[2] -= 0.02705 # To establish contacts with the ground.
    print(f'x0:{q0 + v0}')
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFloatingBase(state)


    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)

    uResidual = crocoddyl.ResidualModelControl(state, nu)
    uRegActivation = crocoddyl.ActivationModelWeightedQuad(np.array(4 * [1.0, 1.0, 1.0])) 
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

    # runningCostModel.addCost("xReg", xRegCost, 1e-2)
    # Constraints (friction cones + complementarity contraints)
    constraintModelManager = crocoddyl.ConstraintModelManager(state, nu)
    constraintModelManager0 = crocoddyl.ConstraintModelManager(state, nu)
    

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
    # constraintModelManager.addConstraint("StateLimitConstraint",su StateLimitConstraint)

    # # Control limits
    ControlLimit = np.array(4 * [23.7, 23.7, 45.43])
    ControlRedisual = crocoddyl.ResidualModelControl(state, nu)
    ControlLimitConstraint = crocoddyl.ConstraintModelResidual(state, ControlRedisual, -ControlLimit, ControlLimit)
    constraintModelManager.addConstraint("ControlLimitConstraint", ControlLimitConstraint)
    constraintModelManager0.addConstraint("ControlLimitConstraint", ControlLimitConstraint)

    if solver_type == "CSQP":
        relaxation = 1e-6
        for idx, fid in enumerate(fids):
            footSlippingResidual = ResidualModelFootSlipping(state, nu, fid)
            footSlippingConstraint = crocoddyl.ConstraintModelResidual(state, footSlippingResidual, np.array([-relaxation]), np.array([np.inf]))
            constraintModelManager.addConstraint(f"footSlipping_{idx}", footSlippingConstraint)
        w = 0

    elif solver_type == "FDDP":
        w = 1e1
    
    for idx, fid in enumerate(fids):
        footClearanceResidual = ResidualModelFootClearance(state, nu, fid, sigmoid_steepness=-3)
        footClearanceActivation = crocoddyl.ActivationModelSmooth1Norm(nr=1, eps=1e-12)
        footClearanceCost = crocoddyl.CostModelResidual(state, footClearanceActivation, footClearanceResidual)
        runningCostModel.addCost(f"footClearance_{idx}", footClearanceCost, w)
        terminalCostModel.addCost(f"footClearance_{idx}", footClearanceCost, dt * w)

    Px_des = 0.5
    Vx_des = 1.0

    P_des = [Px_des, 0.0, 0.2800]
    O_des = pin.Quaternion(pin.utils.rpyToMatrix(0.0, 0.0, 0.0))

    V_des = [Vx_des, 0.0, 0.0]
    W_des = [0.0, 0.0, 0.0]
    x_des = np.array(P_des + 
                    [O_des[0], O_des[1], O_des[2], O_des[3]] + 
                    q0[7:] +
                    V_des + 
                    W_des + 
                    v0[6:])
    
    xDesActivationRunning = crocoddyl.ActivationModelWeightedQuad(np.array(
                                                                    1 * [1e2] +  # base x position
                                                                    1 * [1e2] +  # base y position
                                                                    1 * [1e2] +  # base z position
                                                                    3 * [1e0] +  # base orientation
                                                                    12 * [0] +  #joint positions
                                                                    3 * [1e1] +  # base linear velocity
                                                                    3 * [1e0] +  # base angular velocity
                                                                    12 * [1e-1]))  # joint velocities

    xDesActivationTerminal = crocoddyl.ActivationModelWeightedQuad(np.array(
                                                                    1 * [1e3] +  # base x position
                                                                    1 * [1e3] +  # base y position
                                                                    1 * [1e3] +  # base z position
                                                                    3 * [1e1] +  # base orientation
                                                                    12 * [1e0] +  #joint positions
                                                                    3 * [0] +  # base linear velocity
                                                                    3 * [1e0] +  # base angular velocity
                                                                    12 * [1e-1]))  # joint velocities


    runningCostModel.addCost("uReg", uRegCost, 1e-3)

    xDesResidual = crocoddyl.ResidualModelState(state, x_des, nu)
    xDesCostRunning = crocoddyl.CostModelResidual(state, xDesActivationRunning, xDesResidual)
    xDesCostTerminal = crocoddyl.CostModelResidual(state, xDesActivationTerminal, xDesResidual)

    runningCostModel.addCost("xDes_running", xDesCostRunning, 1e1)
    terminalCostModel.addCost("xDes_terminal", xDesCostTerminal, 1e1)


    terminal_DAM = DifferentialActionModelForceMJ(mj_model, state, nu, njoints, fids, terminalCostModel, None)
    

    runningModels = [IntegratedActionModelForceMJ
                     (DifferentialActionModelForceMJ(mj_model, state, nu, njoints, fids, runningCostModel, constraintModelManager), dt, True) 
                    for _ in range(T)]
    
    runningModels[0] = IntegratedActionModelForceMJ(DifferentialActionModelForceMJ(mj_model, state, nu, njoints, fids, runningCostModel, constraintModelManager0), dt, False)
    
    terminalModel = IntegratedActionModelForceMJ(terminal_DAM, 0., True)

    
    for runningModel in runningModels:
        runningModel.x_lb = StateLimit_lb
        runningModel.x_ub = StateLimit_ub
        runningModel.u_lb = -ControlLimit  
        runningModel.u_ub = ControlLimit

    x0 = np.array(q0 + v0)
    x0[2] = 0.28800
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)

    base_x_start = x0[0].copy()
    base_x_end = Px_des
    num_steps = T + 1 
    base_x_values = np.linspace(base_x_start, base_x_end, num_steps)

    # Initialize xs_init with the interpolated X position
    xs_init = [np.copy(x0) for _ in range(num_steps)]
    if solver_type == "CSQP":
        for i in range(1, num_steps):
            xs_init[i][2] = 0.28800 
            xs_init[i][0] = base_x_values[i] 
            # xs_init[i][nq] = Vx_des #assign desired base velocity to initial guess
    
    elif solver_type == "FDDP":
        for i in range(1, num_steps):
            xs_init[i][2] = 0.29800
            xs_init[i][0] = base_x_values[i] 
            # xs_init[i][nq] = Vx_des #assign desired base velocity to initial guess

    us_init = [np.zeros(nu) for i in range(T)]
    # xs_init, us_init = load_arrays("go2_walking_MJ_CSQP1")
    print(f'xs_init[0]: {xs_init[0]}')
    print(f'xs_init[-1]: {xs_init[-1]}')
    maxIter = 100

    print('Start solving')

    if solver_type == "CSQP":
        solver = mim_solvers.SolverCSQP(problem)
        solver.mu_constriant = 10000.
        solver.mu_dynamic = 10.
        solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
        solver.use_filter_line_search = True
        solver.verbose = True
        solver.termination_tolerance = 1e-3
        solver.remove_reg = False
        solver.max_qp_iters = 2500
        flag = solver.solve(xs_init, us_init, maxiter=maxIter, isFeasible=False)

    elif solver_type == "FDDP":
        solver = crocoddyl.SolverBoxFDDP(problem)
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
    runningDatas = problem.runningDatas
    runningModels = problem.runningModels
    terminalData = problem.terminalData
    terminalModel = problem.terminalModel

    

    # import pdb; pdb.set_trace()
    print(f'Final state: \n {xs[-1]}')
    # if solver_type == "CSQP":
    #     mim_solvers.plotConvergence(log.convergence_data)
    # elif solver_type == "FDDP":
    #     crocoddyl.plotConvergence(log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2)
    # print(f'forces: \n {runningModel.forces}')
    input("Press to display")
    crocoddyl.plotOCSolution(xs, us)

    forces = np.array([runningModel.forces[:, :3] for runningModel in problem.runningModels])
    contacts = [runningModel.contacts for runningModel in problem.runningModels]
    mj_datas = [runningModel.mj_data for runningModel in problem.runningModels]

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
        runningData = runningDatas[i]
        runningModel = runningModels[i]
        force_t = us[i][njoints:]
        x_t = xs[i]
        print(f"\n********************Time:{i*dt}********************\n")
        print(f'Base position:{x_t[:3]}')
        print(f'Contacts:\n {contacts[i]}')
        for eff, fid in enumerate(fids):
            q, v = x_t[:rmodel.nq], x_t[rmodel.nq:]
            pin.framesForwardKinematics(rmodel, rdata, q)
            pin.computeAllTerms(rmodel, rdata, q, v)
            pin.updateFramePlacements(rmodel, rdata)
            # cntForce, _, _ = LocalWorldAlignedForceDerivatives(force_t[3*eff:3*(eff+1)], x_t, fids[eff], rmodel, rdata)
            print(f'foot id:{eff}')
            print(f'forces: {forces[i][eff]}')
            # print(f'contact forces:{force_t[3*eff:3*(eff+1)]}')
            print(f'distance:{rdata.oMf[fid].translation[2] - GO2_FOOT_RADIUS}')

            name = "footClearance_" + str(eff)


            foot_clearance_cost = runningData.differential.costs.costs[name].cost
            foot_clearance_weight = runningModel.differential.costs.costs[name].weight

            print(f'foot clearance cost: {foot_clearance_cost * foot_clearance_weight}\n')

            ureg_cost = runningData.differential.costs.costs["uReg"].cost
            ureg_weight = runningModel.differential.costs.costs["uReg"].weight


        print(f'\nureg_cost: {ureg_cost * ureg_weight} ')

        print(f'__________________________________________')

            # print(f'complementarity constraint:{rdata.oMf[fid].translation[2] * force_t[3*eff:3*(eff+1)]}')
            # arrows[eff].anchor_as_vector(rdata.oMf[fid].translation, force_t[3*eff:3*(eff+1)].copy())        
        # time.sleep(0.1)
        viz.display(xs[i][:rmodel.nq])
        # time.sleep(0.1)
        input()
        
    print(f'Final cost: {terminalData.differential.costs.costs["xDes_terminal"].cost * terminalModel.differential.costs.costs["xDes_terminal"].weight}')


if __name__   == "__main__":
    main()
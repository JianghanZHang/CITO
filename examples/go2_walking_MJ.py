import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../robots/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'python/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'robots/')))
from robots.robot_env import create_go2_env_force_MJ, create_go2_env
import crocoddyl
import pinocchio as pin
import numpy as np
from differential_model_MJ import DifferentialActionModelMJ
from integrated_action_model_MJ import IntegratedActionModelForceMJ
from CallbackLogger_force import CallbackLogger
import mim_solvers
import meshcat
from force_derivatives import LocalWorldAlignedForceDerivatives
from utils import save_arrays, load_arrays, extend_trajectory
from utils import Arrow
from utils import plot_base_poses, plot_normal_forces, plot_sliding
from utils import xyzw2wxyz, wxyz2xyzw, add_noise_to_state
from python.ResidualModels import ResidualModelFootClearanceNumDiff, ResidualModelFootSlippingNumDiff, ResidualModelFootClearance, ResidualModelFootSlipping
import sys
from robots.robot_env import GO2_FOOT_RADIUS

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
    jids = pin_env["jointFids"]
    njoints = mj_env["njoints"]
    nq = mj_env["nq"]
    nv = mj_env["nv"]
    mj_model = mj_env["mj_model"]

    ################# OCP params ################
    dt = 0.01                                 
    mj_model.opt.timestep = dt                
    T = 100                                               
    T_total = dt * T                          

    print(f'x0:{q0 + v0}')
    ################# Initialize crocoddyl models ################
    state = crocoddyl.StateMultibody(rmodel)
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)
    constraintModelManager = crocoddyl.ConstraintModelManager(state, nu)    

    ################ State limits ################
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

    ################ Control limits ################
    ControlLimit = np.array(4 * [23.7, 23.7, 45.43])
    ControlRedisual = crocoddyl.ResidualModelControl(state, nu)
    ControlLimitConstraint = crocoddyl.ConstraintModelResidual(state, ControlRedisual, -ControlLimit, ControlLimit)
    constraintModelManager.addConstraint("ControlLimitConstraint", ControlLimitConstraint)

    ################ Foot Clearance Cost ################
    w = 1.0

    for idx, fid in enumerate(fids):

        sigmoid_steepness = -30
        footClearanceResidual = ResidualModelFootClearanceNumDiff(state, nu, fid, sigmoid_steepness=sigmoid_steepness)
        footClearanceActivation = crocoddyl.ActivationModelSmooth1Norm(nr=1, eps=1e-12)
        footClearanceCost = crocoddyl.CostModelResidual(state, footClearanceActivation, footClearanceResidual)
        runningCostModel.addCost(f"footClearance_{idx}", footClearanceCost, w)
        terminalCostModel.addCost(f"footClearance_{idx}", footClearanceCost, w)
    ################ State Cost ######################
    Px_des = 1.0
    Vx_des = Px_des / T_total

    P_des = [Px_des, 0.0, 0.2700]
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
                                                                        1 * [1] +  # base x position
                                                                        1 * [1] +  # base y position
                                                                        1 * [10] +  # base z position
                                                                        3 * [10] +  # base orientation
                                                                        4 * [5e-4, 5e-4, 5e-4] +  #joint positions
                                                                        3 * [1] +  # base linear velocity
                                                                        3 * [1] +  # base angular velocity
                                                                        4 * [5e-4, 5e-4, 5e-4]))  # joint velocities

    xDesActivationTerminal = crocoddyl.ActivationModelWeightedQuad(np.array(
                                                                        1 * [10] +  # base x position
                                                                        1 * [10] +  # base y position
                                                                        1 * [100] +  # base z position
                                                                        3 * [10] +  # base orientation
                                                                        4 * [5e-1, 5e-1, 5e-1] +  #joint positions
                                                                        3 * [1] +  # base linear velocity
                                                                        3 * [1] +  # base angular velocity
                                                                        4 * [5e-4, 5e-4, 5e-4]))  # joint velocities



    xDesResidual = crocoddyl.ResidualModelState(state, x_des, nu)
    xDesCostRunning = crocoddyl.CostModelResidual(state, xDesActivationRunning, xDesResidual)
    xDesCostTerminal = crocoddyl.CostModelResidual(state, xDesActivationTerminal, xDesResidual)

    runningCostModel.addCost("xDes_running", xDesCostRunning, 1)
    terminalCostModel.addCost("xDes_terminal", xDesCostTerminal, 1)

    ################## Control Regularization Cost##################
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    uRegActivation = crocoddyl.ActivationModelWeightedQuad(np.array(12 * [1.0]))
    uRegCost = crocoddyl.CostModelResidual(state, uRegActivation, uResidual)
    runningCostModel.addCost("uReg", uRegCost, 1e-2)

    ################### Initialize running and terminal models ################
    runningModels = [IntegratedActionModelForceMJ
                     (DifferentialActionModelMJ(mj_model, state, nu, njoints, fids, runningCostModel, constraintModelManager), dt, True) 
                    for _ in range(T)]
    
    
    terminalModel = IntegratedActionModelForceMJ(DifferentialActionModelMJ(mj_model, state, nu, njoints, fids, terminalCostModel, None), 0., True)

    # Added for FDDP solver
    for runningModel in runningModels:
        runningModel.x_lb = StateLimit_lb
        runningModel.x_ub = StateLimit_ub
        runningModel.u_lb = -ControlLimit  
        runningModel.u_ub = ControlLimit

    x0 = np.array(q0 + v0)
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)

    num_steps = T + 1 

    ################### Initialize xs_init using interpolated trajs with noises to warmstart the solver ###################
    xs_init = [np.copy(x0) for _ in range(num_steps)]
    base_x_start = x0[0].copy()
    base_x_end = Px_des
    base_x_values = np.linspace(base_x_start, base_x_end, num_steps)
    us_init = [np.zeros(nu) for i in range(T)]

    # xs_init, us_init = load_arrays("go2_walking_MJ_CSQP")
    # xs_init, us_init = extend_trajectory(xs_init, us_init, 10)

    if solver_type == "CSQP":
        for i in range(1, num_steps):
            xs_init[i][2] = 0.2800
            xs_init[i][0] = base_x_values[i] 
            xs_init[i][nq] = Vx_des #assign desired base velocity to initial guess
            xs_init[i] = add_noise_to_state(rmodel, xs_init[i], scale=0.02).copy()
    
    elif solver_type == "FDDP":
         for i in range(1, num_steps):
            xs_init[i][2] = 0.2800
            xs_init[i][0] = base_x_values[i] 
            xs_init[i][nq] = Vx_des #assign desired base velocity to initial guess
            xs_init[i] = add_noise_to_state(rmodel, xs_init[i], scale=0.02).copy()
    else:
        exit("Invalid solver type")

    maxIter = 400
    ################### Set solver params and solve ###################
    print('Start solving')
    if solver_type == "CSQP":
        solver = mim_solvers.SolverCSQP(problem)
        solver.mu_constraint = 100.
        solver.mu_dynamic = 100.
        solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
        solver.use_filter_line_search = False
        solver.verbose = True
        solver.termination_tolerance = 1e-3
        solver.remove_reg = False
        solver.max_qp_iters = 25000
        flag = solver.solve(xs_init, us_init, maxiter=maxIter, isFeasible=False)

    elif solver_type == "FDDP":
        solver = crocoddyl.SolverBoxFDDP(problem)
        solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
        flag = solver.solve(xs_init, us_init, maxIter, False)

    else:
        exit("Invalid solver type")

    xs, us = solver.xs, solver.us
    log = solver.getCallbacks()[-1]
    print(f'Solved: {flag}')

    #Extract data from the solver for visualizations
    runningDatas = problem.runningDatas
    runningModels = problem.runningModels
    terminalData = problem.terminalData
    terminalModel = problem.terminalModel

    print(f'Final state: \n {xs[-1]}')
    if solver_type == "CSQP":
        mim_solvers.plotConvergence(log.convergence_data)
    elif solver_type == "FDDP":
        crocoddyl.plotConvergence(log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2)
    crocoddyl.plotOCSolution(xs, us)

    input("Press to display")

    forces = np.array([runningModel.forces[:, :3] for runningModel in problem.runningModels])
    contacts = [runningModel.contacts for runningModel in problem.runningModels]
    formatter = {'float_kind': lambda x: "{:.4f}".format(x)}
    np.set_printoptions(linewidth=210, precision=4, suppress=False, formatter=formatter)

    from pinocchio.visualize import MeshcatVisualizer
    viz = MeshcatVisualizer(rmodel, pin_env["gmodel"], pin_env["vmodel"])

    import imageio
    frames = []
    fps = int(0.5/dt) 

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
    PLOT_DIRECTORY = "visualizations/plots/ocp/"
    VIDEO_DIRECTORY = "visualizations/videos/ocp/"
    IMAGE_DIRECTORY = "visualizations/frames/"  # Directory to store individual frames
    with_clearance = "with_clearance_"
    TASK = "walking_"
    foots_velocities = []
    import os
    import subprocess
    if not os.path.exists(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY)

    input("Press to display")
    for i in range(len(fids)):
        arrows.append(Arrow(viz.viewer, "force" + str(i), length_scale=0.001))

    for i in range(len(xs)-1):
        runningData = runningDatas[i]
        runningModel = runningModels[i]
        x_t = xs[i]
        print(f"\n********************Time:{i*dt}********************\n")
        print(f'Controls:{us[i][:njoints]}')
        print(f'Base position:{x_t[:3]}')
        print(f'Contacts:\n {contacts[i]}')
        foots_velocity = []
        for eff, fid in enumerate(fids):
            q, v = x_t[:rmodel.nq], x_t[rmodel.nq:]
            pin.framesForwardKinematics(rmodel, rdata, q)
            pin.computeAllTerms(rmodel, rdata, q, v)
            pin.updateFramePlacements(rmodel, rdata)
            print(f'foot id:{eff}')
            print(f'forces: {forces[i][eff]}')
            print(f'distance:{rdata.oMf[fid].translation[2] - GO2_FOOT_RADIUS}')

            name = "footClearance_" + str(eff)

            pin.forwardKinematics(rmodel, rdata, q, v)

            foot_velocity = pin.getFrameVelocity(rmodel, rdata, fid, pin.LOCAL_WORLD_ALIGNED)
            foots_velocity.append(foot_velocity.linear)
            print(f'foot velocity: {foot_velocity.linear}')
            foot_clearance_cost = runningData.differential.costs.costs[name].cost
            foot_clearance_weight = runningModel.differential.costs.costs[name].weight

            print(f'foot clearance cost: {foot_clearance_cost * foot_clearance_weight}\n')
            
            ureg_cost = runningData.differential.costs.costs["uReg"].cost
            ureg_weight = runningModel.differential.costs.costs["uReg"].weight
            force_eff = forces[i][eff]
            arrows[eff].anchor_as_vector(rdata.oMf[fid].translation, force_eff)    

        foots_velocities.append(foots_velocity)

        viz.display(xs[i][:rmodel.nq])
        frame = np.asarray(viz.viewer.get_image())  # Modify this line to capture frames properly in the right format
        imageio.imwrite(os.path.join(IMAGE_DIRECTORY, f'frame_{i:04d}.png'), frame)

        print(f'number of frames: {len(frames)}')
        print(f'\nureg_cost: {ureg_cost * ureg_weight} ')
        print(f'__________________________________________')
    
    print(f'Final cost: {terminalData.differential.costs.costs["xDes_terminal"].cost * terminalModel.differential.costs.costs["xDes_terminal"].weight}')
    
    plot_normal_forces(forces, fids, dt, output_file=PLOT_DIRECTORY + TASK +with_clearance + "normal_forces_" + solver_type + ".pdf")

    positions = [x[:3] for x in xs]  # Extract the first three elements (positions) from each array in xs
    orientations = [x[3:7] for x in xs]  # Extract elements 3 to 6 (quaternions) from each array in xs
    plot_base_poses(positions, orientations, dt, output_file= PLOT_DIRECTORY+ TASK + with_clearance+ "base_pose_" + solver_type + ".pdf")

    plot_sliding(foots_velocities, forces, fids, dt, output_file= PLOT_DIRECTORY+ TASK+with_clearance+"sliding_" + solver_type + ".pdf")

    # Call FFmpeg manually to convert the images into a video with the probesize option
    output_file = VIDEO_DIRECTORY + TASK + with_clearance + "CSQP" + "_1ms.mp4"
    subprocess.call([
        'ffmpeg', '-y', '-probesize', '50M', '-framerate', str(fps),
        '-i', os.path.join(IMAGE_DIRECTORY, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_file
    ])

    print(f"Video saved to {output_file}")
    
    save = input("Save trajectory? (y/n)")
    if save == "y":
        save_arrays(xs, us, "precomputed_trajectory_" + solver_type)


if __name__   == "__main__":
    main()
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../robots/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'python/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'robots/')))
from robots.robot_env import create_trifinger_cube_env
import crocoddyl
import pinocchio as pin
import numpy as np
from CallbackLogger_force import CallbackLogger
from utils import add_noise_to_state
import mim_solvers
import meshcat
import mim_solvers
import sys
import cito



def main():
    cube_position_index = 9
    cube_velocity_index = 16 + 9
   
    pin_model, gmodel, vmodel, mj_model, cube_frame_id = create_trifinger_cube_env()

    pin_data = pin_model.createData()
    

    nq = pin_model.nq
    nv = pin_model.nv

    v0 = list(np.zeros(nv))

    nu = 9
    ################# OCP params ################
    dt = 0.005                              
    mj_model.opt.timestep = dt                
    T = 200                                              
    T_total = dt * T                          

    FINGER_CONFIGURATION = [0.0, -0.6, -1.2]


    q_finger = np.array(3 * FINGER_CONFIGURATION)

    q_cube_mj = np.array([0, 0, 0.00001, 
                          
                          1, 0, 0, 0])
    
    q_cube_pin = np.array([0, 0, 0.00001, 
                          
                          0, 0, 0, 1])
    


    q0 = list(np.hstack((q_finger, q_cube_pin)))

    # print(f'x0:{q0 + v0}')
    ################# Initialize crocoddyl models ################
    state = crocoddyl.StateMultibody(pin_model)
    
    # actuation = crocoddyl.ActuationModelFloatingBase(state)

    actuation = cito.ActuationModelFloatingBaseManipulation(state)

    print(f"NU: {actuation.nu}")
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)
    constraintModelManager = crocoddyl.ConstraintModelManager(state, nu)    


    ################ Cube Target ################
    Px_des = 0.0
    Py_des = -0.2
    Pz_des = 0.0
    CubeTarget = np.array([Px_des, Py_des, Pz_des])


    CubePositionActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1, 1, 1]))
    CubePositionResidual = crocoddyl.ResidualModelFrameTranslation(state, id=cube_frame_id, xref=CubeTarget, nu=9)
    CubePositionCost = crocoddyl.CostModelResidual(state, CubePositionActivation, CubePositionResidual)
    runningCostModel.addCost("Cube_position_cost", CubePositionCost, 10)


    terminalCostModel.addCost("Cube_position_cost", CubePositionCost, 1000)

    Vx_des = Px_des/T_total
    Vy_des = Py_des/T_total
    Vz_des = Pz_des/T_total

    ################## Control Regularization Cost##################
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    uRegActivation = crocoddyl.ActivationModelWeightedQuad(np.array(9 * [1.0]))
    uRegCost = crocoddyl.CostModelResidual(state, uRegActivation, uResidual)
    runningCostModel.addCost("uReg", uRegCost, 1e-3)


    runningModels = [cito.IntegratedActionModelContactMj(
                    cito.DifferentialActionModelContactMj(mj_model, state, actuation, runningCostModel, constraintModelManager), dt) 
                    for _ in range(T)]
    
    terminal_DAM = cito.DifferentialActionModelContactMj(mj_model, state, actuation, terminalCostModel, None)
    
    terminalModel = cito.IntegratedActionModelContactMj(terminal_DAM, 0.)

    x0 = np.array(q0 + v0)
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)

    num_steps = T + 1 

    xs_init = [np.copy(x0) for _ in range(num_steps)]
    us_init = [np.zeros(nu) for i in range(T)]

    base_x_start = x0[cube_position_index].copy()
    base_x_end = Px_des
    base_x_values = np.linspace(base_x_start, base_x_end, num_steps)

    base_y_start = x0[cube_position_index+1].copy()
    base_y_end = Py_des
    base_y_values = np.linspace(base_y_start, base_y_end, num_steps)

    base_z_start = x0[cube_position_index+2].copy()
    base_z_end = Pz_des
    base_z_values = np.linspace(base_z_start, base_z_end, num_steps)

    for i in range(1, num_steps):
            xs_init[i][cube_position_index] = base_x_values[i] 
            xs_init[i][cube_position_index+1] = base_y_values[i] 
            xs_init[i][cube_position_index+2] = base_z_values[i] 
            
            xs_init[i][cube_velocity_index] = Vx_des #assign desired base velocity to initial guess
            xs_init[i][cube_velocity_index+1] = Vy_des #assign desired base velocity to initial guess
            xs_init[i][cube_velocity_index+2] = Vz_des #assign desired base velocity to initial guess

            xs_init[i] = add_noise_to_state(pin_model, xs_init[i], scale=0.0005).copy()

    maxIter = 100

    solver = mim_solvers.SolverCSQP(problem)

    solver.mu_constraint = 10.
    solver.mu_dynamic = 10.
    solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
    solver.use_filter_line_search = False
    solver.verbose = True
    solver.termination_tolerance = 1e-3
    solver.remove_reg = False
    solver.max_qp_iters =500

    # solver = crocoddyl.SolverFDDP(prolem)
    # solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])

    print(f'Solving')
    xs_init
    flag = solver.solve(xs_init, us_init, maxIter, False)

    xs, us = solver.xs, solver.us
    log = solver.getCallbacks()[-1]
    print(f'Solved: {flag}')


    from meshcat.animation import Animation
    import meshcat.transformations as tf    

    from pinocchio.visualize import MeshcatVisualizer
    viz = MeshcatVisualizer(pin_model, gmodel, vmodel)
    try:
        viz.initViewer(open=True)
    except ImportError as err:
        print(err)
        sys.exit(0)

    

    print(f'cube_final_position:{xs[-1][cube_position_index:cube_position_index+3]}')
#################Visualizations storage settings############################################################
    import imageio
    import os
    import subprocess
    
    frames = []
    fps = int(1/dt) 
    fps = int(fps/2) # Using half speed w.r.t. real-time
    TRAJECTORY_DIECTORY = "trajectories/"
    PLOT_DIRECTORY = "visualizations/plots/ocp/"
    VIDEO_DIRECTORY = "visualizations/videos/ocp/"
    IMAGE_DIRECTORY = "visualizations/frames/"  # Directory to store individual frames for video creation
    TASK = "walking_"                          # Task name for the video
    if not os.path.exists(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY)

####################################################################################################

    # save_arrays(xs, us, TRAJECTORY_DIECTORY + "precomputed_trajectory_" + solver_type)
    viz.loadViewerModel()
    viz.initializeFrames()
    viz.display_frames = True
    arrows = []
    foots_velocities = []
    # fids = fids
    for i in range(len(xs)-1):
        runningModel = runningModels[i]
        x_t = xs[i]
        print(f"\n********************Time:{i*dt}********************\n")
        print(f'Finger 1 controls:{us[i][:3]}')
        print(f'Finger 1 controls:{us[i][3:6]}')
        print(f'Finger 1 controls:{us[i][6:9]}')
        print(f'Cube position:{x_t[cube_position_index:cube_position_index+3]}')
        # print(f'Contacts:\n {contacts[i]}')
        foots_velocity = []
        # for eff, fid in enumerate(fids):
        #     q, v = x_t[:pin_model.nq], x_t[pin_model.nq:]
        #     pin.framesForwardKinematics(pin_model, pin_data, q)
        #     pin.computeAllTerms(pin_model, pin_data, q, v)
        #     pin.updateFramePlacements(pin_model, pin_data)
           

        #     pin.forwardKinematics(pin_model, pin_data, q, v)

        #     # foot_velocity = pin.getFrameVelocity(pin_model, pin_data, fid, pin.LOCAL_WORLD_ALIGNED)
        #     # foots_velocity.append(foot_velocity.linear)
        #     # print(f'foot velocity: {foot_velocity.linear}')
            

        # foots_velocities.append(foots_velocity)
        
        # viz.setCameraTarget(np.array([x_t[0]+0.1, 0.0, 0.0]))
    

        # viz.setCameraPosition(np.array([0.5, -1.0, 0.3]))

        viz.display(xs[i][:pin_model.nq])
        frame = np.asarray(viz.viewer.get_image())  # Modify this line to capture frames properly in the right format
        imageio.imwrite(os.path.join(IMAGE_DIRECTORY, f'frame_{i:04d}.png'), frame)

        print(f'number of frames: {len(frames)}')
        # print(f'\nureg_cost: {ureg_cost * ureg_weight} ')
        print(f'__________________________________________')
    
    # print(f'Final cost: {terminalData.differential.costs.costs["xDes_terminal"].cost * terminalModel.differential.costs.costs["xDes_terminal"].weight}')
    
    cube_position_index = 9

    positions = [x[cube_position_index : cube_position_index+3] for x in xs]  # Extract the first three elements (positions) from each array in xs
    orientations = [x[cube_position_index+3 : cube_position_index+7] for x in xs]  # Extract elements 3 to 6 (quaternions) from each array in xs


    # Call FFmpeg manually to convert the images into a video with the probesize option
    output_file = VIDEO_DIRECTORY + TASK + "CSQP" + ".mp4"
    subprocess.call([
        'ffmpeg', '-y', '-probesize', '50M', '-framerate', str(fps),
        '-i', os.path.join(IMAGE_DIRECTORY, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_file
    ])

    print(f"Video saved to {output_file}")


if __name__   == "__main__":
    main()




    



    

    
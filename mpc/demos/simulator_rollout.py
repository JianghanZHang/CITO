import mujoco 
from simulators.go2_mujoco import Go2Sim
import numpy as np
import os
import sys
sys.path.append('/home/jianghan/Devel/workspace_autogait/src/auto_gait_generation')

sys.path.append('.')
import pinocchio as pin
from launch_utils import load_config_file
# from utils import stateMapping_pin2mj
import mediapy as media
from utils import load_arrays, extend_trajectory

URDF_PATH = os.path.join(os.path.dirname(__file__), "../../robots/go2_robot_sdk/urdf/go2.urdf")
PACKAGE_DIRS = ["/home/jianghan/Devel/workspace_autogait/src/auto_gait_generation/robots"]

# Use different XML files for OCP and simulation due to different time steps
OCP_XML_PATH = os.path.join(os.path.dirname(__file__), '../../robots/unitree_go2/scene_foot_collision.xml')
SIM_XML_PATH = os.path.join(os.path.dirname(__file__), '../../robots/unitree_go2/scene_simulation.xml')

def main():
    
    # Initialize the controller and robot model
    pin_robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, PACKAGE_DIRS, root_joint= pin.JointModelFreeFlyer(), verbose=True)
    mj_model = mujoco.MjModel.from_xml_path(OCP_XML_PATH)
    nq = pin_robot.model.nq
    nv = pin_robot.model.nv
    config = load_config_file('walking_go2_rollout')
    print(f'Finished loading configuration')
    dt_sim = config['dt_ctrl']
    dt_ocp = config['dt_ocp']
    ratio_ocp_sim = int(dt_ocp / dt_sim)
    total_time = config['T_total']
    num_steps = int(total_time / dt_sim)

    Kp = config['Kp'] * np.ones(nq-7)
    Kd = config['Kd'] * np.ones(nq-7)

    # Load configuration and precomputed trajectory


    # Assuming the precomputed trajectory contains [q_des_mj, v_des_mj, tau_ff] for each time step
    x_des_trajectory, tau_ff_trajectory = load_arrays('precomputed_trajectory_CSQP')  # Load trajectory from file
    q_des_trajectory = []
    v_des_trajectory = []
    
    x_des_trajectory, tau_ff_trajectory = extend_trajectory(x_des_trajectory, tau_ff_trajectory, ratio_ocp_sim)

    for i in range(1, len(x_des_trajectory)):
        q_des_trajectory.append(x_des_trajectory[i][7:nq])
        v_des_trajectory.append(x_des_trajectory[i][nq+6:])    
    
    # Get simulation parameters

    # Initialize the simulator
    simulator = Go2Sim(mode='lowlevel', render=False, dt=dt_sim, xml_path=SIM_XML_PATH)
    print(f'Finished initializing simulator')

    for step in range(num_steps):
        print(f'Sim Step: {step}')

        # Get the precomputed control inputs for this step
        OCP_step = int(step / ratio_ocp_sim)
        print(f'OCP step: {OCP_step}')
        q_joints_des_mj = q_des_trajectory[step][:]
        v_joints_des_mj = v_des_trajectory[step][:]
        tau_ff = tau_ff_trajectory[step]

        # (Optional) Log or visualize current state
        Position, _ = simulator.getPose()
        
        print(f'current body position: \n {Position}')
        print(f'tau_ff: \n {tau_ff}')

        # Set the commands directly from the precomputed trajectory
        simulator.setCommands(q_joints_des_mj, v_joints_des_mj, Kp, Kd, tau_ff)

        # Step the simulation
        simulator.stepLowlevel()
        
        print(f'Control signal:\n {simulator.actuator_tau}')
        print(f'Finished step {step}')
    
    frames = simulator.frames.copy()
    # media.show_video(frames)
    media.write_video('go2_walking_rollout_trajectory.mp4', frames)

if __name__ == "__main__":
    main()

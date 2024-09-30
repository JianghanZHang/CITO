import mujoco 
from simulators.go2_mujoco import Go2Sim
import numpy as np
import os
import sys
sys.path.append('.')
from controllers.walking_csqp import Go2WalkingCSQP
from controllers.walking_fddp import Go2WalkingFDDP
import pinocchio as pin
from launch_utils import load_config_file
from utils import stateMapping_pin2mj
import mediapy as media

URDF_PATH = os.path.join(os.path.dirname(__file__), "../../robots/go2_robot_sdk/urdf/go2.urdf")
PACKAGE_DIRS = ["/home/jianghan/Devel/workspace_autogait/src/auto_gait_generation/robots"]

# Use different XML files for OCP and simulation due to different time steps
OCP_XML_PATH = os.path.join(os.path.dirname(__file__), '../../robots/unitree_go2/scene_foot_collision.xml')
SIM_XML_PATH = os.path.join(os.path.dirname(__file__), '../../robots/unitree_go2/scene_simulation.xml')

def main():
    
    # initialize the controller
    pin_robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, PACKAGE_DIRS, root_joint= pin.JointModelFreeFlyer(), verbose=True)
    mj_model = mujoco.MjModel.from_xml_path(OCP_XML_PATH)


    # config = load_config_file('walking_go2_CSQP')
    # controller = Go2WalkingCSQP(pin_robot, mj_model, config, True)
    config = load_config_file('walking_go2_FDDP')
    controller = Go2WalkingFDDP(pin_robot, mj_model, config, True)
    print(f'Finished initializing controller')

    nq = pin_robot.model.nq
    nv = pin_robot.model.nv

    dt_sim = controller.dt_ctrl

    # initialize the simulator
    simulator = Go2Sim(mode='lowlevel', render=False, dt=dt_sim, xml_path=SIM_XML_PATH)
    print(f'Finished initializing simulator')
    total_time = controller.config['T_total']
    num_steps = int(total_time / dt_sim)

    Kp = np.asarray(controller.config['Kp'])
    Kd = np.asarray(controller.config['Kd'])
    # Kp, Kd = np.zeros(12), np.zeros(12) 
    
    #TODO: Add a warm-up phase to the controller
    controller.warmUp()
    print(f'Finished warm-up')

    for step in range(num_steps):
        print(f'Step: {step}')
        q_mj, v_mj = simulator.getState()

        controller.updateState(q_mj, v_mj)

        tau_ff, xnext_pin, cost = controller.computeControl()

        # import pdb; pdb.set_trace()
        x_next_mj, _, _ = stateMapping_pin2mj(xnext_pin, pin_robot.model)

        q_des_mj, v_des_mj = x_next_mj[7:nq], x_next_mj[nq+6:]

        current_state = np.concatenate([controller.joint_positions, controller.joint_velocities])
        print(f'current state: \n {current_state}')
        print(f'Cost: \n{cost}')
        print(f'tau_ff: \n {tau_ff}')
        simulator.setCommands(q_des_mj, v_des_mj, Kp, Kd, tau_ff)

        simulator.stepLowlevel()
        
        print(f'control signal:\n {simulator.actuator_tau}')
        print(f'Finished step {step}')
    
    frames = simulator.frames.copy()
    # media.show_video(frames)
    media.write_video('go2_walking_FDDP_withClearanc_harder_simulator.mp4', frames)

if __name__ == "__main__":
    main()
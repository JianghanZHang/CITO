# Adjust these imports to match where you have placed the trifinger code:
MODEL_PATH = 'models/nyufinger/trifinger_nyu_elephant_scene.xml'
CONFIG_PATH = 'configs/mppi_trifinger_planar_push.yml'
SIM_PATH = 'models/nyufinger/trifinger_nyu_elephant_scene.xml'
import numpy as np

import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from control.controllers.mppi_manipulation import manipulation_MPPI
from interface.simulator import Simulator


def main():
    # ---------------------------
    # Simulation and Controller Parameters
    # ---------------------------
    T = 2000  # total steps, e.g. 20 seconds if dt=0.01

    position_threshold = 0.01
    orientation_threshold = 0.2

    total_trail = 100
    success = 0

    VIEWER = True
    SIMULATION_STEP = 0.01
    CTRL_UPDATE_RATE = 100     # control update frequency
    # Soft contact model parameters
    TIMECONST = 0.02
    DAMPINGRATIO = 1.0
    simulator = Simulator(
        agent=None,
        viewer=VIEWER,
        T=T,
        dt=SIMULATION_STEP,
        timeconst=TIMECONST,
        dampingratio=DAMPINGRATIO,
        model_path=SIM_PATH,
        ctrl_rate=CTRL_UPDATE_RATE
    )
    rng = np.random.default_rng(101)  # Create a Generator with a fixed seed


    for i in range(total_trail):
        print("\n------------------------------------------------")
        print(f"Trail {i}")
        
        x = rng.uniform(-0.06, 0.06)
        y = rng.uniform(-0.06, 0.06)
        alpha = rng.uniform(-np.pi/2, np.pi/2)

        goal_position = (x, y)
        goal_orientation = alpha    
        
        task_data = {
            "object_state":[x, y, 0.03, # Postion - x, y, z
                            0, 0, alpha], # Orientation - roll, pitch, yaw

            "model_path": MODEL_PATH, # this is the one that controller gets.
            "config_path": CONFIG_PATH,
            "sim_path": SIM_PATH # this is the one that simulator gets.
        }

        agent = manipulation_MPPI(task = None, task_data=task_data)
        simulator.reset(agent)

        obj_position_idx = agent.nq_robot
        obj_orientation_idx = agent.nq_robot + 3

        observation = np.concatenate([simulator.data.qpos, simulator.data.qvel], axis=0)
        
        flag = False
        for t in range(T):
            action = agent.update(observation)
            qpos, qvel = simulator.step(action)
            observation = np.concatenate([qpos, qvel], axis=0)

            current_position = simulator.data.qpos[obj_position_idx : obj_position_idx+2]

            # The orientation is represented as a quaternion, so we need to convert it to euler angles to compare with the goal orientation
            quat = simulator.data.qpos[obj_orientation_idx : obj_orientation_idx+4]
            w, x, y, z = quat
            # Standard conversion for yaw (Z) from a quaternion.
            # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
            yaw = np.atan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z))
            current_orientation = yaw

            position_diff = np.linalg.norm(current_position - goal_position)
            orientation_diff = np.linalg.norm(current_orientation - goal_orientation)

            if orientation_diff <= orientation_threshold and position_diff <= position_threshold:
                print(f"Success at step {t}")
                flag = True
                success += 1
                break
        
        if flag == False:
            print(f"Failed to reach the goal position and orientation in {T} steps.")

    print(f"Success rate: {success}/{total_trail}")

if __name__ == "__main__":


    main()

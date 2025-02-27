"""
Task definitions for robot navigation and behavior scenarios.

Each task is represented as a dictionary containing key parameters:
- `goal_pos`: List of target positions in the format [x, y, z].
- `default_orientation`: Default orientation of the robot as a quaternion [w, x, y, z].
- `cmd_vel`: Commanded velocities in the format [linear x, linear y] in body frame.
- `goal_thresh`: Thresholds for achieving goals.
- `desired_gait`: Gait type for each phase of the task.
- `waiting_times`: Time in milliseconds to wait at each phase.
- `model_path`: Path to the robot's model file.
- `config_path`: Path to the robot's configuration file.
- `sim_path`: Path to the simulation file.
"""

import numpy as np
# DEFAULT_MODEL_PATH = 'models/trifinger/trifinger_scene.xml'
# DEFAULT_CONFIG_PATH = 'configs/mppi_trifinger_reaching.yml'
# DEFAULT_SIM_PATH = 'models/trifinger/trifinger_scene.xml'

DEFAULT_MODEL_PATH = 'models/nyufinger/trifinger_nyu_scene.xml'
DEFAULT_CONFIG_PATH = 'configs/mppi_trifinger_reaching.yml'
DEFAULT_SIM_PATH = 'models/nyufinger/trifinger_nyu_scene.xml'

# DEFAULT_MODEL_PATH = 'models/nyufinger/trifinger_nyu_nocollision.xml'
# DEFAULT_CONFIG_PATH = 'configs/mppi_trifinger_reaching.yml'
# DEFAULT_SIM_PATH = 'models/nyufinger/trifinger_nyu_nocollision.xml'


# MANIPULATION_MODEL_PATH = 'models/trifinger/trifinger_cube_scene.xml'
# MANIPULATION_CONFIG_PATH = 'configs/mppi_trifinger_manipulation.yml'
# MANIPULATION_SIM_PATH = 'models/trifinger/trifinger_cube_scene.xml'

MANIPULATION_MODEL_PATH = 'models/nyufinger/trifinger_nyu_cube_scene.xml'
MANIPULATION_CONFIG_PATH = 'configs/mppi_trifinger_manipulation.yml'
MANIPULATION_SIM_PATH = 'models/nyufinger/trifinger_nyu_cube_scene_simulation.xml'

PLANAR_PUSH_MODEL_PATH = 'models/nyufinger/trifinger_nyu_cube_scene.xml'
PLANAR_PUSH_CONFIG_PATH = 'configs/mppi_trifinger_planar_push.yml'
PLANAR_PUSH_SIM_PATH = 'models/nyufinger/trifinger_nyu_cube_scene_simulation.xml'


DEFAULT_ORIENTATION = [[1, 0, 0, 0]]

TASKS = {
    "reaching": {
        "finger_tips_pos": [[0.0, 0.0, 0.10],
                            [0.0, 0.0, 0.10],
                            [0.0, 0.0, 0.10]],

        "model_path": DEFAULT_MODEL_PATH,
        "config_path": DEFAULT_CONFIG_PATH,
        "sim_path": DEFAULT_SIM_PATH
    },

    "cube_manipulation": {
        # The center of the cube staying on the table is (0, 0, 0.013 = 0.125 + 0.005)
        "cube_state":[-0.0, 0.0, 0.06, # Postion - x, y, z
                      0, 0, np.pi/2], # Orientation - roll, pitch, yaw

        "model_path": MANIPULATION_MODEL_PATH, # this is the one that controller gets.
        "config_path": MANIPULATION_CONFIG_PATH,
        "sim_path": MANIPULATION_SIM_PATH # this is the one that simulator gets.
    },

    "cube_planar_push": {
        # The center of the cube staying on the table is (0, 0, 0.013 = 0.125 + 0.005)
        "cube_state":[0., -0.05, 0.03, # Postion - x, y, z
                        0, 0, np.pi/3], # Orientation - roll, pitch, yaw

        "model_path": PLANAR_PUSH_MODEL_PATH, # this is the one that controller gets.
        "config_path": PLANAR_PUSH_CONFIG_PATH,
        "sim_path": PLANAR_PUSH_SIM_PATH # this is the one that simulator gets.
    }
}

def get_task(task_name):
    """
    Retrieve task configuration by name.

    Args:
        task_name (str): Name of the task. Must be one of the keys in TASKS.

    Returns:
        dict: Task configuration dictionary.

    Raises:
        ValueError: If the task_name is not found in TASKS.
    """
    if task_name not in TASKS:
        raise ValueError(f"Task '{task_name}' not found. Available tasks: {list(TASKS.keys())}")
    return TASKS[task_name]

import os

import mujoco
import numpy as np
import yaml

# Local imports (ensure these are part of your package structure)
from control.controllers.base_controller import BaseMPPI
from scipy.spatial.transform import Rotation as R
from utils.tasks import get_task

# from utils.transforms import batch_world_to_local_velocity, calculate_orientation_quaternion
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NQ = 9

class manipulation_MPPI(BaseMPPI):
    """
    Model Predictive Path Integral (MPPI) Controller for quadruped robots.

    Attributes:
        - Task-specific parameters and goals.
        - Gait scheduler and configurations.
        - MPPI sampling and cost calculation configurations.
    """

    def __init__(self, task='manipulation') -> None:
        """
        Initialize the MPPI controller with task-specific configurations.

        Args:
            task (str): The name of the task ('stand', 'walk').
        """
        print("Task: ", task)

        # Retrieve task-specific parameters
        self.task = task
        self.task_data = get_task(task)

        model_path = self.task_data['model_path']
        config_path = self.task_data['config_path']
        # waiting_times = self.task_data['waiting_times']

        # Dynamically resolve paths for model and configuration files
        CONFIG_PATH = os.path.join(BASE_DIR, config_path)
        MODEL_PATH = os.path.join(BASE_DIR, "../..", model_path)

        # Initialize base MPPI
        super().__init__(MODEL_PATH, CONFIG_PATH)
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube_link')
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'cube_geom')
        self.model.body_mass[cube_id] = 0.2 # Increase cube mass perception to the controller for manipulation task
        # self.model.body_inertia[cube_id] = [0.2, 0.3, 0.4]
        # self.model.geom_friction[geom_id] = [0.4, 0.1, 0.005]
        # load the configuration file
        with open(CONFIG_PATH, 'r') as file:
            params = yaml.safe_load(file)

        # Cost weights
        self.Q = np.diag(np.array(params['Q_diag']))

        self.R = np.diag(np.array(params['R_diag']))

        self.W_frame_pos = np.diag(np.array(params['W_frame_pos']))

        self.W_cube_state = np.diag(np.array(params['W_cube_state']))

        # Set initial parameters and state
        self.obs = None
        self.internal_ref = True
        self.exp_weights = np.ones(self.n_samples) / self.n_samples  # Initial MPPI weights
        # self.waiting_times = waiting_times

        # Initialize planner and goals
        self.reset_planner()


        self.joints_ref_1d = np.hstack((self.model.key_qpos[0, :9], self.model.key_qvel[0, :9])) # Key qpos and qvel of the robot

        self.joints_ref = np.tile(self.joints_ref_1d[None, :], (self.horizon, 1))

        # self.cube_state_ref_1d = np.array(self.task_data['cube_state'])
        cube_position = self.task_data['cube_state'][:3]
        cube_orientation_rpy = self.task_data['cube_state'][3:]

        # Converting RPY to quaternion
        r = R.from_euler('xyz', cube_orientation_rpy, degrees=False)  # MuJoCo uses XYZ order
        quat_xyzw = r.as_quat()  # SciPy returns [x, y, z, w]
        quat_wxyz = np.roll(quat_xyzw, shift=1)  # Convert to [w, x, y, z]

        cube_orientation = quat_wxyz

        self.cube_state_ref_1d = np.hstack((cube_position, cube_orientation))
        self.cube_state_ref = np.tile(self.cube_state_ref_1d[None, :], (self.horizon, 1))

        self.task_success = False

    # @profile
    def update(self, obs):
        """
        Update the MPPI controller based on the current observation.

        Args:
            obs (np.ndarray): Current state observation.
        Returns:
            np.ndarray: Selected action based on the optimal trajectory.
        """
         # Generate perturbed actions for rollouts
        actions = self.perturb_action()
        self.obs = obs

        # Perform rollouts using threaded rollout function
        self.rollout_func(self.rollout_models, self.state_rollouts, actions, np.repeat(np.array([np.concatenate([[0], obs])]), self.n_samples, axis=0), self.sensor_datas, num_workers=self.num_workers, nstep=self.horizon)
        
        # self.rollout_func(self.state_rollouts, actions, np.repeat(
        #     np.array([np.concatenate([[0], obs])]), self.n_samples, axis=0), self.sensor_datas,
        #     num_workers=self.num_workers, nstep=self.horizon)


        # Calculate costs for each sampled trajectory
        costs_sum = self.cost_func(self.state_rollouts[:, :, 1:], actions, self.sensor_datas, self.joints_ref, self.cube_state_ref)

        # Calculate MPPI weights for the samples
        min_cost = np.min(costs_sum)
        max_cost = np.max(costs_sum)
        self.exp_weights = np.exp(-1 / self.temperature * ((costs_sum - min_cost) / (max_cost - min_cost)))

        # Weighted average of action deltas
        weighted_delta_u = self.exp_weights.reshape(self.n_samples, 1, 1) * actions
        weighted_delta_u = np.sum(weighted_delta_u, axis=0) / (np.sum(self.exp_weights) + 1e-10)
        updated_actions = np.clip(weighted_delta_u, self.act_min, self.act_max)

        # Update the trajectory with the optimal action
        self.selected_trajectory = updated_actions
        self.trajectory = np.roll(updated_actions, shift=-1, axis=0)
        self.trajectory[-1] = updated_actions[-1]

        # Return the first action in the trajectory as the output action
        return updated_actions[0]

    def quaternion_distance_np(self, q1, q2):
        """
        Compute the distance between two sets of quaternions.

        Args:
            q1 (np.ndarray): Array of quaternions (N x 4).
            q2 (np.ndarray): Array of quaternions (N x 4).

        Returns:
            np.ndarray: Array of distances between the quaternions.
        """
        # Compute dot product between corresponding quaternions
        dot_products = np.einsum('ij,ij->i', q1, q2)
        # Compute distance as 1 - absolute dot product
        return 1 - np.abs(dot_products)


    def trifinger_cost_np(self, x, action, joints_ref, cube_state_ref, sensor_data):
        """
        Compute the cost for trifinger based on state, action, and some FK errors.

        Args:
            x (np.ndarray): Current states (N x state_dim).
            u (np.ndarray): Current actions (N x action_dim).
            x_ref (np.ndarray): Reference states (N x state_dim).
            sensor_data: Current sensor data (N x sensor_dim=9; 3 for each tip position)
            sensor_data_ref: Reference sensor data (N x sensor_dim)

        Returns:
            np.ndarray: Computed cost for each sample.
        """
        kp = 1  # Proportional gain for joint error
        kd = 0   # Derivative gain for joint velocity error

        # Compute state error relative to the reference
        q_joint = x[:, :NQ]
        v_joint = x[:, NQ+7:2*NQ+7]

        joints_state = np.hstack((q_joint, v_joint))

        joints_error = joints_state - joints_ref

        cube_state = x[:, NQ:NQ+7]

        # TODO: Fix this state error calculation, now the quaternion error is wrong (Need to in the quaternion space)
        cube_state_error = cube_state - cube_state_ref

        tips_frame_pos = sensor_data[:, :9]
        # tips_frame_pos_ref = sensor_data_ref[:, :9]

        # Set the reference position of the tips frame to be the center of the cube
        tips_frame_pos_ref = np.tile(cube_state[:, :3], (1,3))

        tips_frame_pos_error = tips_frame_pos - tips_frame_pos_ref

        # Compute joint and velocity errors
        x_joint = x[:, :NQ]
        v_joint = x[:, NQ+7:2*NQ+7]
        u_error = kp * (action - x_joint) - kd * v_joint

        # L2_norm_tips_pos_cost = np.einsum('ij,ik,jk->i', tips_frame_pos_error, tips_frame_pos_error, self.W_frame_pos)

        # cube_state_error[-1]

        # Give more weight to the terminal error
        cube_state_error[self.horizon-1::self.horizon, :] *= self.horizon
        # tips_frame_pos_error[self.horizon-1::self.horizon, :] *= 10

        L1_norm_cube_state_cost = np.abs(np.dot(cube_state_error, self.W_cube_state)).sum(axis=1)
        L1_norm_tips_pos_cost = np.abs(np.dot(tips_frame_pos_error, self.W_frame_pos)).sum(axis=1)


        # # Compute positional cost (L1 norm for positional error)
        # L1_norm_tips_pos_cost = np.abs(np.dot(tips_frame_pos_error, self.Q[:3, :3])).sum(axis=1)
        # Compute total cost
        # state_error = np.einsum('ij,ik,jk->i', joints_error, joints_error, self.Q)

        # control_error = np.einsum('ij,ik,jk->i', u_error, u_error, self.R)

        cost = (
            np.einsum('ij,ik,jk->i', joints_error, joints_error, self.Q) +
            np.einsum('ij,ik,jk->i', u_error, u_error, self.R) +
            L1_norm_tips_pos_cost+
            L1_norm_cube_state_cost
            # control_error
        )

        return cost


    def calculate_total_cost(self, states, actions, sensor_datas, joints_ref, cube_state_ref):
        """
        Calculate the total cost for all rollouts.

        Args:
            states (np.ndarray): Rollout states (samples x time steps x state_dim).
            actions (np.ndarray): Rollout actions (samples x time steps x action_dim).
            joints_ref (np.ndarray): Reference joint positions (time steps x joint_dim).
            body_ref (np.ndarray): Reference body state (state_dim).

        Returns:
            np.ndarray: Total cost for each sample.
        """
        num_samples = states.shape[0]
        num_pairs = states.shape[1]


        # Flatten states and actions for batch processing
        states = states.reshape(-1, states.shape[2])
        actions = actions.reshape(-1, actions.shape[2])
        sensor_datas = sensor_datas.reshape(-1, sensor_datas.shape[2])

        joints_ref = np.tile(joints_ref, (num_samples, 1))


        cube_state_ref = np.tile(cube_state_ref, (num_samples, 1))

        # Compute cost for each rollout
        costs = self.trifinger_cost_np(states, actions, joints_ref, cube_state_ref, sensor_datas)

        # Sum costs across time steps for each sample
        total_costs = costs.reshape(num_samples, num_pairs).sum(axis=1)


        return total_costs

    def eval_best_trajectory(self):
        """
        Evaluate the cost of the best trajectory selected by MPPI.

        Returns:
            float: Cost of the best trajectory, or None if no observation is available.
        """
        if self.obs is None:
            # If no observation is available, return None
            return None
        else:
            # Create a rollout array for the best trajectory
            best_rollouts = np.zeros((1, self.horizon, mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)))
            # Perform rollout for the best trajectory
            sensor_data_rollout = np.zeros((1, self.horizon, self.sensor_data_size))

            self.rollout_func(self.rollout_model,
                              best_rollouts,
                              np.array([self.selected_trajectory]),
                              np.repeat(np.array([np.concatenate([[0],self.obs])]), 1, axis=0),
                              sensor_data_rollout,
                              num_workers=self.num_workers,
                              nstep=self.horizon)

            # self.rollout_func(best_rollouts,
            #                   np.array([self.selected_trajectory]),
            #                   np.repeat(np.array([np.concatenate([[0],self.obs])]), 1, axis=0),
            #                   sensor_data_rollout,
            #                   num_workers=self.num_workers,
            #                   nstep=self.horizon)

        # Compute and return the cost of the best trajectory
        return (self.cost_func(best_rollouts[:,:,1:],
                np.array([self.selected_trajectory]),
                sensor_data_rollout, self.joints_ref_1d,
                self.cube_state_ref_1d))[0]

    # def __del__(self):
    #     self.shutdown()

if __name__ == "__main__":

    mppi = manipulation_MPPI()

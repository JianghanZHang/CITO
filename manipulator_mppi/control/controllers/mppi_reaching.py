import os
import yaml
import mujoco
import numpy as np

# Local imports (ensure these are part of your package structure)
from utils.tasks import get_task
from control.controllers.base_controller import BaseMPPI
# from utils.transforms import batch_world_to_local_velocity, calculate_orientation_quaternion
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys


class reaching_MPPI(BaseMPPI):
    """
    Model Predictive Path Integral (MPPI) Controller for quadruped robots.

    Attributes:
        - Task-specific parameters and goals.
        - Gait scheduler and configurations.
        - MPPI sampling and cost calculation configurations.
    """

    def __init__(self, task='reaching') -> None:
        """
        Initialize the MPPI controller with task-specific configurations.

        Args:
            task (str): The name of the task ('stand', 'walk').
        """
        print("Task: ", task)
        self.nq_robot = 9

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

        # load the configuration file
        with open(CONFIG_PATH, 'r') as file:
            params = yaml.safe_load(file)

        # Cost weights
        self.Q = np.diag(np.array(params['Q_diag']))

        self.R = np.diag(np.array(params['R_diag']))

        self.W_frame_pos = np.diag(np.array(params['W_frame_pos']))

        # Set initial parameters and state
        self.obs = None
        self.internal_ref = True
        self.exp_weights = np.ones(self.n_samples) / self.n_samples  # Initial MPPI weights
        # self.waiting_times = waiting_times
      
        # Initialize planner and goals
        self.reset_planner()

        self.tips_frame_pos_ref_1d = np.array(self.task_data['finger_tips_pos']).flatten()

        self.tips_frame_pos_ref = np.tile(self.tips_frame_pos_ref_1d[None, :], (self.horizon, 1))

        self.joints_ref_1d = np.array([0.0, 0.7, -1.4] * 3 + [0.0, 0.0, 0.0] * 3)  # shape (18,)
        self.joints_ref = np.tile(self.joints_ref_1d[None, :], (self.horizon, 1))

        self.task_success = False


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

        # print('Entering rollout from update()')
        # import pdb; pdb.set_trace()
        # Perform rollouts using threaded rollout function
        self.rollout_func(self.state_rollouts, actions, np.repeat(
            np.array([np.concatenate([[0], obs])]), self.n_samples, axis=0), self.sensor_datas,
            num_workers=self.num_workers, nstep=self.horizon)


        # Calculate costs for each sampled trajectory
        costs_sum = self.cost_func(self.state_rollouts[:, :, 1:], actions, self.sensor_datas, self.joints_ref, self.tips_frame_pos_ref)
        
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

        # import pdb; pdb.set_trace()
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


    def trifinger_cost_np(self, x, action, x_ref, sensor_data, sensor_data_ref):
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
        kp = 0  # Proportional gain for joint error
        kd = 0   # Derivative gain for joint velocity error
        
        # Compute state error relative to the reference
        x_error = x - x_ref

        # tip_frame_pos_0 = sensor_data[:3]
        # tip_frame_pos_120 = sensor_data[3:6]
        # tip_frame_pos_240 = sensor_data[6:9]

        # tip_frame_pos_0_ref = sensor_data_ref[:3]
        # tip_frame_pos_120_ref = sensor_data_ref[3:6]
        # tip_frame_pos_240_ref = sensor_data_ref[6:9]

        tips_frame_pos = sensor_data[:, :9]
        tips_frame_pos_ref = sensor_data_ref[:, :9]


        tips_frame_pos_error = tips_frame_pos - tips_frame_pos_ref


        # Compute joint and velocity errors
        x_joint = x[:, :self.nq_robot]
        v_joint = x[:, self.nq_robot:]
        u_error = kp * (action - x_joint) - kd * v_joint

        L2_norm_tips_pos_cost = np.einsum('ij,ik,jk->i', tips_frame_pos_error, tips_frame_pos_error, self.W_frame_pos)

        L1_norm_tips_pos_cost = np.abs(np.dot(tips_frame_pos_error, self.W_frame_pos)).sum(axis=1)

        # # Compute positional cost (L1 norm for positional error)
        # L1_norm_tips_pos_cost = np.abs(np.dot(tips_frame_pos_error, self.Q[:3, :3])).sum(axis=1)
        # Compute total cost
        state_error = np.einsum('ij,ik,jk->i', x_error, x_error, self.Q)

        control_error = np.einsum('ij,ik,jk->i', u_error, u_error, self.R)

        cost = (
            np.einsum('ij,ik,jk->i', x_error, x_error, self.Q) +
            np.einsum('ij,ik,jk->i', u_error, u_error, self.R) +
            L1_norm_tips_pos_cost
        )

        return cost


    def calculate_total_cost(self, states, actions, sensor_datas, joints_ref, tips_frame_pos_ref):
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

        # # Repeat and reshape joint references for batch processing
        # joints_ref = joints_ref.T
        # joints_ref = np.tile(joints_ref, (num_samples, 1, 1))
        # joints_ref = joints_ref.reshape(-1, joints_ref.shape[2])

        joints_ref = np.tile(joints_ref, (num_samples, 1))  

        # tips_frame_pos_ref = tips_frame_pos_ref.T
        tips_frame_pos_ref = np.tile(tips_frame_pos_ref, (num_samples, 1))

        # Concatenate body and joint references for full reference state
        x_ref = joints_ref

        # Compute cost for each rollout
        costs = self.trifinger_cost_np(states, actions, x_ref, sensor_datas, tips_frame_pos_ref)

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
            # import pdb; pdb.set_trace()
            # print('Entering rollout from eval_best_trajectory()')
            self.rollout_func(best_rollouts, np.array([self.selected_trajectory]), np.repeat(np.array([np.concatenate([[0],self.obs])]), 1, axis=0), sensor_data_rollout, num_workers=self.num_workers, nstep=self.horizon)

            # import pdb; pdb.set_trace()
            print(f'tip0 position{sensor_data_rollout[0, 0, 0:3]}')
            print(f'tip120 position{sensor_data_rollout[0, 0, 3:6]}')
            print(f'tip240 position{sensor_data_rollout[0, 0, 6:9]}')

        # Compute and return the cost of the best trajectory
        return (self.cost_func(best_rollouts[:,:,1:], np.array([self.selected_trajectory]), sensor_data_rollout, self.joints_ref_1d, self.tips_frame_pos_ref_1d))[0]

    def __del__(self):
        self.shutdown()
    
if __name__ == "__main__":
    mppi = reaching_MPPI()
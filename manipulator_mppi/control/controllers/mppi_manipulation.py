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


        # self.model.body_mass[cube_id] = 0.2 # Increase cube mass perception to the controller for manipulation task
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

        self.W_stability = np.diag(np.array(params['W_stability']))

        self.W_tips_contact = np.diag(np.array(params['W_tips_contact']))
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

    def compute_quaternion_distance(self, q1, q2):
        """
        Compute the distance between two sets of quaternions.

        Args:
            q1 (np.ndarray): Array of quaternions (N x 4).
            q2 (np.ndarray): Array of quaternions (N x 4).

        Returns:
            np.ndarray: Array of distances between the quaternions.
        """
        size = q1.shape[0]
        # Compute dot product between corresponding quaternions
        dot_products = np.einsum('ij,ij->i', q1, q2)
        # Compute distance as 1 - absolute dot product

        distance = 1 - np.abs(dot_products)
        return distance.reshape(size, 1)

    def compute_orientation_distance(self, q1, q2):
        """
        Compute the orientation error between two sets of quaternions q1 and q2.
        
        Both q1 and q2 are expected to be numpy arrays of shape (N, 4) in (w, x, y, z) order.
        
        Returns:
            np.ndarray: An array of shape (N, 3) where each row is the rotation vector (axis*angle, in radians)
                        representing the minimal rotation that transforms q1 into q2.
        """
        # Convert from (w, x, y, z) to (x, y, z, w) for SciPy.
        q1_scipy = np.hstack((q1[:, 1:], q1[:, 0:1]))
        q2_scipy = np.hstack((q2[:, 1:], q2[:, 0:1]))
        
        # Create Rotation objects from the batch of quaternions.
        r1 = R.from_quat(q1_scipy)
        r2 = R.from_quat(q2_scipy)
        
        # Compute the relative rotation for each pair: r_err = r2 * inv(r1)
        r_err = r2 * r1.inv()
        
        # Return the rotation vector for each relative rotation.
        return r_err.as_rotvec()
    
    def compute_tips_distance(self, x1, x2):
        """
        Compute the per-finger-tip distance between two sets of states x1 and x2.

        Each state is a 9-element vector, with groups of 3 representing the xyz position
        of a finger tip. This function returns an (N, 3) array containing the distance for
        each finger tip.

        Args:
            x1 (np.ndarray): Array of states with shape (N, 9).
            x2 (np.ndarray): Array of states with shape (N, 9).

        Returns:
            np.ndarray: An array of shape (N, 3) where each column represents the distance
                        (L2 norm) for one finger tip.
        """
        # Reshape x1 and x2 to (N, 3, 3), where axis=1 indexes the finger tip
        # and axis=2 holds the x, y, z coordinates.
        x1_reshaped = x1.reshape(-1, 3, 3)
        x2_reshaped = x2.reshape(-1, 3, 3)

        # Compute the Euclidean distance for each finger tip (along the coordinate axis)
        distances = np.linalg.norm(x1_reshaped - x2_reshaped, axis=2)
        return distances
    
    def compute_cube_distance(self, x1, x2):
        # Calculate the element-wise difference
        diff = x1 - x2  # Shape: (N, 3)
        # Compute the Euclidean distance (norm) for each row and keep dimensions (N, 1)
        distance = np.linalg.norm(diff, axis=1, keepdims=True)
        return distance



    def compute_stability_cost(self, x_obj, x_tips):
        """
        Compute the grasp stability cost based on fingertip directions relative to the object.
        stablity cost: SUM(R_obj.T (p_tip_i - p_obj) / ||p_tip_i - p_obj||)
        
        Parameters:
            x_obj: numpy array of shape (N, 7). For each sample, the first 3 columns are the object position,
                and the next 4 are the quaternion [w, x, y, z] (object orientation).
            x_tips: numpy array of shape (N, 3, 3). For each sample, the positions of the 3 fingertips (each a 3D vector).
        
        Returns:
            cost: numpy array of shape (N,). A lower cost indicates a more balanced (stable) grasp.
        """
        def quaternion_to_matrix(q):
            """
            Convert an array of quaternions into rotation matrices using mujoco.mju_quat2Mat.
            
            Parameters:
                q: numpy array of shape (N, 4), where each row is a quaternion [w, x, y, z].
            
            Returns:
                R: numpy array of shape (N, 3, 3) where each R[i] is the rotation matrix for q[i].
            """
            N = q.shape[0]
            R = np.empty((N, 3, 3))
            # MuJoCo expects the output rotation matrix as a flat array of 9 elements.
            mat = np.empty(9, dtype=np.float64)
            for i in range(N):
                # Convert the quaternion q[i] to a 3x3 rotation matrix (flattened into 9 elements)
                mujoco.mju_quat2Mat(mat, q[i])
                # Reshape the flat array into a 3x3 matrix and store it
                R[i] = mat.reshape(3, 3)
            return R
        

        # Extract object position and quaternion from the object state vector.
        pos_obj = x_obj[:, :3]    # (N, 3)
        quat_obj = x_obj[:, 3:]   # (N, 4)

        # Convert each quaternion to a 3x3 rotation matrix (object-to-world).
        R_obj = quaternion_to_matrix(quat_obj)  # (N, 3, 3)

        # Reshape x_tips from (N, 9) to (N, 3, 3) so that each sample has three 3D fingertip positions.
        x_tips_reshaped = x_tips.reshape(-1, 3, 3)  # (N, 3, 3)

        # Compute the difference between each fingertip position and the object center in the world frame.
        diff_world = x_tips_reshaped - pos_obj[:, None, :]  # (N, 3, 3)

        # Transform these difference vectors to the object frame using the transpose of R_obj.
        diff_obj = np.matmul(R_obj.transpose(0, 2, 1), diff_world)  # (N, 3, 3)

        # Normalize each difference vector to obtain unit vectors.
        norm = np.linalg.norm(diff_obj, axis=2, keepdims=True)  # (N, 3, 1)
        unit_diff_obj = diff_obj / (norm + 1e-8)

        # Sum the unit vectors for the 3 fingertips for each sample.
        sum_unit = np.sum(unit_diff_obj, axis=1)  # (N, 3)
        
        error = np.linalg.norm(sum_unit, axis=1).reshape(sum_unit.shape[0], 1)  # (N,)
        return error
    
    def compute_contact_cost(self, sensor_data, touch_idx_start, touch_idx_end):
        """
        Compute the contact cost based on the sensor data and the touch index.

        Args:
            sensor_data (np.ndarray): Sensor data (N x sensor_dim)
            touch_idx (np.ndarray): index of touch sensors.

        """
        inContact = sensor_data[:, touch_idx_start:touch_idx_end]
        inContact = (inContact == 0).astype(int)

        return inContact.sum(axis=1).reshape(inContact.shape[0], 1)

    
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

        # cube_position_error = cube_state[:,:3] - cube_state_ref[:,:3]
        # cube_orientation_error = self.compute_orientation_distance(cube_state[:,3:], cube_state_ref[:,3:])

        cube_position_error = self.compute_cube_distance(cube_state[:,:3], cube_state_ref[:,:3])
        cube_orientation_error = self.compute_quaternion_distance(cube_state[:,3:], cube_state_ref[:,3:])

        tips_frame_pos = sensor_data[:, :9]
        # Set the reference position of the tips frame to be the center of the cube
        tips_frame_pos_ref = np.tile(cube_state[:, :3], (1,3))
        tips_position_error = self.compute_tips_distance(tips_frame_pos, tips_frame_pos_ref)
        tips_object_stability_error = self.compute_stability_cost(cube_state, tips_frame_pos)
        tips_contact_error = self.compute_contact_cost(sensor_data, 12, 15)

        # Compute joint and velocity errors
        x_joint = x[:, :NQ]
        v_joint = x[:, NQ+7:2*NQ+7]
        u_error = kp * (action - x_joint) - kd * v_joint

        # Assign terminal cost
        # Set the some costs to zero except for the terminal node
        mask = np.zeros_like(cube_position_error)
        mask[self.horizon-1::self.horizon, :] = 1

        cube_position_error *= mask  # This sets all other rows to zero
        cube_orientation_error *= mask
        
        cube_position_error[self.horizon-1::self.horizon, :] *= self.horizon # Scale the selected rows
        cube_orientation_error[self.horizon-1::self.horizon, :] *= self.horizon

        L1_norm_cube_position_cost = np.abs(np.dot(cube_position_error, self.W_cube_state[:1, :1])).sum(axis=1)
        L1_norm_cube_orientation_cost = np.abs(np.dot(cube_orientation_error, self.W_cube_state[1:, 1:])).sum(axis=1)
        L1_norm_tips_position_cost = np.abs(np.dot(tips_position_error, self.W_frame_pos)).sum(axis=1)
        L1_norm_joint_cost = np.abs(np.dot(joints_error, self.Q)).sum(axis=1)  
        L1_norm_control_cost = np.abs(np.dot(u_error, self.R)).sum(axis=1)  
        L1_norm_stability_cost = np.abs(np.dot(tips_object_stability_error, self.W_stability)).sum(axis=1)  
        L1_norm_tips_contact_cost = np.abs(np.dot(tips_contact_error, self.W_tips_contact)).sum(axis=1)
        

        # # Compute positional cost (L1 norm for positional error)
        L2_norm_cube_position_cost = np.einsum('ij,ik,jk->i', cube_position_error, cube_position_error, self.W_cube_state[:1, :1])
        L2_norm_cube_orientation_cost = np.einsum('ij,ik,jk->i', cube_orientation_error, cube_orientation_error, self.W_cube_state[1:, 1:])
        L2_norm_tips_position_cost = np.einsum('ij,ik,jk->i', tips_position_error, tips_position_error, self.W_frame_pos)
        L2_norm_joint_cost = np.einsum('ij,ik,jk->i', joints_error, joints_error, self.Q)
        L2_norm_control_cost = np.einsum('ij,ik,jk->i', u_error, u_error, self.R) 
        L2_norm_stability_cost = np.einsum('ij,ik,jk->i', tips_object_stability_error, tips_object_stability_error, self.W_stability) 
        L2_norm_tips_contact_cost = np.einsum('ij,ik,jk->i', tips_contact_error, tips_contact_error, self.W_tips_contact) 

        cost = (
            L1_norm_joint_cost +
            L1_norm_control_cost +
            L1_norm_cube_orientation_cost+
            L1_norm_cube_position_cost+
            L1_norm_tips_position_cost+
            L1_norm_stability_cost+
            L1_norm_tips_contact_cost
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

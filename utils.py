## This class draws arrows to depic the force in meshcat
## Author : Huaijiang Zhu

import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf
import pinocchio as pin
import meshcat
import mujoco
from robot_env import ROOT_JOINT_INDEX

class Arrow(object):
    def __init__(self, meshcat_vis, name, 
                 location=[0,0,0], 
                 vector=[0,0,1],
                 length_scale=0.001,
                 color=0xff22dd):
        
        self.vis = meshcat_vis[name]
        self.cone = self.vis["cone"]
        self.line = self.vis["line"]
        self.material = g.MeshBasicMaterial(color=color, reflectivity=0.5)
        
        self.location, self.length_scale = location, length_scale
        self.anchor_as_vector(location, vector)
    
    def _update(self):
        translation = tf.translation_matrix(self.location)
        rotation = self.orientation
        offset = tf.translation_matrix([0, self.length/2, 0])
        self.pose = translation @ rotation @ offset
        self.vis.set_transform(self.pose)
        
    def set_length(self, length, update=True):
        self.length = length * self.length_scale
        cone_scale = self.length/0.08
        self.line.set_object(g.Cylinder(height=self.length, radius=0.005), self.material)
        self.cone.set_object(g.Cylinder(height=0.015, 
                                        radius=0.05, 
                                        radiusTop=0., 
                                        radiusBottom=0.01),
                             self.material)
        self.cone.set_transform(tf.translation_matrix([0.,cone_scale*0.04,0]))
        if update:
            self._update()
        
    def set_direction(self, direction, update=True):
        orientation = np.eye(4)
        if np.linalg.norm(direction - np.array([1.0, 0, 0])) < 1e-1 or np.linalg.norm(direction - np.array([-1.0, 0, 0])) < 1e-1:
            orientation[:3, 0] = np.cross([0.0,1.0,0], direction)
        else:
            orientation[:3, 0] = np.cross([1.0,0,0], direction)
        orientation[:3, 1] = direction.copy()
        orientation[:3, 2] = np.cross(orientation[:3, 0], orientation[:3, 1])
        self.orientation = orientation
        if update:
            self._update()
    
    def set_location(self, location, update=True):
        self.location = location
        if update:
            self._update()
        
    def anchor_as_vector(self, location, vector, update=True):
        self.set_direction(np.array(vector)/np.linalg.norm(vector), False)
        self.set_location(location, False)
        self.set_length(np.linalg.norm(vector), False)
        if update:
            self._update()

    def delete(self):
        self.vis.delete()

def add_noise_to_state(rmodel, x, scale = .01, noise_levels=[1, np.pi, np.pi, 1, 1]):
    nq = rmodel.nq  # Get the number of joint positions (19 based on your example)
    nv = rmodel.nv  # Get the number of joint velocities (18 based on your example)
    noise_levels = np.array(noise_levels) * scale
    # Copy the positions and velocities
    q_noisy = x[:nq].copy()
    
    # Generate noise for position and orientation
    base_position_noise = np.random.normal(0, noise_levels[0], 3)
    base_orientation_noise = np.random.normal(0, noise_levels[1], 3)
    joint_position_noise = np.random.normal(0, noise_levels[2], nv - 6)  # Subtract 6 for the base
    
    # Combine all position noise
    position_noise = np.hstack([base_position_noise, base_orientation_noise, joint_position_noise])
    
    # Add noise to the joint positions using Pinocchio's integrate function
    q_noisy = pin.integrate(rmodel, q_noisy, position_noise)
    
    # Generate noise for the velocities
    base_velocities_noise = np.random.normal(0, noise_levels[3], 6)  # 6 for base velocities
    joint_velocities_noise = np.random.normal(0, noise_levels[4], nv - 6)  # Subtract 6 for the base velocities
    
    # Combine all velocity noise
    velocity_noise = np.hstack([base_velocities_noise, joint_velocities_noise])
    
    # Add noise to the velocities
    v_noisy = x[nq:] + velocity_noise
    
    # Combine noisy positions and velocities
    x_noisy = np.hstack((q_noisy, v_noisy))
    return x_noisy

def change_convention_pin2mj(x):
    x_tmp = x.copy()
    quat = x[3:7]
    quat = xyzw2wxyz(quat)
    x_tmp[3:7] = quat
    return x_tmp

def change_convention_mj2pin(x):
    x_tmp = x.copy()
    quat = x[3:7]
    quat = wxyz2xyzw(quat)
    x_tmp[3:7] = quat
    return x_tmp

# Converting quaternion from mujoco to pinocchio
def wxyz2xyzw(quat):
    return np.roll(quat, -1)

# Converting quaternion from pinocchio to mujoco
def xyzw2wxyz(quat):
    return np.roll(quat, 1)

def stateMapping_mj2pin(x_mj, rmodel):
    rdata = rmodel.createData()
    nq, nv = rmodel.nq, rmodel.nv
    x_pin = change_convention_mj2pin(x_mj)
    v_mj = x_mj[-nv:]
    q_pin = x_pin[:nq]

    pin.framesForwardKinematics(rmodel, rdata, q_pin)

    pin.updateFramePlacements(rmodel, rdata)
    R = rdata.oMi[ROOT_JOINT_INDEX].rotation
    BaseLinVel_mj = v_mj[:3]
    BaseLinVel_pin = R.T @ BaseLinVel_mj  # Change the base linear velocity to body frame
    x_pin[19:22] = BaseLinVel_pin.copy()
    
    M_ = np.eye(2*nv)
    M_[nv:nv+3, nv:nv+3] = R.T
    M_inv_ = np.eye(2*nv)
    M_inv_[nv:nv+3, nv:nv+3] = R
    return np.array(x_pin), M_, M_inv_

def stateMapping_pin2mj(x_pin, rmodel):
    rdata = rmodel.createData()
    nq, nv = rmodel.nq, rmodel.nv
    q_pin, v_pin = x_pin[:nq], x_pin[-nv:]
    pin.forwardKinematics(rmodel, rdata, q_pin, v_pin)
    pin.updateFramePlacements(rmodel, rdata)

    x_mj = change_convention_pin2mj(x_pin)
    R = rdata.oMi[ROOT_JOINT_INDEX].rotation
    BaseLinVel_pin = x_pin[19:22]
    # BaseLinVel_mj = pin.getFrameVelocity(rmodel, rdata, ROOT_JOINT_INDEX, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
    BaseLinVel_mj = R @ BaseLinVel_pin # Change the base linear velocity to global frame
    x_mj[19:22] = BaseLinVel_mj.copy()

    M_ = np.eye(2*nv)
    M_[nv:nv+3, nv:nv+3] = R
    M_inv_ = np.eye(2*nv)
    M_inv_[nv:nv+3, nv:nv+3] = R.T
    return np.array(x_mj), M_, M_inv_

import matplotlib.pyplot as plt

def plot_normal_forces(forces, fids, dt, output_file="normal_forces_plot.png", fontsize=20):
    """
    Plots the normal force (z-component) vs time for each foot in separate subplots and saves it as an image.
    
    :param forces: List of forces at each timestep for all feet.
    :param fids: Foot ids (FL, FR, RL, RR).
    :param dt: Time step duration.
    :param total_time: Total simulation time.
    :param output_file: The path where the plot will be saved (e.g., "normal_forces_plot.png").
    """
    time_steps = [i * dt for i in range(len(forces))]  # Time array
    normal_forces = {fid: [] for fid in fids}  # To store normal forces (f_z) for each foot
    tangential_forces = {fid: [] for fid in fids}  # To store tangential force magnitudes (||f_x, f_y||_2)
    contact_switches = {fid: [] for fid in fids}  # To store contact switching states (ON/OFF)

    for i in range(len(forces)):
        for eff, fid in enumerate(fids):
            # Extract force components
            force_x, force_y, force_z = forces[i][eff][0], forces[i][eff][1], forces[i][eff][2]

            # Compute normal force (f_z)
            normal_forces[fid].append(force_z)

            # Compute tangential force magnitude (||f_x, f_y||_2)
            tangential_force_magnitude = np.linalg.norm([force_x, force_y])
            tangential_forces[fid].append(tangential_force_magnitude)

            # Contact is ON if any force component is non-zero, otherwise it's OFF
            if np.any(forces[i][eff] != 0):
                contact_switches[fid].append(1)  # Contact ON
            else:
                contact_switches[fid].append(0)  # Contact OFF

    # Creating subplots (2 rows, 2 columns for 4 feet)
    fig, axs = plt.subplots(4, 1, figsize=(15, 20))
    fig.suptitle('Forces vs. Time', fontsize=fontsize+3)

    # Titles and labels for each subplot
    foot_labels = ['FL', 'FR', 'RL', 'RR']

    # Plotting normal forces, tangential forces, and contact switches for each foot in separate subplots
    for idx, ax in enumerate(axs.flat):
        fid = fids[idx]
        
        # Plot normal force
        ax.plot(time_steps, normal_forces[fid], label=f'Normal Force', color='red')
        
        # Plot tangential force magnitude (||f_x, f_y||_2)
        ax.plot(time_steps, tangential_forces[fid], label=f'Tangential Force', color='green')
        
        # Set x and y labels (foot name as y label)
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        ax.set_ylabel(f'{foot_labels[idx]}', fontsize=fontsize)
        
        # Create a secondary axis for contact switching
        ax2 = ax.twinx()
        ax2.plot(time_steps, contact_switches[fid], label = f'Contact Switching', linestyle='--', color='blue', marker='o', alpha=0.7)
        
        ax2.set_ylim(-0.05, 2.05)  # Ensures the contact switching only shows 0 and 1
        ax2.set_yticks([])  # Hides y-ticks for contact switch axis
        
        # Place the combined legend in the upper right corner
        ax.legend(loc='upper left', fontsize=fontsize)
        ax2.legend(loc='upper right', fontsize=fontsize)

    plt.tight_layout()

    # Save the plot as a PDF
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()  # Close the figure to free up memory

# Example call (place this after the loop where forces are being processed)
# plot_normal_forces_and_contacts_over_time(forces, fids, dt, total_time, "normal_forces_contacts_plot.pdf")

def plot_base_poses(base_positions, base_orientations, dt, output_file, fontsize=20):
    """
    Plots the base position and orientation (RPY) over time in three subplots (x, y, z axes).
    
    :param base_positions: List of base positions at each timestep (x, y, z).
    :param base_orientations: List of base orientations at each timestep (as quaternions: x, y, z, w).
    :param dt: Time step duration.
    :param output_file: The path where the plot will be saved (e.g., "base_pose_plot.pdf").
    """
    time_steps = [i * dt for i in range(len(base_positions))]  # Time array

    # Convert all quaternions to RPY (roll, pitch, yaw)
    rpy_orientations = [quaternion_to_rpy(q) for q in base_orientations]

    # Create subplots for base positions (x, y, z) and corresponding orientations (roll, pitch, yaw)
    fig, axs = plt.subplots(3, 1, figsize=(15, 20))
    fig.suptitle('Base Pose vs. Time', fontsize=fontsize+3)

    # Base position and orientation labels
    axis_labels = ['X', 'Y', 'Z']
    rpy_labels = ['Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']
    colors = ['red', 'green', 'blue']  # Red for X, Green for Y, Blue for Z

    # Plotting base positions and orientations for x, y, z axes
    for idx in range(3):
        # Plot positions (x, y, z)
        axs[idx].plot(time_steps, [bp[idx] for bp in base_positions], label=f'Position {axis_labels[idx]}', color=colors[idx])

        # Plot RPY orientations (roll, pitch, yaw)
        axs[idx].plot(time_steps, [rpy[idx] for rpy in rpy_orientations], label=f'Orientation {rpy_labels[idx]}', color=colors[idx], linestyle='--')

        # Set axis labels and titles
        axs[idx].set_xlabel('Time (s)', fontsize = fontsize)
        axs[idx].set_ylabel(f'{axis_labels[idx]} Axis', fontsize = fontsize)
        axs[idx].grid(True)
        axs[idx].legend(loc='upper right', fontsize = fontsize)

    plt.tight_layout()

    # Save the plot as a PDF
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()  # Close the figure to free up memory

data_path = 'trajectory/trajectory_data/'
suffix = '.npz'
def save_arrays(xs, us, filename='data_compressed'):
    """
    Saves lists of numpy arrays `xs` and `us` into a compressed npz file.

    Parameters:
        xs (list of np.ndarray): List of state arrays.
        us (list of np.ndarray): List of control arrays.
        filename (str): Filename to save the compressed arrays.
    """
    full_path = data_path + filename + suffix
    np.savez_compressed(full_path, xs=np.array(xs), us=np.array(us))

def load_arrays(filename='data_compressed'):
    """
    Loads lists of numpy arrays `xs` and `us` from a compressed npz file.

    Parameters:
        filename (str): Filename to load the arrays from.

    Returns:
        tuple: A tuple containing two lists of np.ndarray, `xs` and `us`.
    """
    full_path = data_path + filename + suffix
    data = np.load(full_path, allow_pickle=True)
    xs = [np.array(x) for x in data['xs']]
    us = [np.array(u) for u in data['us']]
    return xs, us

def extend_trajectory(xs, us, scale=2):
    scale = int(scale)
    T = len(xs)-1
    xs_init = []
    us_init = []
    for t in range(T):
        x0, x1 = xs[t], xs[t+1]
        u0 = us[t]
        x = x0.copy()
        xs_init.append(x0)
        us_init.append(u0)
        for i in range(1, scale):
            x = x0 + (x1 - x0) * i / scale
            xs_init.append(x)
            us_init.append(u0)
        
    xs_init.append(xs[-1])
    
    return xs_init, us_init
# Example usage
# plot_base_poses(positions, orientations, dt, output_file="base_pose_plot.pdf")


# Example call (place this after the loop where base_positions and base_orientations are being processed)
# plot_base_poses(base_positions, base_orientations, dt, total_time, "base_pose_plot.pdf")

from scipy.spatial.transform import Rotation as R

def quaternion_to_rpy(quaternion):
    """
    Converts a quaternion (x, y, z, w) to roll, pitch, yaw (RPY) angles in radians.
    
    :param quaternion: A list or tuple of the quaternion components (x, y, z, w).
    :return: A tuple of RPY angles (roll, pitch, yaw) in radians.
    """
    # Convert quaternion (x, y, z, w) to Roll, Pitch, Yaw (RPY) in radians
    rotation = R.from_quat(quaternion)  # Quaternion order should be (x, y, z, w)
    rpy_angles = rotation.as_euler('xyz')  # Get RPY angles in radians
    return rpy_angles

# Example usage:
# quaternion = [x, y, z, w]
# rpy = quaternion_to_rpy(quaternion)
# print(f"Roll: {rpy[0]}, Pitch: {rpy[1]}, Yaw: {rpy[2]}")

def plot_sliding(velocities, forces, fids, dt, output_file, fontsize=20):
    """
    Plots the extent of sliding (||V_tangential||_2 * ||F_tangential||_2) for each foot over time in separate subplots.
    
    :param velocities: List of velocities at each timestep for all feet.
    :param forces: List of forces at each timestep for all feet.
    :param fids: Foot ids (FL, FR, RL, RR).
    :param dt: Time step duration.
    :param total_time: Total simulation time.
    :param output_file: The path where the plot will be saved (e.g., "sliding_plot.pdf").
    """
    time_steps = [i * dt for i in range(len(forces))]  # Time array
    extent_of_sliding = {fid: [] for fid in fids}  # To store extent of sliding (||V_tangential||_2 * ||F_tangential||_2)

    # Process forces and velocities to compute extent of sliding for each foot
    for i in range(len(forces)):
        for eff, fid in enumerate(fids):
            # Extract force components
            force_x, force_y = forces[i][eff][0], forces[i][eff][1]

            # Extract velocity components
            vel_x, vel_y = velocities[i][eff][0], velocities[i][eff][1]

            # Compute tangential force magnitude (||f_x, f_y||_2)
            tangential_force_magnitude = np.linalg.norm([force_x, force_y])

            # Compute tangential velocity magnitude (||v_x, v_y||_2)
            tangential_velocity_magnitude = np.linalg.norm([vel_x, vel_y])

            # Compute extent of sliding (||V_tangential||_2 * ||F_tangential||_2)
            sliding_value = tangential_velocity_magnitude * tangential_force_magnitude
            extent_of_sliding[fid].append(sliding_value)

    # Create subplots for sliding (1x4 layout)
    fig, axs = plt.subplots(4, 1, figsize=(15, 20))
    fig.suptitle('Extent of Sliding vs. Time', fontsize=fontsize+3)

    # Titles and labels for each subplot
    foot_labels = ['FL', 'FR', 'RL', 'RR']

    # Plotting extent of sliding for each foot
    for idx, ax in enumerate(axs.flat):
        fid = fids[idx]

        # Plot extent of sliding
        ax.plot(time_steps, extent_of_sliding[fid], label=f'||V_t|| * ||F_t||: {foot_labels[idx]}', color='red')
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        ax.set_ylabel(f'{foot_labels[idx]}', fontsize=fontsize)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=fontsize)

    plt.tight_layout()

    # Save the plot as a PDF
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()  # Close the figure to free up memory

# Example call (place this after the loop where velocities and forces are being processed)
# plot_sliding_over_time(velocities, forces, fids, dt, total_time, "sliding_plot.pdf")

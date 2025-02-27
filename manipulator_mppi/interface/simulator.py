import os

import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
import numpy as np
import tqdm
import mujoco.viewer 
from PIL import Image


class Simulator:
    """
    A class representing a simulator for controlling and estimating the state of a system (Trifinger).

    Attributes:
        filter (object): The filter used for state estimation.
        agent (object): The agent used for control.
        model_path (str): The path to the XML model file.
        T (int): The number of time steps.
        dt (float): The time step size.
        viewer (bool): Flag indicating whether to enable the viewer.
        gravity (bool): Flag indicating whether to enable gravity.
        model (object): The MuJoCo model.
        data (object): The MuJoCo data.
        qpos (ndarray): The position trajectory.
        qvel (ndarray): The velocity trajectory.
        finite_diff_qvel (ndarray): The finite difference of velocity.
        ctrl (ndarray): The control trajectory.
        sensordata (ndarray): The sensor data trajectory.
        noisy_sensordata (ndarray): The noisy sensor data trajectory.
        time (ndarray): The time trajectory.
        cost (ndarray): The cost trajectory (if agent provides eval_best_trajectory).
        viewer (object): The MuJoCo viewer object.
        update_ratio (int): The ratio of simulation steps to control updates.
        save_dir (str): Where to save frames if requested.
        save_frames (bool): Whether to save frames or not.
    """

    def __init__(self,
                 filter=None,
                 agent=None,
                 model_path=None,
                 T=200,
                 dt=0.01,
                 viewer=True,
                 gravity=True,
                 timeconst=0.02,
                 dampingratio=1.0,
                 ctrl_rate=100,
                 save_dir="./frames",
                 save_frames=False):

        # If no model_path is specified, point to trifinger XML
        if model_path is None:
            print('No model path specified.')
            exit()

        # filter
        self.filter = filter
        self.agent = agent
        self.ctrl_rate = ctrl_rate
        self.update_ratio = max(1, int(1 / (dt * ctrl_rate)))  # how often to call agent.update
        self.interpolate_cam = False

        # model
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = dt
        # override contact settings
        # self.model.opt.enableflags = 1
        # self.model.opt.o_solref = np.array([timeconst, dampingratio])

        # data
        self.data = mujoco.MjData(self.model)
        self.T = T

        # save frames
        self.save_frames = save_frames
        self.save_dir = save_dir
        if self.save_frames and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # rollout
        mujoco.mj_resetData(self.model, self.data)
        # Use keyframe[0] ("home") as initial condition
        # Make sure your trifinger XML has exactly one key definition at index 0
        # or adjust indices as needed.

        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qvel[:] = self.model.key_qvel[0]
        self.data.ctrl[:] = self.model.key_ctrl[0]

        # turn off gravity if requested
        if not gravity:
            self.model.opt.gravity[:] = 0
            if self.filter is not None:
                self.filter.model.opt.gravity[:] = 0

        # viewer
        if viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, hide_menus=True)
        else:
            self.viewer = None

        # Trajectories to log
        self.qpos = np.zeros((self.model.nq, self.T))
        self.qvel = np.zeros((self.model.nv, self.T))
        self.finite_diff_qvel = np.zeros((self.model.nv, self.T - 1))
        self.ctrl = np.zeros((self.model.nu, self.T))
        self.sensordata = np.zeros((self.model.nsensordata, self.T))
        self.noisy_sensordata = np.zeros((self.model.nsensordata, self.T))
        self.time = np.zeros(self.T)
        self.cost = np.zeros((1, self.T))  # depends on agent
        if agent is not None:
            self.goal_ori = self.agent.cube_state_ref_1d[3:]
            self.goal_pos = self.agent.cube_state_ref_1d[:3]
        # mujoco.mju_quat2Mat(self.goal_ori, self.agent.cube_state_ref_1d[3:])

    def reset(self, agent = None):
        """
        Reset the simulation to the initial state.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qvel[:] = self.model.key_qvel[0]
        self.data.ctrl[:] = self.model.key_ctrl[0]

        mujoco.mj_forward(self.model, self.data)
        self.agent = agent

        if self.agent is not None:
            self.goal_ori = self.agent.cube_state_ref_1d[3:]
            self.goal_pos = self.agent.cube_state_ref_1d[:3]
            
        if self.viewer is not None:
             # Visualize the goal pose
            self.data.mocap_pos[0, :] = self.goal_pos
            self.data.mocap_quat[0, :] = self.goal_ori        
            self.viewer.sync()

    def get_sensor(self):
        return self.data.sensordata

    def step(self, ctrl=None):

        """
        Step the simulation by one timestep with the specified control.
        """
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

        return self.data.qpos, self.data.qvel

    def store_trajectory(self, t):
        """
        Store the current simulation data into trajectory buffers at time index t.
        """
        self.qpos[:, t] = self.data.qpos
        self.qvel[:, t] = self.data.qvel
        self.ctrl[:, t] = self.data.ctrl
        self.sensordata[:, t] = self.data.sensordata
        self.time[t] = self.data.time

        # If your agent has a cost function
        if self.agent is not None and hasattr(self.agent, "eval_best_trajectory"):
            self.cost[0, t] = self.agent.eval_best_trajectory()

    def state_difference(self, pos1, pos2):
        """
        Computes the finite difference between two states in generalized coordinates.
        """
        vel = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, vel, self.model.opt.timestep, pos1, pos2)
        return vel

    def capture_frame(self, frame_index):
        """
        Capture the current frame from the viewer and save it as PNG.
        """
        if self.viewer is None:
            return

        width, height = self.viewer.viewport.width, self.viewer.viewport.height
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(frame, None, self.viewer.viewport, self.viewer.ctx)
        filename = os.path.join(self.save_dir, f"frame_{frame_index}.png")
        image = Image.fromarray(np.flipud(frame))
        image.save(filename)

    def run(self):
        """
        Main rollout loop: run T-1 steps of simulation, optionally calling agent to get actions.
        """
        tqdm_range = tqdm.tqdm(range(self.T - 1))
        for t in tqdm_range:
            self.t = t
            # forward first to ensure sensor reading is up to date
            mujoco.mj_forward(self.model, self.data)

            # store logs
            self.store_trajectory(t)

            # Possibly get action from agent
            if self.agent is not None:
                # Only update the action every update_ratio steps
                if t % self.update_ratio == 0:
                    # The agent might want obs = [qpos; qvel] or something similar
                    observation = np.concatenate([self.data.qpos, self.data.qvel], axis=0)

                    action = self.agent.update(observation)

                    self.data.ctrl[:] = action

             # step
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)


            # Render
            if self.viewer is not None and self.viewer.is_running():
                
                # Visualize the goal pose
                self.data.mocap_pos[0, :] = self.goal_pos
                self.data.mocap_quat[0, :] = self.goal_ori                   

                # self.viewer.render()
                self.viewer.sync()
                if self.save_frames:
                    self.capture_frame(t)
            else:
                break

        # store the last step
        self.store_trajectory(self.T - 1)

        if self.viewer is not None:
            self.viewer.close()



    def plot_trajectory(self):
        """
        Example plotting of states, controls, etc.
        Customize for trifinger data.
        """
        # -- Example: position (qpos) over time
        plt.figure()
        for i in range(self.model.nq):
            plt.plot(self.time, self.qpos[i, :], label=f"qpos[{i}]")
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Position (rad)")
        plt.legend()

        # -- Example: velocity (qvel) over time
        plt.figure()
        for i in range(self.model.nv):
            plt.plot(self.time, self.qvel[i, :], label=f"qvel[{i}]")
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Velocity (rad/s)")
        plt.legend()

        # -- Example: controls
        plt.figure()
        for i in range(self.model.nu):
            plt.plot(self.time, self.ctrl[i, :], label=f"ctrl[{i}]")
        plt.xlabel("Time (s)")
        plt.ylabel("Control")
        plt.legend()

        # -- Example: sensor data (9 = tip0, tip120, tip240 in x,y,z)
        plt.figure()
        for i in range(self.model.nsensordata):
            plt.plot(self.time, self.sensordata[i, :], label=f"sensor[{i}]")
        plt.xlabel("Time (s)")
        plt.ylabel("Sensor reading")
        plt.legend()

        plt.figure()

        plt.title("Cube Positions")
            # Cube (indices 9..11)
        plt.plot(self.time, self.sensordata[9, :], label="cube_x")
        plt.plot(self.time, self.sensordata[10, :], label="cube_y")
        plt.plot(self.time, self.sensordata[11, :], label="cube_z")
        plt.axhline(y=0.2, color='r', linestyle=':', label='desired z position')

        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()

        plt.show()


if __name__ == "__main__":
    # Example usage
    model_path = os.path.join(
        os.path.dirname(__file__),
        "../models/trifinger/trifinger_scene.xml"
    )

    simulator = Simulator(
        filter=None,
        T=300,
        dt=0.002,
        viewer=True,
        gravity=True,
        model_path=model_path
    )
    simulator.run()
    simulator.plot_trajectory()

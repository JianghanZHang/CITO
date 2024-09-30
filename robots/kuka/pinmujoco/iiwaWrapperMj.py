"""iiwawrapper
Iiwa pybullet interface using pinocchio's convention.
License: BSD 3-Clause License
Copyright (C) 2018-2019, New York University , Max Planck Gesellschaft
Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""
# import pybullet 
# from mim_robots.pybullet.wrapper import PinBulletWrapper
import pinocchio as pin
import numpy as np
from mim_robots.robot_loader import load_pinocchio_wrapper
import mujoco
from mim_robots.mujoco.wrapperMJ import PinMujocoWrapper
    # mim_robots.pybullet.wrapper import PinBulletWrapper

import time
import mujoco.viewer
dt = 1e-3

class IiwaRobot(PinMujocoWrapper):
    '''
    Pinocchio-Mujoco wrapper class for the KUKA LWR iiwa 
    '''
    def __init__(self, robotinfo, locked_joints_names=None, qref=np.zeros(7), pos=None, orn=None): 

        self.urdf_path = robotinfo.urdf_path
        mjmodel = mujoco.MjModel.from_xml_path(robotinfo.xml_path)
        mjmodel.opt.timestep = dt
        mjdata = mujoco.MjData(mjmodel)

        # create the passive viewer
        # self.viewer = mujoco.viewer.launch_passive(mjmodel, mjdata)
        # time.sleep(5)

        # Create the robot wrapper in pinocchio.
        robot_full = load_pinocchio_wrapper(robotinfo.name)
        joint_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        base_link_name = "iiwa_base"
        endeff_names = ["EE"]
        self.end_eff_ids = []
        self.end_eff_ids.append(robot_full.model.getFrameId('contact'))
        self.nb_ee = len(self.end_eff_ids)

        use_fix_base = True
        # Creates the wrapper by calling the super.__init__.          
        super().__init__(mjmodel, 
                         mjdata,
                         robot_full,
                         joint_names,
                         endeff_names,
                         use_fix_base)
        self.nb_dof = self.nv
        
    def forward_robot(self, q=None, dq=None):
        if q is None:
            q, dq = self.get_state()
        elif dq is None:
            raise ValueError("Need to provide q and dq or non of them.")

        self.pin_robot.forwardKinematics(q, dq)
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)

    def start_recording(self, file_name):
        raise NotImplementedError("Not implemented for mujoco yet.")

    def stop_recording(self):
        raise NotImplementedError("Not implemented for mujoco yet.")

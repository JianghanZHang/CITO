# import mim_robots.robot_loader
# from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config, Solo12RobotWithoutPybullet
# import mim_robots
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../robots/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'python/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'robots/')))
import mujoco
import pinocchio as pin 

import numpy as np
import hppfcl as fcl
ROOT_JOINT_INDEX = 1
GO2_FOOT_RADIUS = 0.000

go2_init_conf0 = np.array([0.0000, 0.0000, 0.2750, 
                0.0000, 0.0000, 0.0000, 1.0000, 
                0.0000, 0.9000, -1.8000, 
                0.0000, 0.9000, -1.8000, 
                0.0000, 0.9000, -1.8000, 
                0.0000, 0.9000, -1.8000])

go2_v0 = np.array([0.0000, 0.0000, 0.0000, 
                0.0000, 0.0000, 0.0000, 
                0.0000, 0.0000, 0.0000, 
                0.0000, 0.0000, 0.0000, 
                0.0000, 0.0000, 0.0000, 
                0.0000, 0.0000, 0.0000])
                

# def create_solo12_env_free_force():

#     solo12 = Solo12RobotWithoutPybullet()
#     solo12_robot = solo12.pin_robot
#     solo_rmodel, solo_gmodel, solo_vmodel  = solo12_robot.model, solo12_robot.collision_model, solo12_robot.visual_model
#     env = {
#         "nq" : 19,
#         "nv" : 18,
#         "rmodel" : solo_rmodel,
#         "gmodel" : solo_gmodel,
#         "vmodel" : solo_vmodel,
#         "nu" : 24,
#         "njoints" : 12,
#         "ncontacts" : 4,
#         "contactFnames" : ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"],
#         "contactFids" : [],
#         "q0" : Solo12Config.initial_configuration,
#         "v0" : Solo12Config.initial_velocity,
#     }

#     env["nu"] = env["njoints"] + 3 * env["ncontacts"]

#     for idx, frameName in enumerate(env["contactFnames"]):
#         env["contactFids"].append(env["rmodel"].getFrameId(frameName))
    
#     return env

# def create_solo12_env_force_MJ():
#     mj_model = mim_robots.robot_loader.load_mujoco_model("solo12")
#     mj_data = mujoco.MjData(mj_model)
#     env = {
#         "nq" : 19,
#         "nv" : 18,
#         "mj_model" : mj_model,
#         "nu" : 12,
#         "njoints" : 12,
#         "ncontacts" : 4,
#         "contactFnames" : ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"],
#         "contactFids" : [],
#         "q0" : Solo12Config.initial_configuration,
#         "v0" : Solo12Config.initial_velocity,
#     }

#     env["nf"] = 3 * env["ncontacts"]

#     for idx, frameName in enumerate(env["contactFnames"]):
#         env["contactFids"].append(mj_model.body(name=frameName).id)

#     return env

def create_go2_env():
    urdf_path = "robots/go2_robot_sdk/urdf/go2.urdf"
    package_dirs = ["/home/jianghan/Devel/workspace_autogait/src/auto_gait_generation/robots"]

    print(f"URDF Path: {urdf_path}")
    print(f"Package Dirs: {package_dirs}")
    rmodel, gmodel, vmodel = pin.buildModelsFromUrdf(urdf_path, package_dirs, root_joint= pin.JointModelFreeFlyer(), verbose=True)

    env = {
        "nq" : 19,
        "nv" : 18,
        "rmodel" : rmodel,
        "gmodel" : gmodel,
        "vmodel" : vmodel,
        "nu" : 12,
        "njoints" : 12,
        "ncontacts" : 4,
        "contactFnames" : ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        "contactFids" : [],
        "jointFnames" : ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                         "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                         "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                         "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        "jointFids" : [],
        "q0" : list(go2_init_conf0),
        "v0" : list(go2_v0),
    }
    
    env["nf"] = 3 * env["ncontacts"]

    for idx, frameName in enumerate(env["contactFnames"]):
        env["contactFids"].append(env["rmodel"].getFrameId(frameName))

    for idx, frameName in enumerate(env["jointFnames"]):
        env["jointFids"].append(env["rmodel"].getFrameId(frameName))
        
    return env


def create_go2_env_force_MJ():
    xml_path = "robots/unitree_go2/scene_foot_collision.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    env = {
        "nq" : 19,
        "nv" : 18,
        "mj_model" : mj_model,
        "nu" : 12,
        "njoints" : 12,
        "ncontacts" : 4,
        "contactFnames" : ["FL", "FR", "RL", "RR"],
        "contactFids" : [],
        "q0" : list(go2_init_conf0),
        "v0" : list(go2_v0),
    }

    env["nf"] = 3 * env["ncontacts"]

    for idx, frameName in enumerate(env["contactFnames"]):
        env["contactFids"].append(mj_model.geom(name=frameName).id)

    return env

def create_cube(name="cube", color=[1.,0,0.,1.]):
    parent_id = 0
    mass = 0.5
    cube_length = 0.05
    cube_length_collision = 0.05

    rmodel = pin.Model()
    rmodel.name = name
    gmodel = pin.GeometryModel()

    ## Joints
    joint_name = name + "_floating_joint"
    joint_placement = pin.SE3.Identity()
    base_id = rmodel.addJoint(parent_id, pin.JointModelFreeFlyer(), joint_placement, joint_name)
    rmodel.addJointFrame(base_id)
    
    cube_inertia = pin.Inertia.FromBox(mass, cube_length, cube_length, cube_length)
    cube_placement = pin.SE3.Identity()
    rmodel.appendBodyToJoint(base_id, cube_inertia, cube_placement)

    geom_name = name
    shape = fcl.Box(cube_length_collision, cube_length_collision, cube_length_collision)
    shape_placement = cube_placement.copy()

    geom_obj = pin.GeometryObject(geom_name, base_id, shape, shape_placement)
    geom_obj.meshColor = np.array(color)
    gmodel.addGeometryObject(geom_obj)

    return rmodel, gmodel, mass

def create_cube_pusher_env_MJ():
    xml_path = "robots/kuka/xml/iiwa_pusher_v2.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    env = {
        "nq" : 14,
        "nv" : 13,
        "mj_model" : mj_model,
        "nu" : 7,
        "njoints" : 7,
        "ncontacts" : 1,
        "contactFnames" : ["pusher_tip"],
        "contactFids" : [],
        "q0" : list(go2_init_conf0),
        "v0" : list(go2_v0),
    }

    env["nf"] = 3 * env["ncontacts"]

    for idx, frameName in enumerate(env["contactFnames"]):
        env["contactFids"].append(mj_model.geom(name=frameName).id)

    return env

def create_cube_pusher_env():
    urdf_path = "robots/kuka/urdf/iiwa_pusher_v2.urdf"
    package_dirs = ["/home/jianghan/Devel/workspace_autogait/src/auto_gait_generation/robots/kuka"]

    print(f"URDF Path: {urdf_path}")
    print(f"Package Dirs: {package_dirs}")
    pusher_rmodel, pusher_gmodel, pusher_vmodel = pin.buildModelsFromUrdf(urdf_path, package_dirs, verbose=True)
    
    cube_rmodel, cube_gmodel, cube_mass = create_cube("cube")

    rmodel, gmodel = pin.appendModel(
        pusher_rmodel, 
        cube_rmodel, 
        pusher_gmodel,
        cube_gmodel,
        0,
        pin.SE3.Identity()
    )

    env = {
        "nq" : 14,
        "nv" : 13,
        "rmodel" : rmodel,
        "gmodel" : gmodel,
        "nu" : 7,
        "njoints" : 7,
        "cubeMass" : cube_mass,
        "ncontacts" : 1,
        "contactFnames" : ["pusher_tip"],
        "contactFids" : [],
        "q0" : list(go2_init_conf0),
        "v0" : list(go2_v0),
    }

    env["nf"] = 3 * env["ncontacts"]

    for idx, frameName in enumerate(env["contactFnames"]):
        env["contactFids"].append(env["rmodel"].getFrameId(frameName))

    return env
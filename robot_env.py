import mim_robots.robot_loader
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config, Solo12RobotWithoutPybullet
import mujoco
import mim_robots
import pinocchio as pin 
from robot_descriptions.loaders.pinocchio import load_robot_description
from robot_descriptions import go2_description
import numpy as np

go2_init_conf1 = np.array([0.0000, 0.0000, 0.2700 + 0.1225, 
                0.0000, 0.0000, 0.0000, 1.0000, 
                0.0000, 0.4000, -0.8000, 
                0.0000, 0.4000, -0.8000, 
                0.0000, 0.4000, -0.8000, 
                0.0000, 0.4000, -0.8000])
                
go2_init_conf0 = np.array([0.0000, 0.0000, 0.2700, 
                0.0000, 0.0000, 0.0000, 1.0000, 
                0.0000, 0.9000, -1.8000, 
                0.0000, 0.9000, -1.8000, 
                0.0000, 0.9000, -1.8000, 
                0.0000, 0.9000, -1.8000])
                

def create_solo12_env_free_force():

    solo12 = Solo12RobotWithoutPybullet()
    solo12_robot = solo12.pin_robot
    solo_rmodel, solo_gmodel, solo_vmodel  = solo12_robot.model, solo12_robot.collision_model, solo12_robot.visual_model
    env = {
        "nq" : 19,
        "nv" : 18,
        "rmodel" : solo_rmodel,
        "gmodel" : solo_gmodel,
        "vmodel" : solo_vmodel,
        "nu" : 24,
        "njoints" : 12,
        "ncontacts" : 4,
        "contactFnames" : ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"],
        "contactFids" : [],
        "q0" : Solo12Config.initial_configuration,
        "v0" : Solo12Config.initial_velocity,
    }

    env["nu"] = env["njoints"] + 3 * env["ncontacts"]

    for idx, frameName in enumerate(env["contactFnames"]):
        env["contactFids"].append(env["rmodel"].getFrameId(frameName))
        

    return env

def create_solo12_env_force_MJ():
    mj_model = mim_robots.robot_loader.load_mujoco_model("solo12")
    mj_data = mujoco.MjData(mj_model)
    env = {
        "nq" : 19,
        "nv" : 18,
        "mj_model" : mj_model,
        "nu" : 12,
        "njoints" : 12,
        "ncontacts" : 4,
        "contactFnames" : ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"],
        "contactFids" : [],
        "q0" : Solo12Config.initial_configuration,
        "v0" : Solo12Config.initial_velocity,
    }

    env["nf"] = 3 * env["ncontacts"]

    for idx, frameName in enumerate(env["contactFnames"]):
        env["contactFids"].append(mj_model.body(name=frameName).id)

    return env

def create_go2_env():
    import os
    urdf_path = "robots/go2_robot_sdk/urdf/go2.urdf"
    # package_dirs = ["robots/go2_robot_sdk/dae"]
    # package_dirs = [os.path.join(os.getcwd(), "robots/go2_robot_sdk")]
    package_dirs = ["/home/jianghan/Devel/workspace_autogait/src/auto_gait_generation/robots"]
    
    xml_path = "robots/unitree_go2/scene_foot_collision.xml"
    # robot_wrapper = load_robot_description("go2_description")
    # gmodel, vmodel = robot_wrapper.collision_model, robot_wrapper.visual_model
    # rmodel = pin.buildModelFromUrdf(go2_description.URDF_PATH, pin.JointModelFreeFlyer())


    print(f"URDF Path: {urdf_path}")
    print(f"Package Dirs: {package_dirs}")
    rmodel, gmodel, vmodel = pin.buildModelsFromUrdf(urdf_path, package_dirs, pin.JointModelFreeFlyer(), True)
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    q0_stand = np.array(mj_model.key_qpos).reshape(19, )
    q0_stand[3] = 0.0
    q0_stand[4] = 0.0
    q0_stand[5] = 0.0
    q0_stand[6] = 1.0
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
        "q0" : list(go2_init_conf0),
        "v0" : list(mj_model.key_qvel.reshape(18, )),
    }
    
    env["nf"] = 3 * env["ncontacts"]

    for idx, frameName in enumerate(env["contactFnames"]):
        env["contactFids"].append(env["rmodel"].getFrameId(frameName))

    return env


def create_go2_env_force_MJ():
    xml_path = "robots/unitree_go2/scene_foot_collision.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    q0_stand = np.array(mj_model.key_qpos).reshape(19, )
    q0_stand[3] = 0.0
    q0_stand[4] = 0.0
    q0_stand[5] = 0.0
    q0_stand[6] = 1.0
    env = {
        "nq" : 19,
        "nv" : 18,
        "mj_model" : mj_model,
        "mj_data" : mj_data,
        "nu" : 12,
        "njoints" : 12,
        "ncontacts" : 4,
        "contactFnames" : ["FL", "FR", "RL", "RR"],
        "contactFids" : [],
        "q0" : list(go2_init_conf0),
        "v0" : list(mj_model.key_qvel.reshape(18, )),
    }

    env["nf"] = 3 * env["ncontacts"]

    for idx, frameName in enumerate(env["contactFnames"]):
        env["contactFids"].append(mj_model.geom(name=frameName).id)

    return env
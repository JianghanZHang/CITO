import mim_robots.robot_loader
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config, Solo12RobotWithoutPybullet
import mujoco
import mim_robots
import pinocchio as pin 
from robot_descriptions.loaders.pinocchio import load_robot_description
from robot_descriptions import go2_description

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
    xml_path = "robots/unitree_go2/scene_foot_collision.xml"
    rmodel, gmodel, vmodel = pin.buildModelsFromUrdf(filename = go2_description.URDF_PATH, package_dirs = [go2_description.PACKAGE_PATH], 
                                                     root_joint=pin.JointModelFreeFlyer(),
                                                     verbose = True)
    # rmodel = pin.buildModelFromUrdf(go2_description.URDF_PATH, pin.JointModelFreeFlyer())
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    env = {
        "nq" : 19,
        "nv" : 18,
        "rmodel" : rmodel,
        "nu" : 12,
        "njoints" : 12,
        "ncontacts" : 4,
        "contactFnames" : ["FL", "FR", "RL", "RR"],
        "contactFids" : [],
        "q0" : mj_model.key_qpos,
        "v0" : mj_model.key_qvel,
    }
    
    env["nf"] = 3 * env["ncontacts"]

    for idx, frameName in enumerate(env["contactFnames"]):
        env["contactFids"].append(env["rmodel"].getFrameId(frameName))

    return env

def create_go2_env_force_MJ():
    xml_path = "robots/unitree_go2/scene_foot_collision.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
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
        "q0" : mj_model.key_qpos,
        "v0" : mj_model.key_qvel,
    }

    env["nf"] = 3 * env["ncontacts"]

    for idx, frameName in enumerate(env["contactFnames"]):
        env["contactFids"].append(mj_model.geom(name=frameName).id)

    return env
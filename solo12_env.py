from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config, Solo12RobotWithoutPybullet

import pinocchio as pin 

base_path = "/opt/openrobots/share/example-robot-data/robots"
urdf_path = base_path + "/solo_description/robots/solo12.urdf"
mesh_path = base_path + "/solo_description/meshes"

solo12 = Solo12RobotWithoutPybullet()

solo12_robot = solo12.pin_robot

def create_solo12_env():
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


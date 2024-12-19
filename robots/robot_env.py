# import mim_robots.robot_loader
# from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config, Solo12RobotWithoutPybullet
# import mim_robots
import sys
import os
import re
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
                

def create_trifinger_env():
    package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    urdf_path = os.path.join(package_dir, "trifinger/trifinger.urdf")
    xml_path = os.path.join(package_dir, "trifinger/trifinger.xml")
    
    package_dirs = [package_dir]

    print(f"URDF Path: {urdf_path}")
    print(f"Package Dirs: {package_dirs}")

    pin_model, gmodel, vmodel = pin.buildModelsFromUrdf(urdf_path, package_dirs, verbose=True)
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # import pdb; pdb.set_trace()
    xml_urdf_sanity_check(mj_model, pin_model, with_cube=False)

    return pin_model, mj_model

def create_go2_env():
    urdf_path = "robots/go2_robot_sdk/urdf/go2.urdf"
    package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    package_dirs = [package_dir]

    print(f"URDF Path: {urdf_path}")
    print(f"Package Dirs: {package_dirs}")
    rmodel, gmodel, vmodel = pin.buildModelsFromUrdf(urdf_path, package_dirs, root_joint=pin.JointModelFreeFlyer(), verbose=True)

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

# def xml_urdf_sanity_check(mj_model, pin_model, with_cube=True):
#     # Create data
#     pin_data = pin.Data(pin_model)
#     mj_data = mujoco.MjData(mj_model)

#     print("Pinocchio nq:", pin_model.nq)
#     print("MuJoCo nq:", mj_model.nq)

#     qpos = np.zeros(pin_model.nq)
#     mj_data.qpos[:] = qpos
#     mujoco.mj_forward(mj_model, mj_data)

#     pin_q = qpos.copy()

#     pin.forwardKinematics(pin_model, pin_data, pin_q)
#     pin.updateFramePlacements(pin_model, pin_data)
#     if with_cube:
#         cube_body_id = mj_model.body(name="cube").id
#         mj_cube_pos = mj_data.xpos[cube_body_id]
#         mj_cube_mat = mj_data.xmat[cube_body_id].reshape(3,3)

#         # In Pinocchio, find cube frame id:
#         cube_frame_id = pin_model.getFrameId("cube_floating_joint")  # from create_cube() function

#         pin_cube_pos = pin_data.oMf[cube_frame_id].translation
#         pin_cube_mat = pin_data.oMf[cube_frame_id].rotation

#         print("\nCube body comparison:")
#         print("MuJoCo cube position:", mj_cube_pos)
#         print("Pinocchio cube position:", pin_cube_pos)
#         print("MuJoCo cube rotation:\n", mj_cube_mat)
#         print("Pinocchio cube rotation:\n", pin_cube_mat)

def xml_urdf_sanity_check(mj_model, pin_model, with_cube=True):
    # Create data
    pin_data = pin.Data(pin_model)
    mj_data = mujoco.MjData(mj_model)

    print("Pinocchio nq:", pin_model.nq)
    print("MuJoCo nq:", mj_model.nq)

    qpos = np.zeros(pin_model.nq)
    mj_data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, mj_data)

    pin_q = qpos.copy()

    pin.forwardKinematics(pin_model, pin_data, pin_q)
    pin.updateFramePlacements(pin_model, pin_data)
    
    # Check the three base links: finger0_base_link, finger1_base_link, finger2_base_link
    # base_links = ["finger0_base_link", "finger1_base_link", "finger2_base_link"]
    # print("\nBase links position comparison:")
    # for bl in base_links:
    #     # Get MuJoCo body id and position
    #     mj_body_id = mj_model.body(name=bl).id
    #     mj_body_pos = mj_data.xpos[mj_body_id]

    #     # Get Pinocchio frame id and position
    #     pin_frame_id = pin_model.getFrameId(bl)
    #     pin_body_pos = pin_data.oMf[pin_frame_id].translation

    #     print(f"{bl}:")
    #     print("  MuJoCo position:", mj_body_pos)
    #     print("  URDF/Pinocchio position:", pin_body_pos)

    upper_links = ["finger_upper_link_0", "finger_upper_link_120", "finger_upper_link_240"]
    print("\nUpper links position comparison:")
    for ul in upper_links:
        # Get MuJoCo body id and position
        mj_body_id = mj_model.body(name=ul).id
        mj_body_pos = mj_data.xpos[mj_body_id]

        # Get Pinocchio frame id and position
        pin_frame_id = pin_model.getFrameId(ul)
        pin_body_pos = pin_data.oMf[pin_frame_id].translation

        print(f"{ul}:")
        print("  MuJoCo position:", mj_body_pos)
        print("  URDF/Pinocchio position:", pin_body_pos)

    lower_links = ["finger_lower_link_0", "finger_lower_link_120", "finger_lower_link_240"]
    print("\nLower links position comparison:")
    for ll in lower_links:
        # Get MuJoCo body id and position
        mj_body_id = mj_model.body(name=ll).id
        mj_body_pos = mj_data.xpos[mj_body_id]

        # Get Pinocchio frame id and position
        pin_frame_id = pin_model.getFrameId(ll)
        pin_body_pos = pin_data.oMf[pin_frame_id].translation

        print(f"{ll}:")
        print("  MuJoCo position:", mj_body_pos)
        print("  URDF/Pinocchio position:", pin_body_pos)

    tips_xml = ["finger_tip_0", "finger_tip_120", "finger_tip_240"]
    tips_urdf = ["finger_tip_link_0", "finger_tip_link_120", "finger_tip_link_240"]
    print("\nTip links position comparison:")
    for idx in range(len(tips_xml)):
        # Get MuJoCo body id and position
        mj_body_id = mj_model.site(name=tips_xml[idx]).id
        mj_body_pos = mj_data.site_xpos[mj_body_id]

        # Get Pinocchio frame id and position
        pin_frame_id = pin_model.getFrameId(tips_urdf[idx])
        pin_body_pos = pin_data.oMf[pin_frame_id].translation

        print(f"{tips_xml[idx]}:")
        print("  MuJoCo position:", mj_body_pos)
        print("  URDF/Pinocchio position:", pin_body_pos)


    # If with_cube is True, also check cube positions/orientations
    if with_cube:
        cube_body_id = mj_model.body(name="cube").id
        mj_cube_pos = mj_data.xpos[cube_body_id]
        mj_cube_mat = mj_data.xmat[cube_body_id].reshape(3,3)

        # In Pinocchio, find cube frame id:
        cube_frame_id = pin_model.getFrameId("cube_floating_joint")  # from create_cube() function

        pin_cube_pos = pin_data.oMf[cube_frame_id].translation
        pin_cube_mat = pin_data.oMf[cube_frame_id].rotation

        print("\nCube body comparison:")
        print("MuJoCo cube position:", mj_cube_pos)
        print("Pinocchio cube position:", pin_cube_pos)
        print("MuJoCo cube rotation:\n", mj_cube_mat)
        print("Pinocchio cube rotation:\n", pin_cube_mat)



def create_cube(name="cube", color=[1.,0,0.,1.]):
    parent_id = 0
    mass = 0.05
    cube_length = 0.025
    cube_length_collision = 0.05

    rmodel = pin.Model()
    rmodel.name = name
    gmodel = pin.GeometryModel()

    # Free-flyer joint for the cube
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

def create_trifinger_cube_env():
    package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    package_dir_trifinger = os.path.join(package_dir, "trifinger")
    urdf_path = os.path.join(package_dir, "trifinger/trifinger_scene.urdf")
    xml_path = os.path.join(package_dir, "trifinger/trifinger_scene.xml")
    package_dirs = [package_dir_trifinger]
    # import pdb; pdb.set_trace()

    # Load Pinocchio model for trifinger
    finger_rmodel, finger_gmodel,finger_vmodel = pin.buildModelsFromUrdf(
        urdf_path, package_dirs, verbose=True
    )

    # Create cube models
    cube_rmodel, cube_gmodel, cube_mass = create_cube("cube", color=[0., 1., 0., 1.])

    # Append cube to trifinger model
    rmodel_combined, gmodel_combined = pin.appendModel(
        finger_rmodel, cube_rmodel, finger_gmodel, cube_gmodel, 0, pin.SE3.Identity()
    )

    rmodel = rmodel_combined
    gmodel = gmodel_combined
    

    # import pdb; pdb.set_trace()
    q0 = []
    v0 = []
    # Setup initial state of the cube
    q0[:cube_rmodel.nq] = np.zeros(cube_rmodel.nq)
    v0[:cube_rmodel.nq] = np.zeros(cube_rmodel.nq)

    # Setup initial state of the Trifinger
    q0[-finger_rmodel.nq:] = np.zeros(finger_rmodel.nq)
    v0[-finger_rmodel.nq:] = np.zeros(finger_rmodel.nq)

    # Place the cube:
    cube_q_idx = len(go2_init_conf0)
    q0[cube_q_idx:cube_q_idx+3] = [0.0, 0.0, 0.0]     # cube pos
    q0[cube_q_idx+3:cube_q_idx+7] = [1.0, 0.0, 0.0, 0.0]  # cube orientation (quat)

    # For velocities:
    v0[:len(go2_v0)] = go2_v0

    # Load MuJoCo model from XML and insert cube:
    with open(xml_path, 'r') as f:
        xml_str = f.read()

    # Modify meshdir to a proper path (e.g., absolute path)
    mesh_dir_path = os.path.join(package_dir, "trifinger", "meshes")
    xml_str = re.sub(r'meshdir="meshes"', f'meshdir="{mesh_dir_path}"', xml_str)

    cube_body_str = """
    <body name="cube" pos="0 0 0">
      <joint name="cube_freejoint" type="free"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.05" rgba="0 1 0 1"/>
    </body>
    """

    # Insert cube body right after <worldbody>
    xml_str = re.sub(r'(<worldbody[^>]*>)', r'\1' + cube_body_str, xml_str)

    mj_model = mujoco.MjModel.from_xml_string(xml_str)

    env = {
        "rmodel": rmodel,
        "gmodel": gmodel,
        "mj_model": mj_model,
        "q0": list(q0),
        "v0": list(v0),
        "nu": 9,  # trifinger has 9 actuators, cube not actuated
        "njoints": 9,
        "ncontacts": 0,
        "contactFnames": [],
        "contactFids": []
    }

    # Sanity check
    xml_urdf_sanity_check(mj_model, rmodel)

    return env

# Do some test here
def main():
    env = create_trifinger_cube_env()
    mj_model = env["mj_model"]
    mj_data = mujoco.MjData(mj_model)
    import mujoco.viewer as viewer
    viewer.launch(mj_model)

    # import mujoco_viewerx
    # viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data)
    # for _ in range(10000000000000000):
    #     print(f"cube position: {mj_data.qpos[9:12]}")
    #     if viewer.is_alive:
    #         mujoco.mj_step(mj_model, mj_data)
    #         viewer.render()
    #     else:
    #         break

if __name__ == "__main__":
    main()
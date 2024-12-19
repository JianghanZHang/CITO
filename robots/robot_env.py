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
    urdf_path = os.path.join(package_dir, "trifinger/trifinger_scene.urdf")
    xml_path = os.path.join(package_dir, "trifinger/trifinger_scene.xml")
    
    package_dirs = [package_dir]

    print(f"URDF Path: {urdf_path}")
    print(f"Package Dirs: {package_dirs}")

    pin_model, _, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs, verbose=True)
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    xml_urdf_sanity_check(mj_model, pin_model, with_cube=False)

    return pin_model, mj_model

# For Pinocchio
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

    # Load Pinocchio model for trifinger
    finger_pin_model, finger_gmodel, _ = pin.buildModelsFromUrdf(
        urdf_path, package_dirs, verbose=True
    )

    # Create cube models
    cube_pin_model, cube_gmodel, _ = create_cube("cube", color=[0., 1., 0., 1.])

    # Append cube to trifinger model
    pin_model_combined, gmodel_combined = pin.appendModel(
        finger_pin_model, cube_pin_model, finger_gmodel, cube_gmodel, 0, pin.SE3.Identity()
    )

    pin_model = pin_model_combined
    gmodel = gmodel_combined
    

    q0 = []
    v0 = []
    # Setup initial state of the cube
    q0[:cube_pin_model.nq] = np.zeros(cube_pin_model.nq)
    v0[:cube_pin_model.nq] = np.zeros(cube_pin_model.nq)

    # Setup initial state of the Trifinger
    q0[-finger_pin_model.nq:] = np.zeros(finger_pin_model.nq)
    v0[-finger_pin_model.nq:] = np.zeros(finger_pin_model.nq)

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

    # Sanity check
    xml_urdf_sanity_check(mj_model, pin_model)

    return pin_model, mj_model


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



# Do some test here
def main():
    _, mj_model = create_trifinger_cube_env()
    import mujoco.viewer as viewer
    viewer.launch(mj_model)


if __name__ == "__main__":
    main()
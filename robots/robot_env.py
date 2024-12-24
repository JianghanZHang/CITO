import sys
import os
import re
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import pinocchio as pin
import numpy as np
import mujoco  # Ensure MuJoCo is installed and properly licensed
import hppfcl as fcl

FINGER_CONFIGURATION = [0.0, -0.5, 0.4]

def create_trifinger_cube_env():
    package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    urdf_path = os.path.join(package_dir, "trifinger", "trifinger_cube_scene.urdf")
    xml_path = os.path.join(package_dir, "trifinger", "trifinger_scene.xml")
    
    package_dirs = [os.path.dirname(urdf_path)]
    
    print(f"URDF Path: {urdf_path}")
    print(f"Package Dirs: {package_dirs}")

    # Load Pinocchio model from URDF
    try:
        pin_model, gmodel, vmodel = pin.buildModelsFromUrdf(
            urdf_path, package_dirs, verbose=True
        )
    except Exception as e:
        print(f"Error loading URDF with Pinocchio: {e}")
        sys.exit(1)
    
    # Create initial positions and velocities
    q0 = np.zeros(pin_model.nq)
    v0 = np.zeros(pin_model.nv)
    
    # Load MuJoCo model from XML
    try:
        with open(xml_path, 'r') as f:
            xml_str = f.read()
    except Exception as e:
        print(f"Error reading MuJoCo XML file: {e}")
        sys.exit(1)

    # Ensure correct XML path
    xml_path = os.path.join(package_dir, "trifinger", "trifinger_cube_scene.xml")

    try:
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        sys.exit(1)

    # Perform sanity check
    xml_urdf_sanity_check(mj_model, pin_model, with_cube=True)

    # Get cube frame id
    cube_frame_id = pin_model.getFrameId("cube_link")
    
    return pin_model, mj_model, cube_frame_id

def xml_urdf_sanity_check(mj_model, pin_model, with_cube=True):
    # Create data
    pin_data = pin.Data(pin_model)
    mj_data = mujoco.MjData(mj_model)

    print("Pinocchio nq:", pin_model.nq)
    print("MuJoCo nq:", mj_model.nq)
    
    q_finger = np.array(3 * FINGER_CONFIGURATION)

    q_cube_mj = np.array([0, 0, 0, 
                          
                          1, 0, 0, 0])
    
    q_cube_pin = np.array([0, 0, 0, 
                           
                           0, 0, 0, 1])

    mj_data.qpos[:] = np.hstack((q_finger, q_cube_mj))
    mujoco.mj_forward(mj_model, mj_data)

    pin_q = np.hstack((q_finger, q_cube_pin))

    pin.forwardKinematics(pin_model, pin_data, pin_q)
    pin.updateFramePlacements(pin_model, pin_data)

    # Define lists of links to compare
    upper_links = ["finger_upper_link_0", "finger_upper_link_120", "finger_upper_link_240"]
    lower_links = ["finger_lower_link_0", "finger_lower_link_120", "finger_lower_link_240"]
    tips_xml = ["finger_tip_0", "finger_tip_120", "finger_tip_240"]
    tips_urdf = ["finger_tip_link_0", "finger_tip_link_120", "finger_tip_link_240"]

    # Compare upper links positions
    print("\nUpper links position comparison:")
    for ul in upper_links:
        # Get MuJoCo body id and position
        try:
            mj_body_id = mj_model.body(ul).id
            mj_body_pos = mj_data.xpos[mj_body_id]
        except KeyError:
            print(f"  MuJoCo body '{ul}' not found.")
            continue

        # Get Pinocchio frame id and position
        try:
            pin_frame_id = pin_model.getFrameId(ul)
            pin_body_pos = pin_data.oMf[pin_frame_id].translation
        except pin.PinocchioError:
            print(f"  Pinocchio frame '{ul}' not found.")
            continue

        print(f"{ul}:")
        print("  MuJoCo position:", mj_body_pos)
        print("  URDF/Pinocchio position:", pin_body_pos)

    # Compare lower links positions
    print("\nLower links position comparison:")
    for ll in lower_links:
        # Get MuJoCo body id and position
        try:
            mj_body_id = mj_model.body(ll).id
            mj_body_pos = mj_data.xpos[mj_body_id]
        except KeyError:
            print(f"  MuJoCo body '{ll}' not found.")
            continue

        # Get Pinocchio frame id and position
        pin_frame_id = pin_model.getFrameId(ll)
        pin_body_pos = pin_data.oMf[pin_frame_id].translation

        print(f"{ll}:")
        print("  MuJoCo position:", mj_body_pos)
        print("  URDF/Pinocchio position:", pin_body_pos)

    # Compare tip links positions
    print("\nTip links position comparison:")
    for idx in range(len(tips_xml)):
        # Get MuJoCo site id and position
        try:
            mj_site_id = mj_model.site(tips_xml[idx]).id
            mj_site_pos = mj_data.site_xpos[mj_site_id]
        except KeyError:
            print(f"  MuJoCo site '{tips_xml[idx]}' not found.")
            continue

        # Get Pinocchio frame id and position
        try:
            pin_frame_id = pin_model.getFrameId(tips_urdf[idx])
            pin_site_pos = pin_data.oMf[pin_frame_id].translation
        except pin.PinocchioError:
            print(f"  Pinocchio frame '{tips_urdf[idx]}' not found.")
            continue

        print(f"{tips_xml[idx]}:")
        print("  MuJoCo position:", mj_site_pos)
        print("  URDF/Pinocchio position:", pin_site_pos)

    # If with_cube is True, also check cube positions/orientations
    if with_cube:
        try:
            cube_body_id = mj_model.body("cube_link").id
            mj_cube_pos = mj_data.xpos[cube_body_id]
            mj_cube_mat = mj_data.xmat[cube_body_id].reshape(3,3)
        except KeyError:
            print("  MuJoCo body 'cube_link' not found.")
            mj_cube_pos = None
            mj_cube_mat = None

        try:
            cube_frame_id = pin_model.getFrameId("cube_link")
            pin_cube_pos = pin_data.oMf[cube_frame_id].translation
            pin_cube_mat = pin_data.oMf[cube_frame_id].rotation
        except pin.PinocchioError:
            print("  Pinocchio frame 'cube_link' not found.")
            pin_cube_pos = None
            pin_cube_mat = None

        print("\nCube body comparison:")
        if mj_cube_pos is not None and pin_cube_pos is not None:
            print("MuJoCo cube position:", mj_cube_pos)
            print("Pinocchio cube position:", pin_cube_pos)
            print("MuJoCo cube rotation:\n", mj_cube_mat)
            print("Pinocchio cube rotation:\n", pin_cube_mat)
        else:
            print("  Unable to compare cube positions and rotations.")

def main():
    # Create the environment and load models
    pin_model, mj_model, cube_frame_id = create_trifinger_cube_env()

    # Initialize MuJoCo data
    mj_data = mujoco.MjData(mj_model)

    # Initialize the viewer in static mode
    # Instead of using viewer.launch(), we create a Viewer instance and control the loop

    import mujoco_viewer


    # import mujoco.viewer as viewer
    # viewer_instance = viewer.launch(mj_model, mj_data)

    # # Optional: Set initial joint positions if needed
    # # For example, set all joints to zero

    FINGER_CONFIGURATION = [0.0, -1.0, -1.0]

    q_finger = np.array(3 * FINGER_CONFIGURATION)

    q_cube_mj = np.array([0, 0, 0.031, 
                          
                          1, 0, 0, 0])
    
    qpos = np.hstack((q_finger, q_cube_mj))

    mj_data.qpos[:] = qpos.copy()
    # mujoco.mj_forward(mj_model, mj_data)

    # create the viewer object
    viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data)

    # simulate and render
    for _ in range(100000000000):
        if viewer.is_alive:
            mj_data.qpos = qpos.copy()
            mujoco.mj_step(mj_model, mj_data)
            viewer.render()

        else:
            break

    # After closing the viewer, proceed with the rest of the code
    print("\nTrifinger Model Frames:")
    print("*************************************************************************")
    for frame in pin_model.frames:
        print(frame)
    print("*************************************************************************")
    
    print("\nTrifinger Model Joints:")
    for joint in pin_model.joints:
        print(joint)
    print("*************************************************************************")

    # Update Pinocchio kinematics
    q0 = np.zeros(pin_model.nq)
    v0 = np.zeros(pin_model.nv)
    pin_data = pin.Data(pin_model)

    pin.forwardKinematics(pin_model, pin_data, q0, v0)
    pin.updateFramePlacements(pin_model, pin_data)

if __name__ == "__main__":
    main()

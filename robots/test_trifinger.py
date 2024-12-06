# Import necessary libraries
import sys
import os
import mujoco
import pinocchio as pin
import numpy as np

# Set up paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../robots/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'python/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'robots/')))

package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
urdf_path = os.path.join(package_dir, "trifinger/trifinger.urdf")
xml_path = os.path.join(package_dir, "trifinger/trifinger.xml")

package_dirs = [package_dir]

print(f"URDF Path: {urdf_path}")
print(f"Package Dirs: {package_dirs}")

# Load models
rmodel, gmodel, vmodel = pin.buildModelsFromUrdf(
    urdf_path, package_dirs, root_joint=pin.JointModelFreeFlyer(), verbose=True
)
mj_model = mujoco.MjModel.from_xml_path(xml_path)

# Print Pinocchio joints
print("\n--- Pinocchio Joints ---")
for i, joint in enumerate(rmodel.joints):
    print(f"Joint {i}: {joint.shortname()} at index {joint.idx_q}")

# Print MuJoCo joints
print("\n--- MuJoCo Joints ---")
for i in range(mj_model.njnt):
    joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_type = mj_model.jnt_type[i]
    print(f"Joint {i}: {joint_name}, Type: {joint_type}")

# Debugging breakpoint
import pdb; pdb.set_trace()

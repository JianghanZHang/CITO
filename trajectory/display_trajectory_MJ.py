import numpy as np
import crocoddyl
from mim_solvers import SolverSQP, SolverCSQP
import os
import sys
import subprocess
outer_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the outer folder to the system path
sys.path.insert(-1, outer_folder_path)
# from differential_model_force_free import DifferentialActionModelForceExplicit
import pinocchio as pin
import meshcat
import time
from utils import Arrow
from complementarity_constraints_force_free import ResidualModelComplementarityErrorNormal, ResidualModelFrameTranslationNormal, ResidualModelComplementarityErrorTangential
from friction_cone import ResidualLinearFrictionCone
from force_derivatives import LocalWorldAlignedForceDerivatives
from robot_env import create_go2_env_force_MJ, create_go2_env
from utils import load_arrays
import imageio
formatter = {'float_kind': lambda x: "{:.4f}".format(x)}

np.set_printoptions(linewidth=210, precision=5, suppress=False, formatter=formatter)

# Create the robot
pin_env = create_go2_env()
rmodel = pin_env["rmodel"]
rdata = rmodel.createData()

mj_env = create_go2_env_force_MJ()
q0 = mj_env["q0"]
v0 = mj_env["v0"]
nu = mj_env["nu"]
fids = pin_env["contactFids"]
njoints = mj_env["njoints"]
ncontacts = mj_env["ncontacts"]
nq = mj_env["nq"]
nv = mj_env["nv"]
mj_model = mj_env["mj_model"]
# mj_data = mj_env["mj_data"]

###################
# dt = mj_model.opt.timestep         #
dt = 0.001
T = 1000            #
###################

robot = "go2"
task = "takeoff_MJ_CSQP"
file = robot + "_" + task
xs, us = load_arrays(file)
fps = int(1/dt) 
fps = int(fps/2)

np.set_printoptions(linewidth=210, precision=4, suppress=False, formatter=formatter)

from pinocchio.visualize import MeshcatVisualizer
viz = MeshcatVisualizer(rmodel, pin_env["gmodel"], pin_env["vmodel"])
import zmq
try:
    viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
except zmq.ZMQError as e:
    print(f"Failed to connect to Meshcat server: {e}")

print('Connected to meshcat')
viz.initViewer(viewer)
viz.loadViewerModel()
viz.initializeFrames()
viz.display_frames = True
arrows = []
fids = fids

PLOT_DIRECTORY = "visualizations/plots/ocp/"
VIDEO_DIRECTORY = "visualizations/videos/ocp/"
IMAGE_DIRECTORY = "visualizations/frames/"  # Directory to store individual frames
if not os.path.exists(IMAGE_DIRECTORY):
    os.makedirs(IMAGE_DIRECTORY)
# with_clearance = "no_clearance_"
with_clearance = "with_clearance_"
TASK = "walking_"

input("Press if the visualizer is ready")

# Store frames as images
for i in range(len(xs)-1):
    force_t = us[i][njoints:]
    x_t = xs[i]
    print(f"\n********************Time:{i*dt}********************\n")
    print(f'Base position:{x_t[:3]}')
    for eff, fid in enumerate(fids):
        q, v = x_t[:rmodel.nq], x_t[rmodel.nq:]
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.computeAllTerms(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)

    # Capture frames for video
    viz.display(xs[i][:rmodel.nq])
    frame = np.asarray(viz.viewer.get_image())  # Modify this line to capture frames properly in the right format
    imageio.imwrite(os.path.join(IMAGE_DIRECTORY, f'frame_{i:04d}.png'), frame)

    viz.setCameraTarget(np.array([x_t[0]+0.1, 0.0, 0.0]))
    viz.setCameraPosition(np.array([0.5, -1.0, 0.3]))

# Call FFmpeg manually to convert the images into a video with the probesize option
output_file = VIDEO_DIRECTORY + TASK + with_clearance + "CSQP" + "_1ms.mp4"
subprocess.call([
    'ffmpeg', '-y', '-probesize', '50M', '-framerate', str(fps),
    '-i', os.path.join(IMAGE_DIRECTORY, 'frame_%04d.png'),
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_file
])

print(f"Video saved to {output_file}")

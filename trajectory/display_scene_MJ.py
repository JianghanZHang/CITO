import os
import sys

# Get the path to the outer folder
outer_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the outer folder to the system path
sys.path.insert(-1, outer_folder_path)
import numpy as np
import crocoddyl
from mim_solvers import SolverSQP, SolverCSQP
from differential_model_force_free import DifferentialActionModelForceExplicit
import pinocchio as pin
import meshcat
import time
from utils import Arrow
from complementarity_constraints_force_free import ResidualModelComplementarityErrorNormal, ResidualModelFrameTranslationNormal, ResidualModelComplementarityErrorTangential
from friction_cone import  ResidualLinearFrictionCone
from force_derivatives import LocalWorldAlignedForceDerivatives
from robots.robot_env import create_go2_env_force_MJ, create_go2_env
from trajectory_data import load_arrays
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
mj_data = mj_env["mj_data"]

###################
dt = mj_model.opt.timestep         #
T = 50            #
###################

robot = "go2"
task = "takeoff_MJ"
file =  robot + "_" + task
xs, us = load_arrays(file)


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
# import pdb; pdb.set_trace
for i in range(len(fids)):
    arrows.append(Arrow(viz.viewer, "force" + str(i), length_scale=0.01))


q0 = np.array([0.0000, 0.0000, 0.2700 + 0.1225, 
                0.0000, 0.0000, 0.0000, 1.0000, 
                0.0000, 0.4000, -0.8000, 
                0.0000, 0.4000, -0.8000, 
                0.0000, 0.4000, -0.8000, 
                0.0000, 0.4000, -0.8000])
v0 = np.zeros(nv)
x0 = np.hstack([q0, v0])
print(f'q0: {x0[:rmodel.nq]}')
viz.display(q0) # Display the initial state

for eff, fid in enumerate(fids):
        q, v = x0[:rmodel.nq], x0[rmodel.nq:]
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.computeAllTerms(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)
        # cntForce, _, _ = LocalWorldAlignedForceDerivatives(force_t[3*eff:3*(eff+1)], x_t, fids[eff], rmodel, rdata)
        print(f'foot id:{eff}')
        # print(f'contact forces:{force_t[3*eff:3*(eff+1)]}')
        print(f'distance:{rdata.oMf[fid].translation[2]}')
# input("Press if the visualizer is ready")

# for i in range(len(xs)-1):
#     force_t = us[i][njoints:]
#     x_t = xs[i]
#     print(f"\n********************Time:{i*dt}********************\n")
#     print(f'Base position:{x_t[:3]}')
#     for eff, fid in enumerate(fids):
#         q, v = x_t[:rmodel.nq], x_t[rmodel.nq:]
#         pin.framesForwardKinematics(rmodel, rdata, q)
#         pin.computeAllTerms(rmodel, rdata, q, v)
#         pin.updateFramePlacements(rmodel, rdata)
#         # cntForce, _, _ = LocalWorldAlignedForceDerivatives(force_t[3*eff:3*(eff+1)], x_t, fids[eff], rmodel, rdata)
#         print(f'foot id:{eff}')
#         # print(f'contact forces:{force_t[3*eff:3*(eff+1)]}')
#         print(f'distance:{rdata.oMf[fid].translation[2]}')
#         print(f'complementarity constraint:{rdata.oMf[fid].translation[2] * force_t[3*eff:3*(eff+1)]}')
#         # arrows[eff].anchor_as_vector(rdata.oMf[fid].translation, force_t[3*eff:3*(eff+1)].copy())        
#     time.sleep(0.1)
#     viz.display(xs[i][:rmodel.nq])


input()
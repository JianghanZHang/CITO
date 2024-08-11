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
from robot_env import create_solo12_env_free_force
from trajectory_data import load_arrays
formatter = {'float_kind': lambda x: "{:.4f}".format(x)}
    
np.set_printoptions(linewidth=210, precision=5, suppress=False, formatter=formatter)

# Create the robot
env = create_solo12_env_free_force()
nq = env["nq"]
nv = env["nv"]
njoints = env['njoints']
nu = env['nu']
nx = nq + nv
ncontacts = env['ncontacts']
q0 = env['q0']
v0 = env['v0']
rmodel = env['rmodel']
rdata = rmodel.createData()
dt = 5e-3
T = 100

robot = "go2"
task = "takeoff"
file = robot + "_" + task
xs, us = load_arrays(file)

input('Press a key to start the visualization')

viz = pin.visualize.MeshcatVisualizer(rmodel, env["gmodel"], env["vmodel"])
import zmq
try:
    viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
except zmq.ZMQError as e:
    print(f"Failed to connect to Meshcat server: {e}")

viz.initViewer(viewer)
viz.loadViewerModel()
viz.initializeFrames()
viz.display_frames = True
arrows = []
fids = env["contactFids"]
for i in range(len(fids)):
    arrows.append(Arrow(viz.viewer, "force" + str(i), length_scale=0.01))

for i in range(len(xs)-1):
    force_t = us[i][env["njoints"]:]
    x_t = xs[i]
    T = pin.utils.XYZQUATToSe3(x_t[:7])
    print(f"\n********************Time:{i*dt}********************\n")
    print(f'Base position:{T.translation}')
    print(f'Base linear velocity:{x_t[3:6]}')

    print(f'Base rpy:{pin.utils.matrixToRpy(T.rotation)}')
    print(f'Base angular velocity:{x_t[9:12]}')
    for eff, fid in enumerate(fids):
        q, v = x_t[:rmodel.nq], x_t[rmodel.nq:]
        cntForce, _, _ = LocalWorldAlignedForceDerivatives(force_t[3*eff:3*(eff+1)], x_t, fids[eff], rmodel, rdata)
        # print(cntForce, force_t[3*eff:3*(eff+1)])
        print(f'foot id:{eff}')
        print(f'joint torques:{us[i][(3 * eff) :3 * (eff + 1)]}')
        print(f'contact forces:{force_t[3*eff:3*(eff+1)]}')
        print(f'distance:{rdata.oMf[fid].translation[2]}')
        print(f'complementarity constraint:{rdata.oMf[fid].translation[2] * force_t[3*eff:3*(eff+1)]}')
        arrows[eff].anchor_as_vector(rdata.oMf[fid].translation, force_t[3*eff:3*(eff+1)].copy())        
    # time.sleep(0.1)
    viz.display(xs[i][:rmodel.nq])
    input()


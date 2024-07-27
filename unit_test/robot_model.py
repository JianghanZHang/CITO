import numpy as np
import crocoddyl
from mim_solvers import SolverSQP, SolverCSQP
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from differential_model_force_free import DifferentialActionModelForceExplicit
import pinocchio as pin
import meshcat
import time
from utils import Arrow
from complementarity_constraints_force_free import ResidualModelComplementarityErrorNormal, ResidualModelFrameTranslationNormal, ResidualModelComplementarityErrorTangential
from solo12_env import create_solo12_env

# Create the robot
env = create_solo12_env()
nq = env["nq"]
nv = env["nv"]
njoints = env['njoints']
nu = env['nu']
nx = nq + nv
ncontacts = env['ncontacts']
q0 = env['q0']
v0 = env['v0']
rmodel = env['rmodel']
contactFids = env['contactFids']
rdata = rmodel.createData()
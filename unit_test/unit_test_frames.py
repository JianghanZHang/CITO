# unit test functions
import numpy as np
from numerical_differences import numdiffSE3toEuclidian, tol

from frame_derivatives import FrameTranslation, FramePlacementError
from cube_model import *


nq, nv = rmodel.nq, rmodel.nv

rot = np.pi*np.random.rand(3)
quat = pin.Quaternion(pin.utils.rpyToMatrix(rot[0], rot[1], rot[2]))
q0 = np.hstack((2*np.random.rand(3), np.array([quat[0], quat[1], quat[2], quat[3]])))
v0 = np.random.rand(rmodel.nv)
x0 = np.hstack((q0, v0))
grf = np.array([0, 0, mass*9.81, 0, 0, 0]) # ground reaction forces

oMd = pin.SE3(pin.Quaternion(pin.utils.rpyToMatrix(rot[0], rot[1], rot[2])), 2*np.random.rand(3))

val, d1 = FrameTranslation(x0, fids[0], rmodel, rdata)
arg1_func = lambda x : FrameTranslation(x, fids[0], rmodel, rdata)

Jnumdiff_arg1 = numdiffSE3toEuclidian(arg1_func, x0, rmodel)

assert np.linalg.norm(d1 - Jnumdiff_arg1) < tol

val, d1 = FramePlacementError(x0, oMd, fids[0], rmodel, rdata)
arg1_func = lambda x : FramePlacementError(x, oMd, fids[0], rmodel, rdata)

Jnumdiff_arg1 = numdiffSE3toEuclidian(arg1_func, x0, rmodel)

assert np.linalg.norm(d1 - Jnumdiff_arg1) < tol

print("unit test passed ...")
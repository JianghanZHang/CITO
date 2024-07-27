# unit test functions
import numpy as np
from numerical_differences import numdiff, numdiffSE3toEuclidian, tol


from force_derivatives import FrameTranslationForce, LocalWorldAlignedForce
from robot_model import *


F =  10 *  np.random.rand(3)
p0 = np.random.rand(3)
nq, nv = rmodel.nq, rmodel.nv

rot = np.pi*np.random.rand(3)
quat = pin.Quaternion(pin.utils.rpyToMatrix(rot[0], rot[1], rot[2]))
q0 = np.hstack((np.random.rand(3), np.array([quat[0], quat[1], quat[2], quat[3]])))
v0 = np.random.rand(rmodel.nv)
x0 = np.hstack((q0, v0))
grf = np.array([0, 0, mass*9.81, 0, 0, 0]) # ground reaction forces


val, d1, d2 = FrameTranslationForce(F, p0)
arg1_func = lambda F : FrameTranslationForce(F, p0)
arg2_func = lambda p : FrameTranslationForce(F, p)

Jnumdiff_arg1 = numdiff(arg1_func, F)
Jnumdiff_arg2 = numdiff(arg2_func, p0)

assert np.linalg.norm(d1 - Jnumdiff_arg1) < tol
assert np.linalg.norm(d2 - Jnumdiff_arg2) < tol

val, d1, d2 = LocalWorldAlignedForce(F, x0, fids[0], rmodel, rdata)
arg1_func = lambda F : LocalWorldAlignedForce(F, x0, fids[0], rmodel, rdata)
arg2_func = lambda x : LocalWorldAlignedForce(F, x, fids[0], rmodel, rdata)

Jnumdiff_arg1 = numdiff(arg1_func, F)
Jnumdiff_arg2 = numdiffSE3toEuclidian(arg2_func, x0, rmodel)

assert np.linalg.norm(d1 - Jnumdiff_arg1) < tol
assert np.linalg.norm(d2 - Jnumdiff_arg2) < tol


print("unit test passed ...")
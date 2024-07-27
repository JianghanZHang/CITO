# unit test functions
import numpy as np
from numerical_differences import numdiff, numdiffSE3toEuclidian, numdiffEuclidiantoSE3, numdiffSE3toSE3, tol

from dynamics import ContinousDynamicsDerivatives, DiscreteDynamicsDerivatives
from cube_model import *

nq, nv = rmodel.nq, rmodel.nv

rot = np.pi*np.random.rand(3)
quat = pin.Quaternion(pin.utils.rpyToMatrix(rot[0], rot[1], rot[2]))
q0 = np.hstack((np.random.rand(3), np.array([quat[0], quat[1], quat[2], quat[3]])))
v0 = np.random.rand(rmodel.nv)
x0 = np.hstack((q0, v0))
grf = np.array([0, 0, mass*9.81, 0, 0, 0]) # ground reaction forces
F =  10 *  np.random.rand(3)

val, d1, d2  = ContinousDynamicsDerivatives(x0, [F], fids, rmodel, rdata, grf)

arg1_func = lambda x : ContinousDynamicsDerivatives(x, [F], fids, rmodel, rdata, grf)
arg2_func = lambda F : ContinousDynamicsDerivatives(x0, [F], fids, rmodel, rdata, grf)


Jnumdiff_arg1 = numdiffSE3toEuclidian(arg1_func, x0, rmodel)
Jnumdiff_arg2 = numdiff(arg2_func, F)

assert np.linalg.norm(d1 - Jnumdiff_arg1) < tol
assert np.linalg.norm(d2 - Jnumdiff_arg2) < tol

dt = 0.1*np.random.rand()

val, d1, d2  = DiscreteDynamicsDerivatives(x0, [F], fids, rmodel, rdata, dt, grf)

arg1_func = lambda x : DiscreteDynamicsDerivatives(x, [F], fids, rmodel, rdata, dt, grf)
arg2_func = lambda F : DiscreteDynamicsDerivatives(x0, [F], fids, rmodel, rdata, dt, grf)

Jnumdiff_arg1 = numdiffSE3toSE3(arg1_func, x0, rmodel)
Jnumdiff_arg2 = numdiffEuclidiantoSE3(arg2_func, F, rmodel)

assert np.linalg.norm(d1 - Jnumdiff_arg1) < tol
assert np.linalg.norm(d2 - Jnumdiff_arg2) < tol


print("unit test passed ...")

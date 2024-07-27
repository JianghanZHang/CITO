# this is a unit test for the kinematic problem and qp (double integrator system)
# Author : Avadesh Meduri
# Date : 30/05/2024

import numpy as np
from kinematic_problem import KinematicProblem
from diff_qp import quadprog_solve_qp
import matplotlib.pyplot as plt

# parameters
N = 20
dt = 0.1
nq = nv = 2
nbForces = 0
q_lim =  np.array([0.5,0.5])
v_lim = tau_lim =  10* np.array([0.5,0.5])
xinit = 0.3*np.ones(nq + nv)
xdes = np.array([-1.5, 0, 0, 0])

wt = np.array([1., 1., 0.01, 0.01])
wt_ter = 1e3*wt.copy()

problem = KinematicProblem(N, nq, tau_lim, dt)
Q, q, A, b, G, h  = problem.create_matrices(xinit, xdes, wt, wt_ter)

xopt, _ = quadprog_solve_qp(Q, q, G, h, A, b)

qxTraj = xopt[0::3*nq]
qyTraj = xopt[1::3*nq]
vxTraj = xopt[2::3*nq]
vyTraj = xopt[3::3*nq]
axTraj = xopt[4::3*nq]
ayTraj = xopt[5::3*nq]


print(xopt)

assert abs(qxTraj[0] - xinit[0]) < 1e-2
assert abs(qyTraj[0] - xinit[1]) < 1e-2

assert abs(qxTraj[-1] - xdes[0]) < 1e-2
assert abs(qyTraj[-1] - xdes[1]) < 1e-2

print("unit test passed")
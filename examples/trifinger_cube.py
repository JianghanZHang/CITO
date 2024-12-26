import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../robots/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'python/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'robots/')))
from robots.robot_env import create_trifinger_cube_env
import crocoddyl
import pinocchio as pin
import numpy as np
from CallbackLogger_force import CallbackLogger
import mim_solvers
import meshcat
import sys
import cito

def main():
   
    pin_model, mj_model, cube_frame_id = create_trifinger_cube_env()

    pin_data = pin_model.createData()
    

    nq = pin_model.nq
    nv = pin_model.nv

    v0 = list(np.zeros(nv))

    njoints = 9
    nu = 9
    ################# OCP params ################
    dt = 0.01                                 
    mj_model.opt.timestep = dt                
    T = 100                                              
    T_total = dt * T                          

    FINGER_CONFIGURATION = [0.0, -1.0, -1.0]

    q_finger = np.array(3 * FINGER_CONFIGURATION)

    q_cube_mj = np.array([0, 0, 0.031, 
                          
                          1, 0, 0, 0])
    
    q_cube_pin = np.array([0, 0, 0.031, 
                          
                          0, 0, 0, 1])
    


    q0 = list(np.hstack((q_finger, q_cube_pin)))

    # print(f'x0:{q0 + v0}')
    ################# Initialize crocoddyl models ################
    state = crocoddyl.StateMultibody(pin_model)
    
    # actuation = crocoddyl.ActuationModelFloatingBase(state)

    actuation = cito.ActuationModelFloatingBaseManipulation(state)

    print(f"NU: {actuation.nu}")
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)
    constraintModelManager = crocoddyl.ConstraintModelManager(state, nu)    


    ################ Control limits ################
    ControlLimit = np.array(3 * [10.0, 10.0, 10.0])
    ControlRedisual = crocoddyl.ResidualModelControl(state, nu)
    ControlLimitConstraint = crocoddyl.ConstraintModelResidual(state, ControlRedisual, -ControlLimit, ControlLimit)
    constraintModelManager.addConstraint("ControlLimitConstraint", ControlLimitConstraint)

    ################ Cube Target ################
    Px_des = 0.0
    Py_des = 0.0
    Pz_des = 0.2
    CubeTarget = np.array([Px_des, Py_des, Pz_des])

    for frame in pin_model.frames:
        print(frame)

    import pdb; pdb.set_trace()
    CubePositionActivation = crocoddyl.ActivationModelWeightedQuad(np.array(3 * [1.0]))
    CubePositionResidual = crocoddyl.ResidualModelFrameTranslation(state, id=cube_frame_id, xref=CubeTarget, nu=9)
    CubePositionCost = crocoddyl.CostModelResidual(state, CubePositionActivation, CubePositionResidual)
    runningCostModel.addCost("Cube_position_cost", CubePositionCost, 10)

    runningCostModel.addCost("Cube_position_cost", CubePositionCost, 100)

    Vx_des = Px_des/T_total
    Vy_des = Py_des/T_total
    Vz_des = Pz_des/T_total

    ################## Control Regularization Cost##################
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    uRegActivation = crocoddyl.ActivationModelWeightedQuad(np.array(9 * [1.0]))
    uRegCost = crocoddyl.CostModelResidual(state, uRegActivation, uResidual)
    runningCostModel.addCost("uReg", uRegCost, 1e-3)


    runningModels = [cito.IntegratedActionModelContactMj(
                    cito.DifferentialActionModelContactMj(mj_model, state, actuation, runningCostModel, constraintModelManager), dt) 
                    for _ in range(T)]
    
    terminal_DAM = cito.DifferentialActionModelContactMj(mj_model, state, actuation, terminalCostModel, None)
    
    terminalModel = cito.IntegratedActionModelContactMj(terminal_DAM, 0.)

    x0 = np.array(q0 + v0)
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)

    num_steps = T + 1 

    xs_init = [np.copy(x0) for _ in range(num_steps)]
    base_x_start = x0[0].copy()
    base_x_end = Px_des
    base_x_values = np.linspace(base_x_start, base_x_end, num_steps)
    us_init = [np.zeros(nu) for i in range(T)]

    # for i in range(1, num_steps):
    #         xs_init[i][0] = base_x_values[i] 
    #         xs_init[i][3] = Vx_des #assign desired base velocity to initial guess
    
    maxIter = 1000

    solver = mim_solvers.SolverCSQP(problem)
    solver.mu_constraint = 10.
    solver.mu_dynamic = 10.
    solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
    solver.use_filter_line_search = False
    solver.verbose = True
    solver.termination_tolerance = 1e-3
    solver.remove_reg = False
    solver.max_qp_iters =100

    print(f'Solving')
    flag = solver.solve(xs_init, us_init, maxiter=maxIter, isFeasible=False)

    xs, us = solver.xs, solver.us
    log = solver.getCallbacks()[-1]
    print(f'Solved: {flag}')


if __name__   == "__main__":
    main()




    



    

    
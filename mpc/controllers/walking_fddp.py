import numpy as np
import pinocchio as pin 
import mim_solvers
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../*')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/*')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../robots/*')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'python/*')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'robots/*')))

from python.ResidualModels import ResidualModelFootClearanceNumDiff
from differential_model_MJ import DifferentialActionModelMJ
from integrated_action_model_MJ import IntegratedActionModelForceMJ
import time

from robots.robot_env import go2_init_conf0
from croco_mpc_utils.ocp import OptimalControlProblemClassical
import croco_mpc_utils.pinocchio_utils as pin_utils
import crocoddyl 
import sys
from utils import stateMapping_mj2pin, stateMapping_pin2mj


from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

# @profile
def solveOCP(q, v, solver, max_sqp_iter, max_qp_iter, target_reach):
    t = time.time()
    # Update initial state + warm-start
    x = np.concatenate([q, v])
    solver.problem.x0 = x
    
    # initial guess
    xs_init = list(solver.xs[1:]) + [solver.xs[-1]]
    xs_init[0] = x
    us_init = list(solver.us[1:]) + [solver.us[-1]] 
    
    # Update OCP 
    for k in range( solver.problem.T ):
        solver.problem.runningModels[k].differential.costs.costs["xDes_running"].active = True
        solver.problem.runningModels[k].differential.costs.costs["xDes_running"].cost.residual.reference = target_reach[k]
        
    solver.problem.terminalModel.differential.costs.costs["xDes_terminal"].active = True
    solver.problem.terminalModel.differential.costs.costs["xDes_terminal"].cost.residual.reference = target_reach[-1]

    print(f'xDes[-1]:\n{target_reach[-1]}')
    solver.max_qp_iters = max_qp_iter

    solver.solve(xs_init, us_init, max_sqp_iter, False)
    solve_time = time.time() - t

    return  solver.us[0], solver.xs[1], None, solve_time - t, solver.iter, solver.cost#, solver.constraint_norm, solver.gap_norm, solver.qp_iters, solver.KKT

def create_walking_ocp(pin_model, mj_model, config):
    rmodel = pin_model.copy()

    fids = []
    frameNames = config['contactFrameNames']
    for idx, frameName in enumerate(frameNames):
        fids.append(rmodel.getFrameId(frameName))
    
    njoints = config['njoints']
    q0 = config['q0']
    v0 = config['v0']
    x0 = np.concatenate([q0, v0])
    # dt = config['dt']
    dt = mj_model.opt.timestep
    T = config['N_h']

    # Handling constraints
    if "ctrlBox" in config['WHICH_CONSTRAINTS']:
        ctrlLimit = np.asarray(config['ctrlLimit'])  

    if "stateBox" in config['WHICH_CONSTRAINTS']:
        stateLimit_lb = np.asarray(config['StateLimit_lb'])
        stateLimit_ub = np.asarray(config['StateLimit_ub'])
    
    
    # Handling costs
    if "stateReg" in config['WHICH_COSTS']:
        stateRegWeights = np.asarray(config['stateRegWeights'])
        stateRegWeightsTerminal = np.asarray(config['stateRegWeightsTerminal'])
        stateRegRef = np.asarray(config['stateRegRef'])
        stateRegWeight = config['stateRegWeight']
        stateRegWeightTerminal = config['stateRegWeightTerminal']

    if "ctrlReg" in config['WHICH_COSTS']:    
        ctrlRegWeight = config['ctrlRegWeight']
        ctrlRegWeights = np.asarray(config['ctrlRegWeights'])

    if "footClearance" in config['WHICH_COSTS']:
        footClearanceWeight = config['footClearanceWeight']
        # footClearanceWeight = 0.0
        footClearanceWeightTerminal = config['footClearanceWeightTerminal']
        footClearanceSigmoidSteepness = np.asarray(config['footClearanceSigmoidSteepness'])

    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu

    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)

    uResidual = crocoddyl.ResidualModelControl(state, nu)
    uRegActivation = crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights) 
    uRegCost = crocoddyl.CostModelResidual(state, uRegActivation, uResidual)

    runningCostModel.addCost("uReg", uRegCost, ctrlRegWeight)

    # Constraints (friction cones + complementarity contraints)
    constraintModelManager = crocoddyl.ConstraintModelManager(state, nu)



    for idx, fid in enumerate(fids):
        footClearanceResidual = ResidualModelFootClearanceNumDiff(state, nu, fid, sigmoid_steepness=-footClearanceSigmoidSteepness)
        footClearanceActivation = crocoddyl.ActivationModelSmooth1Norm(nr=1, eps=1e-12)
        footClearanceCost = crocoddyl.CostModelResidual(state, footClearanceActivation, footClearanceResidual)
        runningCostModel.addCost(f"footClearance_{idx}", footClearanceCost, footClearanceWeight)
        terminalCostModel.addCost(f"footClearance_{idx}", footClearanceCost, footClearanceWeightTerminal)
   
    
    xDesActivationRunning = crocoddyl.ActivationModelWeightedQuad(stateRegWeights)

    xDesActivationTerminal = crocoddyl.ActivationModelWeightedQuad(stateRegWeightsTerminal)

    xDesResidual = crocoddyl.ResidualModelState(state, stateRegRef, nu)
    xDesCostRunning = crocoddyl.CostModelResidual(state, xDesActivationRunning, xDesResidual)
    xDesCostTerminal = crocoddyl.CostModelResidual(state, xDesActivationTerminal, xDesResidual)

    runningCostModel.addCost("xDes_running", xDesCostRunning, stateRegWeight)
    terminalCostModel.addCost("xDes_terminal", xDesCostTerminal, stateRegWeightTerminal)


    terminal_DAM = DifferentialActionModelMJ(mj_model, state, nu, njoints, fids, terminalCostModel, None)
    

    runningModels = [IntegratedActionModelForceMJ
                     (DifferentialActionModelMJ(mj_model, state, nu, njoints, fids, runningCostModel, constraintModelManager), dt, True) 
                    for _ in range(T)]
    
    
    terminalModel = IntegratedActionModelForceMJ(terminal_DAM, 0., True)

    # # Control limits    
    for runningModel in runningModels:
        if "stateBox" in config['WHICH_CONSTRAINTS']:
            runningModel.x_lb = stateLimit_lb
            runningModel.x_ub = stateLimit_ub

        if "ctrlBox" in config['WHICH_CONSTRAINTS']:
            runningModel.u_lb = -ctrlLimit  
            runningModel.u_ub = ctrlLimit

    x0 = np.array(q0 + v0)
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)

    return problem

class Go2WalkingFDDP:

    def __init__(self, pin_robot, mj_model, config, run_sim):
        """
        Input:
            head              : thread head
            pin_robot         : pinocchio wrapper
            config            : MPC config yaml file
            run_sim           : boolean sim or real
        """
        self.mj_model = mj_model
        self.robot   = pin_robot
        # self.head    = head
        self.RUN_SIM = run_sim
        self.joint_positions  = config['q0']
        self.joint_velocities = config['v0']

        self.max_qp_iter = config['max_qp_iter']
        self.max_sqp_iter = config['maxiter']
        # self.joint_accelerations = head.get_sensor("joint_accelerations")
        # if not self.RUN_SIM:
        #     self.joint_torques     = head.get_sensor("joint_torques_total")
        #     self.joint_ext_torques = head.get_sensor("joint_torques_external")
        #     self.joint_cmd_torques = head.get_sensor("joint_torques_commanded")      


        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv
        self.T_total = config['T_total']

        logger.warning("Controlled model dimensions : ")
        logger.warning(" nq = "+str(self.nq))
        logger.warning(" nv = "+str(self.nv))
        
        # Config
        self.config = config
        if(self.RUN_SIM):
            self.q0 = np.asarray(config['q0'])
            self.v0 = self.joint_velocities.copy()  
        else:
            self.q0 = self.joint_positions.copy()
            self.v0 = self.joint_velocities.copy()
        self.x0 = np.concatenate([self.q0, self.v0])
        
        self.Nh = int(self.config['N_h'])
        # self.dt_ocp  = self.config['dt']
        self.dt_ocp = self.mj_model.opt.timestep
        self.dt_ctrl = 1./self.config['ctrl_freq']
        self.OCP_TO_CTRL_RATIO = int(self.dt_ocp/self.dt_ctrl)
        self.u0 = np.zeros(12)
        # Create OCP         
        problem = create_walking_ocp(self.robot.model, self.mj_model, self.config)

        # Initialize the solver
        if(config['SOLVER'] == 'FDDP'):
            logger.warning("Using the FDDP solver.")
            self.solver = crocoddyl.SolverBoxFDDP(problem)
        else:
            Exception("Solver not implemented yet.")
      

        # Allocate MPC data
        # self.K = self.solver.K[0]
        self.x_des = self.solver.xs[0]
        self.tau_ff = self.solver.us[0]
        self.tau = self.tau_ff.copy() ; self.tau_riccati = np.zeros(self.tau.shape)

        # Initialize target position array  
        self.target_states = np.zeros((self.Nh+1, self.nq+self.nv)) 

        # Solver logs
        self.t_child         = 0
        self.cost            = 0
        self.cumulative_cost = 0
        self.gap_norm        = np.inf
        self.constraint_norm = np.inf
        self.qp_iters        = 0
        self.KKT             = np.inf

        self.v_des = np.array([1.0000, 0.0000, 0.0000, 
                            0.0000, 0.0000, 0.0000, 
                            0.0000, 0.0000, 0.0000, 
                            0.0000, 0.0000, 0.0000, 
                            0.0000, 0.0000, 0.0000, 
                            0.0000, 0.0000, 0.0000])
        
        self.q_des = go2_init_conf0.copy()

        self.solver.xs = [self.x0 for i in range(self.config['N_h']+1)]
        self.solver.us = [self.u0 for i in range(self.config['N_h'])]

    def warmUp(self):
        self.max_sqp_iter = 400
        # self.u0 = pin_utils.get_u_grav(self.q0, self.robot.model, np.zeros(self.robot.model.nq))
        xs_init = [np.copy(self.x0) for _ in range(self.config['N_h']+1)]
        
                
        self.solver.xs = xs_init.copy()

        self.solver.us = [self.u0 for i in range(self.config['N_h'])]
        self.solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])

        q = self.joint_positions
        v = self.joint_velocities

        xs_init = list(self.solver.xs[1:]) + [self.solver.xs[-1]]
        for i in range(1, self.config['N_h']):
            xs_init[i][2] = 0.2800
        
        self.solver.xs = xs_init.copy()


        current_state = np.concatenate([q, v])
        current_position = current_state[:3]
        desired_position = current_position + self.v_des[:3] * (self.Nh * self.dt_ocp)
        desired_position[1] = 0.0
        desired_position[2] = 0.2800
        self.q_des[:3] = desired_position
        desired_state = np.concatenate([self.q_des, self.v_des])

        self.target_states[:] = desired_state
        

        print(f'Current state:\n {current_state}')
        print(f'Desired state:\n {self.target_states[-1]}')

        self.tau_ff, self.x_next, self.K, self.t_child, self.ddp_iter, self.cost = solveOCP(self.joint_positions, 
                                                                                          self.joint_velocities, 
                                                                                          self.solver, 
                                                                                          self.max_sqp_iter, 
                                                                                          self.max_qp_iter, 
                                                                                          self.target_states)

        self.cumulative_cost += self.cost
        self.max_sqp_iter = self.config['maxiter']
        self.max_qp_iter  = self.config['max_qp_iter']
    
    # This method read mj data and change to the pinochio convention.
    def updateState(self, q_mj, v_mj):
        x_mj = np.concatenate([q_mj, v_mj])
        x_pin, _, _ = stateMapping_mj2pin(x_mj, self.robot.model)
       
        self.joint_positions = x_pin[:self.nq].copy()
        self.joint_velocities = x_pin[self.nq:].copy()

    def computeControl(self):        
        # # # # # # # # # 
        # Read sensors  #
        # # # # # # # # # 
        q = self.joint_positions
        v = self.joint_velocities

        # When getting torque measurement from robot, do not forget to flip the sign
        if(not self.RUN_SIM):
            self.joint_torques_measured = -self.joint_torques_total  

        # # # # # # # # # 
        # # Update OCP  #
        # # # # # # # # # 
        current_state = np.concatenate([q, v])
        current_position = current_state[:3]
        desired_position = current_position + self.v_des[:3] * (self.Nh * self.dt_ocp)
        desired_position[1] = 0.0
        desired_position[2] = 0.2800
        self.q_des[:3] = desired_position
        desired_state = np.concatenate([self.q_des, self.v_des])

        self.target_states[:] = desired_state

        # print(f'Current state:\n {current_state}')
        # print(f'Desired state:\n {desired_state}')    
        # # # # # # #  
        # Solve OCP #
        # # # # # # # 
        self.tau_ff, self.x_des, self.K, self.t_child, self.ddp_iter, self.cost = solveOCP(q, 
                                                                                          v, 
                                                                                          self.solver, 
                                                                                          self.max_sqp_iter, 
                                                                                          self.max_qp_iter, 
                                                                                          self.target_states)

        # # # # # # # # 
        # Send policy #
        # # # # # # # #
        self.tau = self.tau_ff.copy()

        # Compute gravity
        # self.tau_gravity = pin.rnea(self.robot.model, self.robot.data, self.joint_positions, np.zeros(self.nv), np.zeros(self.nv))

        # if(self.RUN_SIM == False):
            # self.tau -= self.tau_gravity

        # pin.framesForwardKinematics(self.robot.model, self.robot.data, q)

        return self.tau_ff.copy(), self.x_des.copy(), self.cost
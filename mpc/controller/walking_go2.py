import numpy as np
import pinocchio as pin 
import mim_solvers
from ResidualModels import ResidualModelFootClearanceNumDiff
import time
from robot_env import go2_init_conf0
from croco_mpc_utils.ocp import OptimalControlProblemClassical
import croco_mpc_utils.pinocchio_utils as pin_utils
import crocoddyl 
import sys
sys.path.append('../../python/')

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


# @profile
def solveOCP(q, v, solver, max_sqp_iter, max_qp_iter, target_reach, TASK_PHASE):
    t = time.time()
    # Update initial state + warm-start
    x = np.concatenate([q, v])
    solver.problem.x0 = x
    
    xs_init = list(solver.xs[1:]) + [solver.xs[-1]]
    xs_init[0] = x
    us_init = list(solver.us[1:]) + [solver.us[-1]] 
        
    # Update OCP 
    if(TASK_PHASE == 1):
        # Updates nodes between node_id and terminal node 
        for k in range( solver.problem.T ):
            solver.problem.runningModels[k].differential.costs.costs["stateReg"].active = True
            solver.problem.runningModels[k].differential.costs.costs["stateReg"].cost.residual.reference = target_reach[k]
            solver.problem.runningModels[k].differential.costs.costs["stateReg"].weight = 10. 

        #     if(k > 0):    
        #         solver.problem.runningModels[k].differential.constraints.constraints['translationBox'].constraint.updateBounds(ee_lb, ee_ub)
        # solver.problem.terminalModel.differential.costs.costs["translation"].active = True
        # solver.problem.terminalModel.differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
        # solver.problem.terminalModel.differential.costs.costs["translation"].weight = 50.               
        # solver.problem.terminalModel.differential.constraints.constraints['translationBox'].constraint.updateBounds(ee_lb, ee_ub)
        
    solver.max_qp_iters = max_qp_iter
    solver.solve(xs_init, us_init, maxIter=max_sqp_iter, isFeasible=False)
    solve_time = time.time()

    solver.qp_iters = 1
    solver.KKT = np.inf
    
    return  solver.us[0], solver.xs[1], None, solve_time - t, solver.iter, solver.cost, solver.constraint_norm, solver.gap_norm, solver.qp_iters, solver.KKT

def create_walking_ocp(robot, config):
    rmodel = robot.model
    nq = robot.model.nq
    nv = robot.model.nv
    
    with_callbacks = config['with_callbacks']
    use_filter_line_search = config['use_filter_line_search']
    warm_start = config['warm_start']
    max_qp_iter = config['max_qp_iter']
    qp_termination_tol_abs = config['qp_termination_tol_abs']
    qp_termination_tol_rel = config['qp_termination_tol_rel']
    warm_start_y: False 
    reset_rho: False
    max_iter = config['maxiter']

    fids = []
    frameNames = config['contactFrameNames']
    for idx, frameName in enumerate(frameNames):
        fids.append(rmodel.getFrameId(frameName))
    solver_type = config['SOLVER']
    
    q0 = config['q0']
    v0 = config['v0']
    x0 = np.concatenate([q0, v0])
    dt = config['dt']
    T = config['N_h']

    solver_termination_tolerance = config['solver_termination_tolerance']
    # Handling constraints
    if "ctrlBox" in config['WHICH_CONSTRAINTS']:
        ctrlLimit = config['ctrlLimit']  

    if "stateBox" in config['WHICH_CONSTRAINTS']:
        stateLimit = config['stateLimit']
    
    
    # Handling costs
    if "stateReg" in config['WHICH_COSTS']:
        stateRegWeights = np.asarray(config['stateRegWeights'])
        stateRegWeightsTerminal = np.asarray(config['stateRegWeightsTerminal'])
        stateRegRef = np.asarray(config['stateRegRef'])
        stateRegWeight = np.asarray(config['stateRegWeight'])
        stateRegWeightTerminal = np.asarray(config['stateRegWeightTerminal'])

    if "uReg" in config['WHICH_COSTS']:    
        uRegWeight = config['uRegWeight']
        uRegWeights = np.asarray(config['uRegWeights'])

    if "footClearance" in config['WHICH_COSTS']:
        footClearanceWeight = np.asarray(config['footClearanceWeight'])
        footClearanceWeightTerminal = np.asarray(config['footClearanceWeightTerminal'])
        footClearanceSigmoidSteepness = config['footClearanceSigmoidSteepness']

    

    
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu

    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)

    uResidual = crocoddyl.ResidualModelControl(state, nu)
    uRegActivation = crocoddyl.ActivationModelWeightedQuad(np.array(4 * [1.0, 1.0, 1.0])) 
    uRegCost = crocoddyl.CostModelResidual(state, uRegActivation, uResidual)

    # Constraints (friction cones + complementarity contraints)
    constraintModelManager = crocoddyl.ConstraintModelManager(state, nu)
    constraintModelManager0 = crocoddyl.ConstraintModelManager(state, nu)

    # # Control limits
    ControlLimit = np.array(4 * [23.7, 23.7, 45.43])
    ControlRedisual = crocoddyl.ResidualModelControl(state, nu)
    ControlLimitConstraint = crocoddyl.ConstraintModelResidual(state, ControlRedisual, -ControlLimit, ControlLimit)
    constraintModelManager.addConstraint("ControlLimitConstraint", ControlLimitConstraint)


    for idx, fid in enumerate(fids):
        footClearanceResidual = ResidualModelFootClearanceNumDiff(state, nu, fid, sigmoid_steepness=-50)
        footClearanceActivation = crocoddyl.ActivationModelSmooth1Norm(nr=1, eps=1e-12)
        footClearanceCost = crocoddyl.CostModelResidual(state, footClearanceActivation, footClearanceResidual)
        runningCostModel.addCost(f"footClearance_{idx}", footClearanceCost, stateRegWeight)
        terminalCostModel.addCost(f"footClearance_{idx}", footClearanceCost, stateRegWeightTerminal)
   
    lb = np.array([0.01])
    ub = np.array([10.0])
    groundCollisionBounds = crocoddyl.ActivationBounds(lb, ub)
    groundCollisionActivation = crocoddyl.ActivationModelQuadraticBarrier(groundCollisionBounds)

    
    Px_des = 0.5
    Vx_des = 1.0

    P_des = [Px_des, 0.0, 0.2700]
    O_des = pin.Quaternion(pin.utils.rpyToMatrix(0.0, 0.0, 0.0))

    V_des = [Vx_des, 0.0, 0.0]
    W_des = [0.0, 0.0, 0.0]
    x_des = np.array(P_des + 
                    [O_des[0], O_des[1], O_des[2], O_des[3]] + 
                    q0[7:] +
                    V_des + 
                    W_des + 
                    v0[6:])
    
    xDesActivationRunning = crocoddyl.ActivationModelWeightedQuad(np.array(
                                                                    1 * [1e2] +  # base x position
                                                                    1 * [1e2] +  # base y position
                                                                    1 * [1e2] +  # base z position
                                                                    3 * [1e0] +  # base orientation
                                                                    12 * [0] +  #joint positions
                                                                    3 * [1e0] +  # base linear velocity
                                                                    3 * [1e0] +  # base angular velocity
                                                                    12 * [1e-2]))  # joint velocities

    xDesActivationTerminal = crocoddyl.ActivationModelWeightedQuad(np.array(
                                                                    1 * [1e3] +  # base x position
                                                                    1 * [1e3] +  # base y position
                                                                    1 * [1e3] +  # base z position
                                                                    3 * [1e1] +  # base orientation
                                                                    12 * [5e0] +  #joint positions
                                                                    3 * [0] +  # base linear velocity
                                                                    3 * [1e0] +  # base angular velocity
                                                                    12 * [1e-1]))  # joint velocities


    runningCostModel.addCost("uReg", uRegCost, 1e-3)

    xDesResidual = crocoddyl.ResidualModelState(state, x_des, nu)
    xDesCostRunning = crocoddyl.CostModelResidual(state, xDesActivationRunning, xDesResidual)
    xDesCostTerminal = crocoddyl.CostModelResidual(state, xDesActivationTerminal, xDesResidual)

    runningCostModel.addCost("xDes_running", xDesCostRunning, 10)
    terminalCostModel.addCost("xDes_terminal", xDesCostTerminal, 10)


    terminal_DAM = DifferentialActionModelForceMJ(mj_model, state, nu, njoints, fids, terminalCostModel, None)
    

    runningModels = [IntegratedActionModelForceMJ
                     (DifferentialActionModelForceMJ(mj_model, state, nu, njoints, fids, runningCostModel, constraintModelManager), dt, True) 
                    for _ in range(T)]
    
    runningModels[0] = IntegratedActionModelForceMJ(DifferentialActionModelForceMJ(mj_model, state, nu, njoints, fids, runningCostModel, constraintModelManager0), dt, False)
    
    terminalModel = IntegratedActionModelForceMJ(terminal_DAM, 0., True)

    
    for runningModel in runningModels:
        runningModel.x_lb = StateLimit_lb
        runningModel.x_ub = StateLimit_ub
        runningModel.u_lb = -ControlLimit  
        runningModel.u_ub = ControlLimit

    x0 = np.array(q0 + v0)
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    return problem
class GO2WALKINGCSQP:

    def __init__(self, head, pin_robot, config, run_sim):
        """
        Input:
            head              : thread head
            pin_robot         : pinocchio wrapper
            config            : MPC config yaml file
            run_sim           : boolean sim or real
        """
        self.robot   = pin_robot
        self.head    = head
        self.RUN_SIM = run_sim
        self.joint_positions  = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor("joint_velocities")
        self.joint_accelerations = head.get_sensor("joint_accelerations")
        if not self.RUN_SIM:
            self.joint_torques     = head.get_sensor("joint_torques_total")
            self.joint_ext_torques = head.get_sensor("joint_torques_external")
            self.joint_cmd_torques = head.get_sensor("joint_torques_commanded")      


        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv

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
        self.dt_ocp  = self.config['dt']
        self.dt_ctrl = 1./self.config['ctrl_freq']
        self.OCP_TO_CTRL_RATIO = int(self.dt_ocp/self.dt_ctrl)
        self.u0 = pin_utils.get_u_grav(self.q0, self.robot.model, np.zeros(self.robot.model.nq))
        # Create OCP 
        # problem = OptimalControlProblemClassical(self.robot, self.config).initialize(self.x0)

        problem = create_walking_ocp(self.robot, self.config)
        # Initialize the solver
        if(config['SOLVER'] == 'proxqp'):
            logger.warning("Using the ProxQP solver.")
            self.solver = mim_solvers.SolverProxQP(problem)
        elif(config['SOLVER'] == 'CSQP'):
            logger.warning("Using the CSSQP solver.")            
            self.solver = mim_solvers.SolverCSQP(problem)

            
        self.solver.with_callbacks         = self.config['with_callbacks']
        self.solver.use_filter_line_search = self.config['use_filter_line_search']
        self.solver.filter_size            = self.config['filter_size']
        self.solver.warm_start             = self.config['warm_start']
        self.solver.termination_tolerance  = self.config['solver_termination_tolerance']
        self.solver.max_qp_iters           = self.config['max_qp_iter']
        self.solver.eps_abs                = self.config['qp_termination_tol_abs']
        self.solver.eps_rel                = self.config['qp_termination_tol_rel']
        self.solver.warm_start_y           = self.config['warm_start_y']
        self.solver.reset_rho              = self.config['reset_rho']  
        self.solver.regMax                 = 1e6
        self.solver.reg_max                = 1e6

        # Allocate MPC data
        # self.K = self.solver.K[0]
        self.x_des = self.solver.xs[0]
        self.tau_ff = self.solver.us[0]
        self.tau = self.tau_ff.copy() ; self.tau_riccati = np.zeros(self.tau.shape)

        # Initialize torque measurements 
        if(self.RUN_SIM):
            logger.debug("Initial torque measurement signal : simulation --> use u0 = g(q0)")
            self.u0 = pin_utils.get_u_grav(self.q0, self.robot.model, np.zeros(self.robot.model.nq))
            self.joint_torques_total    = self.u0
            self.joint_torques_measured = self.u0
        # DANGER ZONE 
        else:
            logger.warning("Initial torque measurement signal : real robot --> use sensor signal 'joint_torques_total' ")
            self.joint_torques_total    = head.get_sensor("joint_torques_total")
            logger.warning("      >>> Correct minus sign in measured torques ! ")
            self.joint_torques_measured = -self.joint_torques_total 


        self.frameId = self.robot.model.getFrameId(self.config['frameTranslationFrameName'])

        # Initialize target position array  
        self.target_state = np.zeros((self.Nh+1, self.nq+self.nv)) 
        
        self.lb = np.array([-np.inf]*3)
        self.ub = np.array([np.inf]*3)
        
        self.TASK_PHASE = 0
        self.NH_SIMU   = int(self.Nh*self.dt_ocp/self.dt_ctrl)
        self.T_CIRCLE  = int(self.config['T_CIRCLE']/self.dt_ctrl)
        self.CIRCLE_DURATION = int(2 * np.pi/self.dt_ctrl)
        self.count_circle = 0
        logger.debug("Size of MPC horizon in simu cycles = "+str(self.NH_SIMU))
        logger.debug("Start of circle phase in simu cycles = "+str(self.T_CIRCLE))
        logger.debug("OCP to ctrl time ratio = "+str(self.OCP_TO_CTRL_RATIO))

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

    def warmup(self, thread):
        self.max_sqp_iter = 10  
        self.max_qp_iter  = 100   
        self.u0 = pin_utils.get_u_grav(self.q0, self.robot.model, np.zeros(self.robot.model.nq))
        self.solver.xs = [self.x0 for i in range(self.config['N_h']+1)]
        self.solver.us = [self.u0 for i in range(self.config['N_h'])]
        self.tau_ff, self.x_des, self.K, self.t_child, self.ddp_iter, self.cost, self.constraint_norm, self.gap_norm, self.qp_iters, self.KKT = solveOCP(self.joint_positions, 
                                                                                          self.joint_velocities, 
                                                                                          self.solver, 
                                                                                          self.max_sqp_iter, 
                                                                                          self.max_qp_iter, 
                                                                                          self.target_position,
                                                                                          self.lb,
                                                                                          self.ub,
                                                                                          self.TASK_PHASE)
        self.cumulative_cost += self.cost
        self.max_sqp_iter = self.config['maxiter']
        self.max_qp_iter  = self.config['max_qp_iter']

    def run(self, thread):        
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
        current_time = thread.ti
        current_state = np.concatenate([q, v])
        current_position = current_state[:3]
        desired_position = current_position + self.v_des[:3] * (self.Nh * self.dt_ocp)
        self.q_des[:3] = desired_position
        desired_state = np.concatenate([self.q0, self.v0])

        self.target_state[:] = desired_state
        # # # # # # #  
        # Solve OCP #
        # # # # # # # 
        self.tau_ff, self.x_des, self.K, self.t_child, self.ddp_iter, self.cost, self.constraint_norm, self.gap_norm, self.qp_iters, self.KKT = solveOCP(q, 
                                                                                          v, 
                                                                                          self.solver, 
                                                                                          self.max_sqp_iter, 
                                                                                          self.max_qp_iter, 
                                                                                          self.target_state,
                                                                                          self.lb,
                                                                                          self.ub,
                                                                                          self.TASK_PHASE)

        # # # # # # # # 
        # Send policy #
        # # # # # # # #
        self.tau = self.tau_ff.copy()

        # Compute gravity
        self.tau_gravity = pin.rnea(self.robot.model, self.robot.data, self.joint_positions, np.zeros(self.nv), np.zeros(self.nv))

        if(self.RUN_SIM == False):
            self.tau -= self.tau_gravity

        ###### DANGER SEND ONLY GRAV COMP
        # self.tau = np.zeros_like(self.tau_full)
        
        self.head.set_control('ctrl_joint_torques', self.tau)     


        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
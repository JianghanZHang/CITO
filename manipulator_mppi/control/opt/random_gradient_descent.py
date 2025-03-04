import mujoco 

import numpy as np  


class GradientDescent:
    def __init__(self, cost, rollout, sensor_datas, rollout_models, state_rollouts, horizon, num_workers, temperature=1):
        self.cost = cost
        self.rollout = rollout

        # Attributes used for rollout and cost functions
        self.sensor_datas = sensor_datas # (n_samples, horizon, sensor_dim)
        self.state_rollouts = state_rollouts # (n_samples, horizon, state_dim)
        self.rollout_models = rollout_models # mujoco models used for rollout
        self.horizon = horizon 
        self.num_workers = num_workers
        # Attributes used for utility function
        self.temperature = temperature

    def utility(self, costs, temperature):
        min_cost = np.min(costs)
        utility = np.exp((-1)/temperature * (costs - min_cost))
        return utility
    
    def estimate_utility_gradient(self, costs, perturbations):
        '''
        costs is the cost of each trajectory in the batch (#samples x 1)
        perturbations is the perturbation applied to each trajectory in the batch (#samples x horizon x control_dim)
        '''
        N = costs.shape[0]
        utilities = self.utility(costs, self.temperature) # utilities: N x 1
        sampled_gradients = utilities[:, None, None] * perturbations # sampled_gradients: N x horizon x control_dim
        utility_gradient = np.sum(sampled_gradients, axis=0) / N

        return utility_gradient
    
    # self.rollout_func(self.rollout_models, 
    #                   self.state_rollouts, 
    #                   actions, 
    #                   np.repeat(np.array([np.concatenate([[0], obs])]), self.n_samples, axis=0), 
    #                   self.sensor_datas, 
    #                   num_workers=self.num_workers, 
    #                   nstep=self.horizon)

    # costs_sum = self.cost_func(self.state_rollouts[:, :, 1:], actions, self.sensor_datas)


    def evaluate(self, ):
        '''
        This is the function evaluation step.
        It computes the cost of each trajectory in the batch
        '''
        rollout_states = self.rollout(self.rollout_models, self.state_rollouts, actions, sensor_datas, num_workers, nstep)
        costs = self.cost(rollout_states, actions, sensor_datas)
        return costs

    
    def optimize(self, init_x, )




import numpy as np
from numpy import matlib as ml
import matplotlib
import matplotlib.pyplot as plt

import model
import belief
import traj_opt
import plot
from Classes.vehicle import Vehicle 
import IPython
import math
import cPickle as pickle 

class CarModel(model.Model):
    def __init__(self):
        # model dimensions
        self.xDim = 4 # state space dimension
        self.uDim = 2 # control input dimension
        self.qDim = 0 # dynamics noise dimension
        self.zDim = 2 # observation dimension
        self.rDim = 2 # observtion noise dimension
        self.vehicle = Vehicle()
        #not doing belief space, so its the same
        self.bDim = self.xDim

        self.dT = 1. # time step for dynamics function
        self.T = 15 # Will Pass this in 

        self.alpha_belief = 10. # weighting factor for penalizing uncertainty at intermediate time steps
        self.alpha_final_belief = 10. # weighting factor for penalizing uncertainty at final time step
        self.alpha_control = 0.1 # weighting factor for penalizing control cost
        
        self.xMin = ml.vstack([-1e10,-1e10,-1e10,-1e10]) # minimum limits on state (xMin <= x)
        self.xMax = ml.vstack([1e10,1e10,1e10,1e10]) # maximum limits on state (x <= xMax)
        self.uMin = ml.vstack([-1e20,-1e20]) # minimum limits on control (uMin <= u)
        self.uMax = ml.vstack([1e20,1e20]) # maximum limits on control (u <= uMax)

        self.Q = ml.eye(self.qDim) # dynamics noise variance
        self.R = ml.eye(self.rDim) # observation noise variance

        self.start = ml.zeros([self.xDim,1]) # start state, OVERRIDE
        self.goal = ml.zeros([self.xDim,1]) # end state, OVERRIDE

        self.sqpParams = CarParams()

    def dynamics_func(self, x_t, u_t, q_t):
        return self.vehicle.dynamics(x_t,u_t)

    def obs_func(self, x_t, r_t):
        return x_t 

class CarParams(model.SqpParams):
    def __init__(self):
        self.improve_ratio_threshold = .1
        self.min_trust_box_size = 1e-3
        self.min_approx_improve = 1e-4
        self.max_iter = 50.
        self.trust_shrink_ratio = .1
        self.trust_expand_ratio = 1.5
        self.cnt_tolerance = 1e-4
        self.max_penalty_coeff_increases = 8.
        self.penalty_coeff_increase_ratio = 100.
        self.initial_trust_box_size = 10.
        self.initial_penalty_coeff = 50.

class OptimizeTraj():

    def __init__(self):
        self.model = CarModel()

    def trajOpt(self,X,U):
        X = ml.matrix(X)
        U = ml.matrix(U)
        self.model.T = np.max(X.shape)
        [Xopt, Uopt] = traj_opt.traj_opt_penalty_sqp(X, U, self.model)
        Xopt = np.array(Xopt)
        Uopt = np.array(Uopt)
        pickle.dump([Xopt,Uopt],open('trajectory.p','wb'))
        return Xopt,Uopt

    def loadTraj(self): 
        [Xopt,Uopt] = pickle.load(open('trajectory.p','rb'))
        return Xopt,Uopt


def test_bsp_light_dark():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-plotting',action='store_true',default=False)
    parser.add_argument('--profile',action='store_true',default=False)
    parser.add_argument('--gradient_free',action='store_true',default=False)
    args = parser.parse_args()

    plotting = not args.no_plotting
    profile = args.profile
    gradient_free = args.gradient_free
    model = LightDarkModel()

    X1 = ml.matrix([[-3.5,2],[-3.5,-2],[-4,0],[2,2],[-4,2]]).T
    G =  ml.matrix([[-3.5,-2],[-3.5,2],[-1,0],[2,-2],[-1,-2]]).T

    # [Reference] final belief trajectory costs for verification 
    # Allow for numerical inaccuracy
    verify_cost = [45.181701, 45.181643, 49.430339, 27.687003, 56.720314]

    for i_problem in xrange(0,5):
        # Setup initial conditions for problem
        x1 = X1[:,i_problem] # start (mean of initial belief)
        SqrtSigma1 = ml.eye(2) # initial covariance
        goal = G[:,i_problem] # goal
    
        model.setStartState(x1)
        model.setGoalState(goal)    
    
        # setup initial control vector -- straight line initialization from start to goal
        U = ml.tile(((model.goal - model.start)/(model.T-1)), (1,model.T-1))
    
        B = ml.zeros([model.bDim,model.T])
        B[:,0] = belief.compose_belief(x1, SqrtSigma1, model)
        for t in xrange(0,model.T-1):
            B[:,t+1] = belief.belief_dynamics(B[:,t], U[:,t], None, model,None,None)

        # display initialization
        if plotting:
            plot.plot_belief_trajectory(B, U, model)
    
        if gradient_free :
            [Bopt, Uopt] = belief_grad_free.STOMP_BSP(B,model,plotting,profile)
        else:
            [Bopt, Uopt] = belief_opt.belief_opt_penalty_sqp(B, U, model, plotting, profile)
        if plotting:
            plot.plot_belief_trajectory(Bopt, Uopt, model);
    
	# Forward simulated cost
        cost = belief.compute_forward_simulated_cost(B[:,0], Uopt, model)
        print('Total cost of optimized trajectory: %f' % cost)
    
        # save trajectory to png file 
        # saveas(gcf, sprintf('bsp-light-dark-plan-%i.png',i_problem));
    
        print('For verification (allow for numerical inaccuracy):')
        print('(Reference) Total cost of optimized trajectory: %2.6f' % verify_cost[i_problem])
    
        # Simulate execution of trajectory (test to see why the maximum likelihood observation assumption does not hold)
        #simulate_bsp_trajectory(B(:,1), Uopt, model);
    
        print('press enter to continue to the next problem')
        raw_input()
    

if __name__ == '__main__':
    test_bsp_light_dark()

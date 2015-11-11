import numpy as np
from numpy import matlib as ml
import cvxpy

from math_util import numerical_jac

import IPython

def ekf(x_t, Sigma_t, u_t, z_tp1, model):
# Extract function handles and useful definitions from model
    dynamics_func = model.dynamics_func
    obs_func = model.obs_func
    xDim = model.xDim
    qDim = model.qDim
    rDim = model.rDim
    Q = model.Q
    R = model.R

    dyn_varargin = [x_t, u_t, ml.zeros([qDim,1])]
    obs_varargin = [dynamics_func(x_t, u_t, ml.zeros([qDim,1])), ml.zeros([rDim,1])]

    # dynamics state jacobian
    A = numerical_jac(dynamics_func, 0, dyn_varargin)

   #print "A",A
    # dynamics noise jacobian
    M = numerical_jac(dynamics_func, 2, dyn_varargin)

    #print "M",M
    # observation state jacobian
    H = numerical_jac(obs_func, 0, obs_varargin)

    #print "H",H
    # observation noise jacobian
    N = numerical_jac(obs_func, 1, obs_varargin)
    #print "N",N
    Sigma_tp1_neg = A*Sigma_t*A.T + M*Q*M.T
    K = Sigma_tp1_neg*H.T*ml.linalg.inv(H*Sigma_tp1_neg*H.T + N*R*N.T)

    x = dynamics_func(x_t, u_t, ml.zeros([qDim,1]))
    x_tp1 = x + K*(z_tp1 - obs_func(x, ml.zeros([rDim,1])))
    Sigma_tp1 = (ml.eye(xDim) - K*H)*Sigma_tp1_neg

    return x_tp1, Sigma_tp1

# Belief dynamics: Given belief and control at time t, compute the belief
# at time (t+1) using EKF
def belief_dynamics(b_t, u_t, z_tp1, model,profiler=None,profile=False):
    dynamics_func = model.dynamics_func
    b_t = np.array(b_t)
    u_t = np.array(u_t)
    x_tp1 = dynamics_func(b_t,u_t,0)
    x_tp1 = ml.matrix(x_tp1)
    return x_tp1

# Compose belief vector as [state; square root of covariance matrix]
# Since the principal square root of the covariance matrix is symmetric, we only store the 
# lower (or upper) triangular portion of the square root to eliminate redundancy 
def compose_belief(x, SqrtSigma, model):
    xDim = model.xDim
    bDim = model.bDim

    b = ml.zeros([bDim,1])
    b[0:xDim] = x
    idx = xDim
    for j in xrange(0,xDim):
        for i in xrange(j,xDim):
            b[idx] = 0.5 * (SqrtSigma.item(i,j) + SqrtSigma.item(j,i))
            idx = idx+1

    return b

# Decompose belief vector into mean and square root of covariance
def decompose_belief(b, model):

    xDim = model.xDim

    x = b[0:xDim]
    idx = xDim
    
    SqrtSigma = ml.zeros([xDim, xDim])

    for j in xrange(0,xDim):
        for i in xrange(j,xDim):
            SqrtSigma[i,j] = b[idx]
            SqrtSigma[j,i] = SqrtSigma[i,j]
            idx = idx+1

    return x, SqrtSigma

def cvxpy_sigma_trace(b, model):
    xDim = model.xDim

    SqrtSigma = np.zeros([xDim,xDim]).tolist()

    idx = xDim
    for j in xrange(0,xDim):
        for i in xrange(j,xDim):
            SqrtSigma[i][j] = b[idx]
            SqrtSigma[j][i] = SqrtSigma[i][j]
            idx = idx+1
    
    trace = 0
    for i in xrange(0,xDim):
        trace += sum(cvxpy.square(SqrtSigma[i][j]) for j in xrange(0,xDim))

    return trace

# Compute foward simulated cost of belief trajectory given initial belief and set of control inputs (integrated forward)
def compute_forward_simulated_cost(b, U, model):
 
    T = model.T
    cost = 0
    
    b_t = b

    for t in xrange(0,T-1):
        x_t, s_t = decompose_belief(b_t, model)
        cost += model.alpha_belief*ml.trace(s_t*s_t)
        cost += model.alpha_control*ml.sum(U[:,t].T*U[:,t])
        b_t = belief_dynamics(b_t, U[:,t], None, model)
    
    x_T, s_T = decompose_belief(b_t, model)
    cost += model.alpha_final_belief*ml.trace(s_T*s_T)

    return cost

if __name__ == '__main__':
    import model
    model = model.Model()
    model.xDim = 4
    x = np.matrix(np.random.rand(4,1))
    svec = np.matrix([0,1,2,3,4,5,6,7,8,9]).T
    b = np.vstack((x,svec))
    x, s, constraints = cvxpy_decompose_belief(b,model)

    x_actual, s_actual = decompose_belief(b,model)

    IPython.embed()

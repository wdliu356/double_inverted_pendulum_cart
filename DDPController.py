import pybullet as p
import pybullet_data as pd
from base_env import BaseEnv
import gym

import autograd.numpy as np
from autograd import grad
from autograd import jacobian
from autograd.numpy import sin, cos

# Programmer: Junkai Zhang, April 8th 2023
# Purpose: This program implements the Differential Dynamic Programming (DDP) 
#          algorithm based on the autograd package.
    

class DDPcontroller:
    # Finite horizon Discrete-time Differential Dynamic Programming(DDP)

    def __init__(
        self, 
        dynamics,
        cost,
        tolerance = 1e-5,
        max_iter = 100,
        T = 100,
        state_dim = 6,
        control_dim = 1,

    ):
        self.dynamics = dynamics
        self.cost = cost
        self.running_cost = cost.running_cost
        self.terminal_cost = cost.terminal_cost
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.T = T # Total time steps, actions should be T - 1 times

    def command(self, state):
        # DDP algorithm
        # state: current state with shape (state_dim, )
        # return: control action with shape (control_dim, )

        # Define functions and derivatives needed
        # Define V, Vx, Vxx functions
        V = self.terminal_cost
        Vx = grad(V)
        Vxx = jacobian(Vx)

        # Define L, Lx, Lxx, Lu, Luu, Lxu functions
        L = self.running_cost
        Lx = grad(L, 0)
        Lu = grad(L, 1)
        Lxx = jacobian(Lx, 0)
        Luu = jacobian(Lu, 1)
        Lxu = jacobian(Lx, 1)
        Lux = jacobian(Lu, 0)

        # define F, Fx, Fu functions
        F = self.dynamics
        Fx = jacobian(F, 0)
        Fu = jacobian(F, 1)


        # Initialize the state
        x_init = state
        # Initialize the control with random values
        U = np.random.rand(self.T - 1, self.control_dim)
        
        # Initialize the trajectory
        # TODO: Use rollout function to initialize the trajectory
        X = self._rollout(x_init, U)


        # Initialize the cost
        prev_cost = self._compute_total_cost(X, U)
        print(prev_cost)

        for i in range(self.max_iter):

            # Backward pass
            # Initialize the cost-to-go
            Vx_val = Vx(X[-1])
            Vxx_val = Vxx(X[-1])
            
            for i in range(self.T - 2, -1, -1):
                # Compute the cost-to-go
                # TODO: Add miu_1 and miu_2 adjustment
                Qx = Lx(X[i], U[i]) + Fx(X[i], U[i]).T @ Vx_val
                Qu = Lu(X[i], U[i]) + Fu(X[i], U[i]).T @ Vx_val

                miu_1 = 0
                miu_2 = 0
                eye_x = np.eye(self.state_dim)
                eye_u = np.eye(self.control_dim)
                Qxx = Lxx(X[i], U[i]) + Fx(X[i], U[i]).T @ (Vxx_val + miu_1 * eye_x) @ Fx(X[i], U[i])
                # Qux = Lxu(X[i], U[i]) + Fu(X[i], U[i]).T @ Vxx_val @ Fx(X[i], U[i])
                Qux = Lux(X[i], U[i]) + Fu(X[i], U[i]).T @ (Vxx_val + miu_1 * eye_x) @ Fx(X[i], U[i])
                Quu = Luu(X[i], U[i]) + Fu(X[i], U[i]).T @ (Vxx_val + miu_1 * eye_x) @ Fu(X[i], U[i]) + miu_2 * eye_u
                
                # Compute the control gain
                k = -np.linalg.inv(Quu) @ Qu
                K = -np.linalg.inv(Quu) @ Qux
                # Update the cost-to-go
                Vx_val = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
                Vxx_val = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            
            fx_val = Fx(X[-1], U[-1])


    def _compute_dynamics(self,state,control):
        return self.dynamics(state,control)
    
    def _compute_total_cost(self,state,control):
        # Compute the total cost of the trajectory
        # state: current state
        # control: current control
        # return: total cost
        total_cost = 0.
        for i in range(self.T - 1):
            total_cost += self.cost.running_cost(state[i], control[i])
        total_cost += self.terminal_cost(state[-1])
        return total_cost
    
    def _rollout(self, init_state, controls):
        # Rollout the trajectory
        # state: current state
        # control: current control
        # return: trajectory
        states = [init_state]
        for i in range(controls.shape[0]):
            states.append(self._compute_dynamics(states[-1], controls[i]))
        
        return np.array(states)
        
    


    
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd.numpy import sin, cos

def cartpole_dynamics_analytic(x, u):
    
    return x


class CartpoleCost:
    def __init__(self, x_final, terminal_scale, Q, R):
        self.x_final = x_final
        self.terminal_scale = terminal_scale
        self.Q = Q
        self.R = R

    def running_cost(self, x, u):
        Q = self.Q
        R = self.R
        dx = self.x_delta(self.x_final, x)
        u = u[np.newaxis]
        return np.squeeze(dx.T @ Q @ dx + u.T @ R @ u)

    def terminal_cost(self, x):
        Q = self.Q
        dx = self.x_delta(self.x_final, x)
        return self.terminal_scale * dx @ Q @ dx
    @staticmethod
    def x_delta(x1, x2):
        dx = x1 - x2
        d_theta = np.mod(dx[2] + np.pi, 2 * np.pi) - np.pi
        # d_theta = np.arccos(cos(dx[2])
        return np.array([dx[0], dx[1], d_theta, dx[3]])
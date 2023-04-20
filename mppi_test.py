import torch
import numpy as np
import pybullet as p
import pybullet_data as pd
from base_env import BaseEnv
import gym
from mppi import MPPI
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from IPython import display
from tqdm import tqdm
from celluloid import Camera
import os
import imageio

# %load_ext autoreload
# %autoreload 2

from double_pendulum_system import *
controller = DoublePendulumControl(dynamics=dynamics_analytic,cost_function=cost_function,horizon = 17)
initial_state = torch.from_numpy(np.random.randn(6))
initial_state = torch.tensor([0,0,np.pi,0,np.pi,0])
state = initial_state
target = torch.tensor([0,0,0,0,0,0])
num_steps = 100
pbar = tqdm(range(num_steps))

# if not os.path.exists('plots'):
#     os.makedirs('plots')

fig, ax = plt.subplots()
for i in pbar:
    
    action = controller.control_calcu(state)
    # print(action)
    
    # action -= K*state[0]

    state = dynamics_analytic(state,action)
    state = state.squeeze()
    # print(state)
    error_i = np.linalg.norm((state-target)@torch.diag(torch.tensor([0.1, 0.1, 1, 0.1, 1, 0.1])))
    pbar.set_description(f'Goal Error: {error_i:.4f}')

    # --- Start plotting

    ax = plt.axes(xlim=(state[0]-10, state[0]+10), ylim=(-2, 2))
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Double Pendulum at t={:.2f}'.format(i*0.05))
    x = state[0]
    theta1 = state[2]
    theta2 = state[4]
    L1 = 0.5
    L2 = 0.5
    x1 = x + L1*torch.sin(theta1)
    y1 = L1*torch.cos(theta1)
    x2 = x1 + L2*torch.sin(theta2)
    y2 = y1 + L2*torch.cos(theta2)
    plt.plot([x,x1],[0,y1],color='black')   
    plt.plot([x1,x2],[y1,y2],color='black')
    plt.draw()
    plt.pause(1e-17)
    # time.sleep(0.025)
    plt.clf()
    
    
    if error_i < 0.4:
        break
    # --- End plotting
plt.show()
plt.close()

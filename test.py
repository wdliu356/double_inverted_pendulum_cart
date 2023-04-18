import os
import matplotlib.pyplot as plt
import numpy as np
import imageio

# Create a directory to store the plot images
if not os.path.exists('plots'):
    os.makedirs('plots')

# Generate some data to plot
x = np.linspace(0, np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Plot the data and save the images
for i in range(len(x)):
    fig, ax = plt.subplots()
    ax.plot(y1[i], y2[i], label='Sin')
    # ax.plot(x, y2, label='Cos')
    ax.set_title('Plots at x={:.2f}'.format(x[i]))
    ax.legend()
    filename = os.path.join('plots', 'plot_{:03d}.png'.format(i))
    plt.savefig(filename)
    plt.close()

# Create the GIF animation
images = []
for i in range(len(x)):
    filename = os.path.join('plots', 'plot_{:03d}.png'.format(i))
    images.append(imageio.imread(filename))
imageio.mimsave('plots.gif', images, duration=0.1)
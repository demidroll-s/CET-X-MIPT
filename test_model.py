import model as md
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage[utf8]{inputenc}',
            r'\usepackage[russian]{babel}',
            r'\usepackage{amsmath}', 
            r'\usepackage{siunitx}']

c = 1403
p_0 = 50.0
rho = 1000.0
a = 2.0 

t = np.linspace(0, 1.0, 201)
x = np.linspace(0, 1000.0, 81)

pressure = np.zeros((len(t), len(x)))
velocity = np.zeros((len(t), len(x)))

for n in range(len(t)):
    for l in range(len(x)):
        pressure[n][l] = md.pressure_case_a(t[n], x[l], c, a, rho, p_0)
        velocity[n][l] = md.velocity_case_a(t[n], x[l], c, a, rho, p_0) * 10**5

def plot_result(t, x, magnitude, title, save_filename):
     _, ax = plt.subplots(figsize=(16,8), dpi=300)
     plt.xticks(color='k', size=16)
     plt.yticks(color='k', size=16)
     im = ax.contourf(x, t, magnitude, extend='both')
     divider = make_axes_locatable(ax)
     cax = divider.append_axes('right', size='3%', pad=0.05)
     plt.colorbar(im, cax=cax, orientation='vertical')
     plt.xticks(color='k', size=16)
     plt.yticks(color='k', size=16)
     ax.set_xlabel(r'Coordinate $x$, m', fontsize=20)
     ax.set_ylabel(r'Time $t$, s', fontsize=20)
     ax.set_title(title, fontsize=20)
     plt.savefig(save_filename, dpi=300)

contourlevels = np.arange(0, 1.0, 0.05)
plot_result(t, x, pressure, r'Pressure $p$, atm', 'Results/Computational model/pressure-analytical.png')
plot_result(t, x, velocity, r'Velocity $w$, m/s', 'Results/Computational model/velocity-analytical.png')
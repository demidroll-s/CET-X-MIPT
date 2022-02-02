import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize.nonlin import nonlin_solve
from scipy.optimize import curve_fit
import time as tm

import os,sys,inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import signal_processing as sp

n = 1000
time_interval = 1.0
noise_level = 0.1
reference_point = 0.5
width = 0.0

time, signal = sp.signal_construction_width(n, time_interval, reference_point, noise_type='gaussian', noise_level=noise_level, width=width)
time_0, signal_0 = sp.signal_construction_width(n, time_interval, reference_point, noise_type='gaussian', noise_level=0.0, width=width)

fig, ax = plt.subplots(figsize=(16,8), dpi=300)
sp.plot_signal(ax, time_0, signal_0, plot_color='blue', plot_alpha=1.0, plot_label='signal with noise', reference_point=None, rp_color=None)
sp.plot_signal(ax, time, signal, plot_color='red', plot_alpha=0.7, plot_label='pure signal', reference_point=reference_point, rp_color=None)
plt.savefig('images/signal.png', dpi=300)

"""
level = 4
wavelet = 'db4'
fig, axarr = plt.subplots(nrows=level, ncols=2, figsize=(10,10), dpi=300)
sp.plot_decomposition(axarr, signal_0, level, wavelet)

plt.savefig('DWT_Leakage detection/images/decomposition.png', dpi=300)

fig, ax = plt.subplots(figsize=(16,8), dpi=300)
sp.test_threshold(ax, n=50, thresh=0.4)
plt.savefig('images_1/test_threshold.png', dpi=300)

thresh1 = 0.10
thresh2 = 0.63
rec_signal1 = sp.lowpassfilter(signal_0, thresh=0.10, level=level)
rec_signal2 = sp.lowpassfilter(signal_0, thresh=0.63, level=level)
fig, ax = plt.subplots(figsize=(16,8), dpi=300)
sp.plot_signal(ax, time_0, signal_0, plot_color='red', plot_alpha=0.4, plot_label='signal with noise')
sp.plot_signal(ax, time_0, rec_signal1, plot_color='green', plot_alpha=0.8, plot_label='reconstructed signal, $\sigma = ' + '{:.2f}'.format(thresh1)  + '$')
sp.plot_signal(ax, time_0, rec_signal2, plot_color='blue', plot_alpha=1.0, plot_label='reconstructed signal, $\sigma = ' + '{:.2f}'.format(thresh2)  + '$', reference_point=0.5, rp_color=None)
plt.savefig('images_1/lowpassfilter_1.png', dpi=300)

fig, ax = plt.subplots(figsize=(16,8), dpi=300)
sp.plot_signal(ax, time_0, rec_signal1, plot_color='green', plot_alpha=0.5, plot_label='reconstructed signal, $\sigma = ' + '{:.2f}'.format(thresh1)  + '$')
sp.plot_signal(ax, time_0, rec_signal2, plot_color='blue', plot_alpha=1.0, plot_label='reconstructed signal, $\sigma = ' + '{:.2f}'.format(thresh2)  + '$', reference_point=0.5, rp_color=None)
plt.savefig('images_1/lowpassfilter_2.png', dpi=300)
"""
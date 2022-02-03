import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
sp.plot_signal(ax, time_0, signal_0, plot_color='blue', plot_alpha=1.0, plot_label='pure signal', reference_point=None, rp_color=None)
sp.plot_signal(ax, time, signal, plot_color='red', plot_alpha=0.7, plot_label='signal with noise', reference_point=reference_point, rp_color=None)
plt.savefig('Results/signal.png', dpi=300)

level = 4
wavelet = 'db4'
fig, axarr = plt.subplots(nrows=level, ncols=2, figsize=(10,10), dpi=300)
sp.plot_decomposition(axarr, signal, level, wavelet)

plt.savefig('Results/decomposition.png', dpi=300)

thresh1 = 0.10
thresh2 = 0.63
rec_signal1 = sp.lowpassfilter(signal, thresh=0.10, level=level)
rec_signal2 = sp.lowpassfilter(signal, thresh=0.63, level=level)

fig, ax = plt.subplots(figsize=(16,8), dpi=300)
sp.plot_signal(ax, time, signal, plot_color='red', plot_alpha=0.4, plot_label='signal with noise')
sp.plot_signal(ax, time, rec_signal1, plot_color='green', plot_alpha=0.8, plot_label='reconstructed signal, $\sigma = ' + '{:.2f}'.format(thresh1)  + '$')
sp.plot_signal(ax, time, rec_signal2, plot_color='blue', plot_alpha=1.0, plot_label='reconstructed signal, $\sigma = ' + '{:.2f}'.format(thresh2)  + '$', reference_point=0.5, rp_color=None)
plt.savefig('Results/lowpassfilter_1.png', dpi=300)

fig, ax = plt.subplots(figsize=(16,8), dpi=300)
sp.plot_signal(ax, time, rec_signal1, plot_color='green', plot_alpha=0.5, plot_label='reconstructed signal, $\sigma = ' + '{:.2f}'.format(thresh1)  + '$')
sp.plot_signal(ax, time, rec_signal2, plot_color='blue', plot_alpha=1.0, plot_label='reconstructed signal, $\sigma = ' + '{:.2f}'.format(thresh2)  + '$', reference_point=0.5, rp_color=None)
plt.savefig('Results/lowpassfilter_2.png', dpi=300)

fig = plt.figure(figsize=(12 ,12), dpi=300)
spec = gridspec.GridSpec(ncols=20, nrows=8)
top_ax = fig.add_subplot(spec[0:2, 0:18])
bottom_ax = fig.add_subplot(spec[2:, 0:19])

sp.plot_signal(top_ax, time, signal, plot_color='red', plot_alpha=0.7, plot_label='signal', reference_point=reference_point, legend_size='large')

scales = np.arange(1, 512, 1.0)

time, scale, coefficients = sp.plot_wavelet_transform(top_ax, bottom_ax, time, signal, scales, waveletname='gaus2', reference_point=reference_point, rp_color=None, legend_size='large')
plt.savefig('Results/wavelet_transform.png', dpi=300)

fig, ax = plt.subplots(figsize=(12,12), dpi=300)

sp.maximum_of_wavelet_transform_modulus(ax, time, scale, coefficients, reference_point=reference_point, rp_color=None, legend_size='large')
plt.savefig('Results/wavelet_maximum_modulus.png', dpi=300)

fig = plt.figure(figsize=(12 ,12), dpi=300)
spec = gridspec.GridSpec(ncols=20, nrows=8)
top_ax = fig.add_subplot(spec[0:2, 0:18])
bottom_ax = fig.add_subplot(spec[2:, 0:19])

sp.plot_signal(top_ax, time, signal, plot_color='red', plot_alpha=0.7, plot_label='signal', reference_point=reference_point, legend_size='large')

scales = np.arange(1, 512, 1.0)

time, scale, coefficients = sp.plot_wavelet_transform(top_ax, bottom_ax, time, signal, scales, waveletname='gaus1', reference_point=reference_point, rp_color=None, legend_size='large')

gutter_result = sp.gutter_method(time, scale, power=abs(coefficients), lower_scale=1.0, upper_scale=2.0)
(reference_point_exp, variance) = gutter_result

distance = 0.004

bottom_ax.axvline(x=reference_point_exp + distance, linestyle='dashdot', color='maroon', label='expected reference point')
sp.gutter_method_rectangle_plot(bottom_ax, gutter_result=gutter_result, lower_scale=1.0, upper_scale=2.0)

bottom_ax.legend(fontsize='large')
plt.savefig('Results/gutter_method.png', dpi=300)
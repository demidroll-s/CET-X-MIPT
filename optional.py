from os import error
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pywt
from scipy.optimize import curve_fit

import signal_processing as sp

import multiprocessing as mp

def test_threshold(ax, n, thresh):
    num = np.linspace(0, n-1, n)
    up_level=[thresh for i in range(n)]
    down_level=[-thresh for i in range(n)]
    sequence = [2 * random.random() - 1 for i in range(n)]
    sequence_soft = pywt.threshold(sequence, value=thresh, mode='soft')
    sequence_hard = pywt.threshold(sequence, value=thresh, mode='hard')
    
    ax.plot(num, sequence, label=r'sequence before threshold procedure', color='maroon', alpha=1.0)
    ax.plot(num, sequence_soft, label=r'soft mode', color='fuchsia', alpha=1.0)
    ax.plot(num, sequence_hard, label=r'hard mode', color='orange', alpha=1.0)
    ax.plot(num, up_level, ls='dashed', color='gray')
    ax.plot(num, down_level, ls='dashed', color='gray')
    ax.set_xlabel(r'$n$', fontsize=20)
    plt.xlabel(r'Test sequence', fontsize=20)
    plt.ylabel(r'Sequence after threshold', fontsize=20)
    plt.xticks(color='k', size=16)
    plt.yticks(color='k', size=16)
    plt.legend(fontsize='large')

def generate_signal_sequence(m, n, reference_point, noise_type, noise_level):
    signal_sequence = [0 for i in range(m)]
    for i in range(len(signal_sequence)):
        signal_sequence[i] = [0 for j in range(2)]
    for i in range(m):
        time, signal = sp.signal_construction(n, reference_point, noise_type, noise_level)
        signal_sequence[i][0] = time
        signal_sequence[i][1] = signal
    return signal_sequence   

def dependence(sequence, reference_point=0.5, wavelet='db4'):
    m = len(sequence)
    num_level = 5
    thresh = []
    for j in range(250):
        thresh.append(j * 0.0025)
    norm = [0 for _ in range(num_level)]
    for l in range(num_level):
        norm[l] = [0 for _ in range(len(thresh))]
        for i in range(len(sequence)):
            time = sequence[i][0]
            signal = sequence[i][1]
            for j in range(len(thresh)):
                reconstructed_signal = sp.lowpassfilter(signal, thresh=thresh[j], level = l + 1, wavelet=wavelet)
                diff = [0 for _ in range(len(time))]
                for k in range(len(time)):
                    if time[k] < reference_point:
                        diff[k] = abs(reconstructed_signal[k] - 1.0)
                    else:
                        diff[k] = abs(reconstructed_signal[k])
                x = np.cumsum(diff)
                norm[l][j] = norm[l][j] + x[-1] * (time[1] - time[0])/m
    return thresh, norm 

def multiprocessing_dependence(sequence):
    part = len(sequence) // mp.cpu_count()
    iterable_sequence = []
    for i in range(mp.cpu_count()):
        frame = sequence[(part * i):(part * (i + 1))]
        iterable_sequence.append(frame)
    with mp.Pool(mp.cpu_count()) as p:
        result = p.map(dependence, iterable_sequence)
    thresh = result[0][0]
    num_level = len(result[0][1])
    for i in range(len(result)):
        norm  = [0 for l in range(num_level)]
        for l in range(num_level):
            norm[l] = [0 for j in range(len(thresh))]
            for j in range(len(thresh)):
                norm[l][j] += result[i][1][l][j]
    return (thresh, norm)

def optimal_thresh(m, n, reference_point, noise_type, wavelet='db4'):
    noise_level = []
    for j in range(15):
         noise_level.append(j * 0.02)
    optimal_thresh = [0 for l in range(5)]
    for l in range(5):
        optimal_thresh[l] = [0 for j in range(len(noise_level))]

    for j in range(len(noise_level)):
        (thresh, norm, q) = dependence(m, n, reference_point, noise_type, noise_level[j], wavelet)
        for l in range(5):
            optimal_thresh[l][j] = q[l]
        print(j)
    return (noise_level, optimal_thresh)

def plot_optimal_tresh_file(ax, colors, filename):
    with open(filename) as f:
        data = []
        for line in f:
            data.append([float(x) for x in line.split()])

    noise_level = []
    optimal_thresh = [[] for l in range(5)]
    for i in range(len(data)):
        noise_level.append(data[i][0])
        for j in range(1, len(data[i])):
            optimal_thresh[j-1].append(data[i][j])

    def f(x, a, b):
        return a * x + b

    for l in range(len(optimal_thresh)):
        popt, pcov = curve_fit(f, noise_level[0:6], optimal_thresh[l][0:6])
        plt.scatter(noise_level, optimal_thresh[l], color=colors[l], label='level = ' + '{}'.format(l + 1))
        plt.plot(np.array(noise_level), f(np.array(noise_level), *popt), color=colors[l], ls='dashed') 

    ax.set_xlim([0, 0.3])
    ax.axvline(x=0.1, linestyle='dashed', color='gray')
    ax.set_xlabel(r'Noise level $\chi$', fontsize=20)
    ax.set_ylabel(r'Optimal threshold parameter $\sigma$', fontsize=20)
    plt.xticks(color='k', size = 18)
    plt.yticks(color='k', size = 18)

    rect_ax = inset_axes(ax, '100%', '100%', loc='lower left', borderpad=0)
    rect_ax.axis('off')
    left, bottom, width, height = (0.0, 0.0, 0.333, 1.0)
    rect_approx = plt.Rectangle((left, bottom), width, height, facecolor='pink', alpha=0.3, transform=rect_ax.transAxes, label='область применимости аппроксимации' )
    rect_ax.add_patch(rect_approx)
    left, bottom, width, height = (0.333, 0.0, 0.667, 1.0)
    rect_non_approx = plt.Rectangle((left, bottom), width, height, facecolor='green', alpha=0.1, transform=rect_ax.transAxes, label='область неприменимости аппроксимации')
    rect_ax.add_patch(rect_non_approx)
    rect_ax.legend(fontsize='xx-large', loc='lower right', facecolor='white')

    ax.legend(fontsize='xx-large', facecolor='white')

def plot_optimal_tresh_list(ax, colors, noise_level, optimal_thresh):
    def f(x, a, b):
        return a * x + b

    for l in range(len(optimal_thresh)):
        popt, pcov = curve_fit(f, noise_level[0:6], optimal_thresh[l][0:6])
        plt.scatter(noise_level, optimal_thresh[l], color=colors[l], label='level = ' + '{}'.format(l + 1))
        plt.plot(np.array(noise_level), f(np.array(noise_level), *popt), color=colors[l], ls='dashed') 

    ax.set_xlim([0, 0.3])
    ax.axvline(x=0.1, linestyle='dashed', color='gray')
    ax.set_xlabel(r'Noise level $\chi$', fontsize=20)
    ax.set_ylabel(r'Optimal threshold parameter $\sigma$', fontsize=20)
    plt.xticks(color='k', size=18)
    plt.yticks(color='k', size=18)

    rect_ax = inset_axes(ax, '100%', '100%', loc='lower left', borderpad=0)
    rect_ax.axis('off')
    left, bottom, width, height = (0.0, 0.0, 0.333, 1.0)
    rect_approx = plt.Rectangle((left, bottom), width, height, facecolor='pink', alpha=0.3, transform=rect_ax.transAxes, label='область применимости аппроксимации' )
    rect_ax.add_patch(rect_approx)
    left, bottom, width, height = (0.333, 0.0, 0.667, 1.0)
    rect_non_approx = plt.Rectangle((left, bottom), width, height, facecolor='green', alpha=0.1, transform=rect_ax.transAxes, label='область неприменимости аппроксимации')
    rect_ax.add_patch(rect_non_approx)
    rect_ax.legend(fontsize='xx-large', loc='lower right', facecolor='white')

    ax.legend(fontsize='xx-large', facecolor='white')

def MFD(time, signal, reference_point):
    diff = []
    for i in range(1, len(signal)):
        diff.append(abs(signal[i] - signal[i-1]))
    MDF_index = diff.index(max(diff))
    reference_point_exp = time[MDF_index]
    delta = abs(reference_point - reference_point_exp)
    return delta

def test_MDF(m, n, noise_type, noise_level, thresh, level, wavelet, eps):
    k = 0
    for _ in range(m):
        reference_point = random.random()
        time, signal = sp.signal_construction(n, reference_point, noise_type, noise_level)
        rec_signal = sp.lowpassfilter(signal, thresh, level, wavelet)
        delta = MFD(time, rec_signal, reference_point)
        if delta < eps:
            k = k + 1
    return k/m
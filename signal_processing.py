from os import error
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pywt
from scipy.optimize import curve_fit

import multiprocessing as mp
import time as tm
import operator

#plt.rc('text', usetex = True)
#plt.rcParams['text.latex.preamble'] = [r'\usepackage[utf8]{inputenc}',
#            r'\usepackage[russian]{babel}',
#            r'\usepackage{amsmath}', 
#            r'\usepackage{siunitx}']

def signal_construction(n, time_interval, reference_point, noise_type, noise_level, dtype='float64'):
    """
    Construction of discrete signal from sensor 
    that registers the negative pressure wave.
    
    Parameters
    ----------
    n : int
        Length of discrete signal.
    time_interval : double
        The time interval at which the signal is defined.
    reference_point : double
        The moment of signal arrival.
    noise_type : str
        Type of noise - 'gaussian' or 'uniform'
    noise_level : double
        Noise level in constructed signal.
        
    Returns
    -------
    time : array_like
        Time grid on which discrete signal is defined.
    signal : array_like
        Constructed signal.
    """
    if not isinstance(n, int):
        raise TypeError('n should be integer')
    if n <= 1:
        raise ValueError('Wrong value of n: n should be higher than 1')

    if not isinstance(time_interval, float):
        raise TypeError('time_interval should be float')
    if time_interval <= 0:
        raise ValueError('Wrong value of time_interval: time_interval should be higher than 0')
    
    time_step = time_interval / n 
    time = [i * time_step for i in range(n + 1)]
    signal = [0.0 for _ in range(n + 1)]

    if not isinstance(reference_point, float):
        raise TypeError('reference_point should be float')
    if reference_point < 0.0 or reference_point > time_interval:
        raise ValueError('Wrong value of t_0: t_0 should be higher than 0.0 and lower than time_interval')

    if not isinstance(noise_level, float):
        raise TypeError('noise_level should be float')
    if noise_level < 0.0:
        raise ValueError('Wrong value of noise_level: noise_level should be higher than 0.0')

    for i in range(len(signal)):
        if time[i] < reference_point:
            signal[i] = 1.0
        else:
            signal[i] = 0.0
    for i in range(len(signal)):
        if noise_type == 'uniform':
            signal[i] = signal[i] + noise_level * (2 * random.random() - 1)
        elif noise_type == 'gaussian':
            signal[i] = signal[i] + random.gauss(0, noise_level)
        else:
            raise ValueError('Wrong value of noise_type: noise_type should be gaussian or uniform')

    time, signal = np.array(time, dtype=dtype), np.array(signal, dtype=dtype)

    return time, signal

def signal_construction_width(n, time_interval, reference_point, width, noise_type, noise_level, dtype='float64'):
    """
    Construction of discrete signal with a width of pressure drop from sensor 
    that registers the negative pressure wave.
    
    Parameters
    ----------
    n : int
        Length of discrete signal.
    time_interval : double
        The time interval at which the signal is defined.
    reference_point : double
        The moment of signal arrival.
    width : double
        Characteristic period of pressure drop.
    noise_type : str
        Type of noise - 'gaussian' or 'uniform'
    noise_level : double
        Noise level in constructed signal.
      
    Returns
    -------
    time : array_like
        Time grid on which discrete signal is defined.
    signal : array_like
        Constructed signal.
    """
    if not isinstance(n, int):
        raise TypeError('n should be integer')
    if n <= 1:
        raise ValueError('Wrong value of n: n should be higher than 1')

    if not isinstance(time_interval, float):
        raise TypeError('time_interval should be float')
    if time_interval <= 0:
        raise ValueError('Wrong value of time_interval: time_interval should be higher than 0')

    time_step = time_interval / n 
    time = [i * time_step for i in range(n + 1)]
    signal = [0.0 for i in range(n + 1)]

    if not isinstance(reference_point, float):
        raise TypeError('reference_point should be float')
    if reference_point < 0.0 or reference_point > time_interval:
        raise ValueError('Wrong value of t_0: t_0 should be higher than 0.0 and lower than time_interval')

    if not isinstance(width, float):
        raise TypeError('width should be float')
    if width < 0.0 or width > time_interval:
        raise ValueError('Wrong value of width: width should be higher than 0.0 and lower than time_interval')

    if not isinstance(noise_level, float):
        raise TypeError('noise_level should be float')
    if noise_level < 0.0:
        raise ValueError('Wrong value of noise_level: noise_level should be higher than 0.0')

    if width == 0:
        time, signal = signal_construction(n, time_interval, reference_point, noise_type, noise_level, dtype)
        return time, signal
    
    for i in range(len(signal)):
        if time[i] < reference_point:
            signal[i] = 1.0
        else:
            if time[i] > reference_point + width:
                signal[i] = 0.0
            else:
                signal[i] = 1.0 - (time[i] - reference_point) / width
    for i in range(len(signal)):
        if noise_type == 'uniform':
            signal[i] = signal[i] + noise_level * (2 * random.random() - 1)
        elif noise_type == 'gaussian':
            signal[i] = signal[i] + random.gauss(0, noise_level)
        else:
            raise ValueError('Wrong value of noise_type: noise_type should be gaussian or uniform')

    time, signal = np.array(time, dtype=dtype), np.array(signal, dtype=dtype)

    return time, signal

def plot_signal(ax, time, signal, plot_color='red', plot_alpha=1.0, plot_label='signal', reference_point=None, rp_color=None, legend_size='xx-large', dtype='float64'):
    """
    Plot signal with noise.

    Parameters
    ----------
    ax : matplotlib.axes class
        Variable of matplotlib Axes class, referring to the plot for signal.
    time : array_like
        Time grid on which discrete signal is defined.
    signal : array_like
        Constructed signal.
    .......
    """
    time, signal = np.array(time, dtype=dtype), np.array(signal, dtype=dtype)
    if time.ndim != 1:
        raise TypeError('time is not one-dimensional array')
    if signal.ndim != 1:
        raise TypeError('signal is not one-dimensional array')
    if time.shape != signal.shape:
        raise ValueError('time and signal lengths do not match')
    
    ax.plot(time, signal, label=plot_label, color=plot_color, alpha=plot_alpha)

    if reference_point != None:
        if not isinstance(reference_point, float):
            raise TypeError('reference_point should be float')
        if reference_point < 0.0 or reference_point > time[-1]:
            raise ValueError('Wrong value of reference_point: reference_point should be higher than 0.0 and lower than time_interval')
        ax.axvline(x=reference_point, linestyle='dashed', color=rp_color, label=r'reference point')

    ax.set_xlim([time[0], time[-1]])
    plt.xticks(color='k', size = 18)
    plt.yticks(color='k', size = 18)
    ax.set_xlabel(r'Time $t$, s', fontsize=20)
    ax.set_ylabel(r'Amplitude', fontsize=20)
    ax.set_title(r'Signal', fontsize=20)
    ax.legend(fontsize=legend_size)

def plot_decomposition(axarr, signal, level, wavelet='db4', approximate_color='purple', detailing_color='green', dtype='float64'):
    """
    Decompose signal using discrete wavelet decomposition.

    Parameters
    ----------
    axarr : array of matplotlib.axes class
        Set of matplotlib Axes class variables, referring to the plot for high- and low-pass coefficients.
    signal : array_like
        Signal to decompose.
    level : int
        Number of decomposition levels.
    wavelet : wavelet_type (pywt)
        Wavelet to use for DWT of signal.
    """
    signal = np.array(signal, dtype=dtype)
    if signal.ndim != 1:
        raise TypeError('signal is not one-dimensional array')
    if not isinstance(level, int):
        raise TypeError('level should be integer')
    if level <= 1:
        raise ValueError('Wrong value of level: level should be higher than 1')

    data = signal
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=len(data), filter_len=w.dec_len)
    if (level > max_level):
        raise ValueError('Number of decomposition levels is too high!')

    for i in range(level):
        (data, coeff_d) = pywt.dwt(data, wavelet, mode='symmetric')
        axarr[i, 0].plot(data, color=approximate_color)
        axarr[i, 1].plot(coeff_d, color=detailing_color)
        axarr[i, 0].set_ylabel(r'Level {}'.format(i + 1), fontsize=14, rotation=90)
        axarr[i, 0].set_yticklabels([])
        axarr[i, 1].set_yticklabels([])
        if i == 0:
            axarr[i, 0].set_title(r'Approximation coefficients', fontsize=14)
            axarr[i, 1].set_title(r'Detail coefficients', fontsize=14)
    plt.tight_layout()

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

def lowpassfilter(signal, thresh, level, wavelet='db4'):
    """
    Signal reconstruction using DWT noise removing techique.

    Parameters
    ----------
    signal : array_like
        Signal to recom.
    thresh : double
        Threshold parameter.
    level : int
        Number of decomposition levels.
    wavelet : wavelet_type (pywt)
        Wavelet to use for DWT of signal.
    """
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode='symmetric', level=level)
    coeff[1:] = (pywt.threshold(i, value=thresh, mode='soft') for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode='sym')
    return reconstructed_signal

def generate_signal_sequence(m, n, reference_point, noise_type, noise_level):
    signal_sequence = [0 for i in range(m)]
    for i in range(len(signal_sequence)):
        signal_sequence[i] = [0 for j in range(2)]
    for i in range(m):
        time, signal = signal_construction(n, reference_point, noise_type, noise_level)
        signal_sequence[i][0] = time
        signal_sequence[i][1] = signal
    return signal_sequence   

def dependence(sequence, reference_point=0.5, wavelet='db4'):
    m = len(sequence)
    num_level = 5
    thresh = []
    for j in range(250):
        thresh.append(j * 0.0025)
    norm = [0 for l in range(num_level)]
    for l in range(num_level):
        norm[l] = [0 for j in range(len(thresh))]
        for i in range(len(sequence)):
            time = sequence[i][0]
            signal = sequence[i][1]
            for j in range(len(thresh)):
                reconstructed_signal = lowpassfilter(signal, thresh=thresh[j], level = l + 1, wavelet=wavelet)
                diff = [0 for k in range(len(time))]
                for k in range(len(time)):
                    if time[k] < reference_point:
                        diff[k] = abs(reconstructed_signal[k] - 1.0)
                    else:
                        diff[k] = abs(reconstructed_signal[k])
                x = np.cumsum(diff)
                norm[l][j] = norm[l][j] + x[-1] * (time[1] - time[0])/m
    return(thresh, norm) 

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
        time, signal = signal_construction(n, reference_point, noise_type, noise_level)
        rec_signal = lowpassfilter(signal, thresh, level, wavelet)
        delta = MFD(time, rec_signal, reference_point)
        if delta < eps:
            k = k + 1
    return k/m

def wavelet_transform(time, signal, scales, waveletname='gaus1', dtype='float64'):
    time, signal = np.array(time, dtype=dtype), np.array(signal, dtype=dtype)
    if time.ndim != 1:
        raise TypeError('time is not one-dimensional array')
    if signal.ndim != 1:
        raise TypeError('signal is not one-dimensional array')
    if time.shape != signal.shape:
        raise ValueError('time and signal lengths do not match')

    scales = np.array(scales, dtype=dtype)
    if scales.ndim != 1:
        raise TypeError('scales is not one-dimensional array')

    """
    Необходимо сделать проверку вейвлета
    """

    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = abs(coefficients)
    scale = 1.0 / frequencies

    return [time, scale, coefficients]

def plot_wavelet_transform(ax, time, signal, scales, waveletname='gaus1', reference_point=None, rp_color=None, legend_size='large', dtype='float64'):
    """
    Plot wavelet-transform diagram of input signal

    Parameters
    ----------
    ax : ___
        _______________________
    time : array_like
        Time grid on which discrete signal is defined.
    signal : array_like
        Test signal.
    scales : array_like
        _______________________
    waveletname : Wavelet object or name
        Wavelet to use
    reference_point: double or None
        _______________________
    rp_color: color or None
        _______________________
    """
    time, signal = np.array(time, dtype=dtype), np.array(signal, dtype=dtype)
    if time.ndim != 1:
        raise TypeError('time is not one-dimensional array')
    if signal.ndim != 1:
        raise TypeError('signal is not one-dimensional array')
    if time.shape != signal.shape:
        raise ValueError('time and signal lengths do not match')

    scales = np.array(scales, dtype=dtype)
    if scales.ndim != 1:
        raise TypeError('scales is not one-dimensional array')

    """
    Необходимо сделать проверку вейвлета
    """

    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = abs(coefficients)
    scale = 1.0 / frequencies
    contourlevels = np.arange(-4.0, 4.0, 0.5)

    im = ax.contourf(time, np.log2(scale), np.log2(power), contourlevels, extend='both')
    
    #plt.xticks(color='k', size=18)
    #plt.yticks(color='k', size=18)
    ax.set_xlabel(r'Time $t$, s', fontsize=20)
    ax.set_ylabel(r'Scale $s$', fontsize=20)
    ax.set_title(r'Wavelet Transform (Power Spectrum) of signal', fontsize=20)
    
    ax.axvline(x=reference_point, linestyle='dashed', color=rp_color, label=r'reference point')

    #left, bottom, width, height = (0.22, np.log2(1.0), 0.06, 1.0)
    #rect = plt.Rectangle((left, bottom), width, height, fill=None, edgecolor='red', linewidth=2.0)
    #ax.add_patch(rect)

    ax.set_xlim([time[0], time[-1]])
    yticks = 2**np.arange(np.ceil(np.log2(scale.min())), np.ceil(np.log2(scale.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')
    plt.tight_layout()

    ax.legend(fontsize=legend_size, loc='upper right')

    return [time, scale, coefficients]

def maximum_of_wavelet_transform_modulus(ax, time, scale, coefficients, reference_point=None, rp_color=None, legend_size='large'):
    """
    Plot the diagram of local maxima of the wavelet transform modulus

    Parameters
    ----------
    ax : ___
        _______________________
    time : array_like
        Time grid on which discrete signal is defined.
    signal : array_like
        Test signal.
    scale : array_like
        Scale grid on which the wavelet transform is set
    reference_point: double or None
        _______________________
    rp_color: color or None
        _______________________
    
    """

    def find_local_max(sequence):
        index_local_max = []
        for i in range(1, len(sequence) - 1):
            if sequence[i] > sequence[i-1] and sequence[i] >= sequence[i+1] or sequence[i] >= sequence[i-1] and sequence[i] > sequence[i+1]:
                index_local_max.append(i)
        return index_local_max

    for i in range(len(scale)):
        index_local_max = find_local_max(abs(coefficients[:][i]))
        time_local_max = []
        scale_local_max = []
        for j in range(len(index_local_max)):
            time_local_max.append(time[index_local_max[j]])
            scale_local_max.append(np.log2(scale[i]))
        ax.scatter(time_local_max, scale_local_max, color='black', s=0.05)

    ax.set_xlabel(r'Time $t$, s', fontsize=18)
    ax.set_ylabel(r'Scale $s$', fontsize=18)
    ax.set_title(r'Local maximums of wavelet transform modulus', fontsize=18)

    ax.axvline(x=reference_point, linestyle='dashed', color=rp_color, label='reference point')

    yticks = 2**np.arange(np.ceil(np.log2(scale.min())), np.ceil(np.log2(scale.max())))
    plt.xticks(color='k', size=18)
    plt.yticks(color='k', size=18)
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(0.585, -5)
    
    ax.legend(fontsize=legend_size, loc='upper right')

def gutter_method(time, scale, power, interval_mode='auto', lower_scale=None, upper_scale=None, search_mode='window', optimal_variance=None, dtype='float64'):
    """
    Find the value of reference point using the gutter method 

    Parameters
    ----------
    
    time : array_like
        Time grid on which discrete signal is defined.
    scale : array_like
        Scale grid on which the wavelet transform is set.
    power : array_like
        Modulus of coefficients of continious wavelet transform calculated on
        time and scale grids.
    search_mode : str
        Mode for searching the value of reference point - 'window' or 'average'
    interval_mode: str
        Mode for interval definition - 'auto' or 'bound'
    lower_scale: double
        Lower bound of scale parameter in 'bound' interval_mode
    upper_scale: double
        Upper bound of scale parameter in 'bound' interval_mode
    """
    time, scale, power = np.array(time, dtype=dtype), np.array(scale, dtype=dtype), np.array(power, dtype=dtype)
    if time.ndim != 1:
        raise TypeError('time is not one-dimensional array')
    if scale.ndim != 1:
        raise TypeError('scale is not one-dimensional array')
    if power.ndim != 2:
        raise TypeError('power is not two-dimensional array')

    #if power.shape[0] != scale.shape:
    #    print(power.shape[0], scale.shape)
    #    raise ValueError('power and scale lengths do not match')
    #if power.shape[1] != time.shape:
    #    raise ValueError('power and time lengths do not match')   
 
    if interval_mode != 'auto' and interval_mode != 'bound':
        raise ValueError('Wrong value of interval_mode: interval_mode should be `auto` or `bound`')

    if interval_mode == 'auto':
        lower_scale = min(scale)
        upper_scale = max(scale)

    if search_mode == 'average':
        in_interval = sum(lower_scale < x < upper_scale for x in scale)
        average = 0
        variance = 0

        for i in range(len(scale)):
            if lower_scale < scale[i] < upper_scale:
                average = average + np.argmin(power[i][0:int(len(time)) // 2])/in_interval
                index_rp_exp = int(np.floor(2 * average))
        
        for i in range(len(scale)):
            if lower_scale < scale[i] < upper_scale:
                variance = variance + (average - np.argmin(power[i][0:int(len(time)) // 2]))**2/in_interval
                variance = 2 * np.sqrt(variance) * (time[1] - time[0])

        if variance < optimal_variance:
            rp_exp = time[index_rp_exp]
            return rp_exp, variance
        else:
            raise ValueError('Gutter method can not find the value of reference point with optimal variance!')
             
    elif search_mode == 'window':
        scale_in_interval = []
        min_t = []

        for i in range(len(scale)):
            if lower_scale < scale[i] < upper_scale:
                scale_in_interval.append(scale[i])
                gutter_pos = np.argmin(power[i][0:int(len(time)//2)])
                min_t.append(time[2 * gutter_pos])

        window = {i: min_t.count(i) for i in min_t} 
        window = {k: v for k, v in sorted(window.items(), key=lambda item: item[1])}
        rp_exp = max(window.items(), key=operator.itemgetter(1))[0]
        return scale_in_interval, min_t, window, rp_exp 

    else:
        raise ValueError('Wrong value of search_mode: search_mode should be `average` or `window`')

def test_gutter_method(m, n, max_noise_level, n_noise_level, optimal_delta):
    reference_point = np.random.random() * 0.1
    scales = np.arange(1, 512, 1.0)

    d_nl = max_noise_level / n_noise_level

    noise_level = [0 for i in range(n_noise_level)]
    test_result = [0 for i in range(n_noise_level)]
    for i in range(len(noise_level)):
        noise_level[i] = (i + 1) * d_nl
        for j in range(m):
            time, signal = signal_construction(n, reference_point, noise_type='gaussian', noise_level=noise_level[i])
            thresh = noise_level[i]
            level = 4
            rec_signal = lowpassfilter(signal, thresh, level)

            time, scale, coefficients = wavelet_transform(time, rec_signal, scales, waveletname='gaus1')
            reference_point_exp, variance = gutter_method(time, scale, power=abs(coefficients), lower_scale=1.0, upper_scale=2.0)
            delta = abs(reference_point - reference_point_exp)
            if delta < optimal_delta:
                test_result[i] += 1/m
        print(i, 'Finished!')
    return [noise_level, test_result]
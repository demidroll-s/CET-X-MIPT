from os import error
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pywt

plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage[utf8]{inputenc}',
            r'\usepackage[russian]{babel}',
            r'\usepackage{amsmath}', 
            r'\usepackage{siunitx}']

def signal_construction(n, time_interval, reference_point, noise_type, noise_level, dtype='float64'):
    """
    Construction of discrete signal from sensor 
    that registers the negative pressure wave.
    
    Parameters
    ----------
    n : int
        Length of discrete signal.
    time_interval : float
        The time interval at which the signal is defined.
    reference_point : float
        The moment of signal arrival.
    noise_type : str
        Type of noise - 'gaussian' or 'uniform'.
    noise_level : float
        Noise level in constructed signal.
    dtype : data type, optional
        The desired data-type for the array. If not given, then the type will be float64.
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
    time = [i * time_step for i in range(n)]
    signal = [0.0 for _ in range(n)]

    if not isinstance(reference_point, float):
        raise TypeError('reference_point should be float')
    if reference_point < 0.0 or reference_point > time_interval:
        raise ValueError('Wrong value of reference_point: reference_point should be higher than 0.0 and lower than time_interval')

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
            raise ValueError('Wrong value of noise_type: noise_type should be ''gaussian'' or ''uniform''')

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
    time_interval : float
        The time interval at which the signal is defined.
    reference_point : float
        The moment of signal arrival.
    width : float
        Characteristic period of pressure drop.
    noise_type : str
        Type of noise - 'gaussian' or 'uniform'
    noise_level : float
        Noise level in constructed signal.
    dtype : data type, optional
        The desired data-type for the array. If not given, then the type will be float64.
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
    parameters : optional
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
        if reference_point < time[0] or reference_point > time[-1]:
            raise ValueError('Wrong value of reference_point: reference_point should be in time_interval')
        ax.axvline(x=reference_point, linestyle='dashed', color=rp_color, label=r'reference point')

    ax.set_xlim([time[0], time[-1]])
    plt.xticks(color='k', size=18)
    plt.yticks(color='k', size=18)
    ax.set_xlabel(r'Time $t$, s', fontsize=20)
    ax.set_ylabel(r'Amplitude', fontsize=20)
    ax.set_title(r'Signal', fontsize=20)
    ax.legend(fontsize=legend_size)

def plot_decomposition(axarr, signal, level=4, wavelet='db4', approximate_color='purple', detailing_color='green', dtype='float64'):
    """
    Decompose signal using discrete wavelet decomposition.

    Parameters
    ----------
    axarr : array of matplotlib.axes class
        Set of matplotlib Axes class variables, referring to the plot for high- and low-pass coefficients.
    signal : array_like
        Signal to decompose.
    level : int, optional
        Number of decomposition levels.
    wavelet : wavelet_type (pywt), optional
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

    """
    Необходимо сделать проверку вейвлета
    """

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

def lowpassfilter(signal, thresh, level=4, wavelet='db4'):
    """
    Signal reconstruction using DWT noise removing techique.

    Parameters
    ----------
    signal : array_like
        Signal to recom.
    thresh : double
        Threshold parameter.
    level : int, optional
        Number of decomposition levels.
    wavelet : wavelet_type (pywt), optional
        Wavelet to use for DWT of signal.
    """

    """
    Необходимо сделать проверку вейвлета
    """

    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode='symmetric', level=level)
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode='sym')
    return reconstructed_signal

def wavelet_transform(time, signal, scales, waveletname='gaus1', dtype='float64'):
    """
    Compute continuous wavelet-transform of input signal

    Parameters
    ----------
    time : array_like
        Time grid on which discrete signal is defined.
    signal : array_like
        Input signal.
    scales : array_like
        The wavelet scales to use.
    waveletname : Wavelet object or name
        Wavelet to use.
    dtype : data type, optional
        The desired data-type for the array. If not given, then the type will be float64.
    Returns
    -------
    time : array_like
        Time grid on which discrete signal is defined.
    scales : array_like
        The wavelet scales on which cwt was calculated.
    coefficients : array_like
        Coefficients of continuous wavelet transform of input signal.

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

    return [time, scale, coefficients]

def plot_wavelet_transform(ax_signal, ax_cwt, time, signal, scales, waveletname='gaus1', reference_point=None, rp_color=None, legend_size='large', dtype='float64'):
    """
    Plot wavelet-transform diagram of input signal

    Parameters
    ----------
    ax_signal : matplotlib.axes class
        Variable of matplotlib Axes class, referring to the plot for signal.
    ax_cwt : matplotlib.axes class
        Variable of matplotlib Axes class, referring to the diagram of CWT.
    time : array_like
        Time grid on which discrete signal is defined.
    signal : array_like
        Test signal.
    scales : array_like
        The wavelet scales to use.
    waveletname : Wavelet object or name
        Wavelet to use
    reference_point : float
        The moment of signal arrival.
    dtype : data type, optional
        The desired data-type for the array. If not given, then the type will be float64.
    parameters : optional
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

    im = ax_cwt.contourf(time, np.log2(scale), np.log2(power), contourlevels, extend='both')
    
    plt.xticks(color='k', size=18)
    plt.yticks(color='k', size=18)
    ax_cwt.set_xlabel(r'Time $t$, s', fontsize=20)
    ax_cwt.set_ylabel(r'Scale $s$', fontsize=20)
    ax_cwt.set_title(r'Wavelet Transform (Power Spectrum) of signal', fontsize=20)
    
    ax_cwt.axvline(x=reference_point, linestyle='dashed', color=rp_color, label='reference point')

    ax_cwt.set_xlim([time[0], time[-1]])
    yticks = 2**np.arange(np.ceil(np.log2(scale.min())), np.ceil(np.log2(scale.max())))
    ax_cwt.set_yticks(np.log2(yticks))
    ax_cwt.set_yticklabels(yticks)
    ax_cwt.invert_yaxis()
    ylim = ax_cwt.get_ylim()
    ax_cwt.set_ylim(ylim[0], -5)
    
    divider = make_axes_locatable(ax_cwt)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')
    plt.tight_layout()

    ax_cwt.legend(fontsize=legend_size)

    ax = [ax_signal, ax_cwt]

    for tick in ax[0].xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax[0].yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax[1].xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax[1].yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    return [time, scale, coefficients]

def maximum_of_wavelet_transform_modulus(ax, time, scale, coefficients, reference_point=None, rp_color=None, legend_size='large'):
    """
    Plot the diagram of local maxima of the wavelet transform modulus

    Parameters
    ----------
    ax : matplotlib.axes class
        Variable of matplotlib Axes class, referring to the plot for signal.
    time : array_like
        Time grid on which discrete signal is defined.
    signal : array_like
        Test signal.
    scale : array_like
        Scale grid on which the wavelet transform is set
    reference_point: float or None
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
    lower_scale: float
        Lower bound of scale parameter in 'bound' interval_mode
    upper_scale: float
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
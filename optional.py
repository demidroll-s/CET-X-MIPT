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
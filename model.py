import numpy as np
from scipy import special

def pressure_case_a(t, x, c, a, rho, p_0):
    if t < x/c:
        return 0
    else:
        kappa = np.linspace(x/c, t, 101)
        xi = []
        for i in range(len(kappa)):
            y = np.sqrt(kappa[i]**2 - (x/c)**2)
            xi.append(y)
        subintegrative = [0 for i in range(len(kappa))]
        for i in range(len(kappa)):
            if xi[i] != 0:
                subintegrative[i] = special.iv(1, a * xi[i])/xi[i] * np.exp((-a) * kappa[i])
            else:
                subintegrative[i] = 0
        integral = (np.cumsum(subintegrative))[-1] * (kappa[1] - kappa[0])
        result = p_0 * np.exp(- a * x/c) + a * p_0 * x/c  * integral
    return result

def velocity_case_a(t, x, c, a, rho, p_0):
    if t < x/c:
        return 0
    else:
        kappa = np.linspace(x/c, t, 101)
        xi = []
        for i in range(len(kappa)):
            y = np.sqrt(kappa[i]**2 - (x/c)**2)
            xi.append(y)
        subintegrative = [0 for i in range(len(kappa))]
        for i in range(len(kappa)):
            if xi[i] != 0:
                subintegrative[i] = (kappa[i] * special.iv(1, a * xi[i])/xi[i] - special.iv(0, a * xi[i])) * np.exp((-a) * kappa[i]) 
            else:
                subintegrative[i] = 0
        integral = (np.cumsum(subintegrative))[-1] * (kappa[1] - kappa[0])
        result = (p_0 * np.exp(- a * x/c) + a * p_0  * integral) / (rho * c)
    return result

def pressure_case_b(t, x, c, a, rho, p_0):
    if t < x/c:
        return 0
    else:
        kappa = np.linspace(x/c, t, 101)
        xi = []
        for i in range(len(kappa)):
            y = np.sqrt(kappa[i]**2 - (x/c)**2)
            xi.append(y)
        subintegrative = [0 for i in range(len(kappa))]
        for i in range(len(kappa)):
            if xi[i] != 0:
                subintegrative[i] = special.iv(1, a * xi[i])/xi[i] * np.exp((-a) * kappa[i])
            else:
                subintegrative[i] = 0
        integral = (np.cumsum(subintegrative))[-1] * (kappa[1] - kappa[0])
        result = p_0 * np.exp(- a * x/c) + a * p_0 * x/c  * integral
    return result

def velocity_case_b(t, x, c, a, rho, p_0):
    if t < x/c:
        return 0
    else:
        kappa = np.linspace(x/c, t, 101)
        xi = []
        for i in range(len(kappa)):
            y = np.sqrt(kappa[i]**2 - (x/c)**2)
            xi.append(y)
        subintegrative = [0 for i in range(len(kappa))]
        for i in range(len(kappa)):
            if xi[i] != 0:
                subintegrative[i] = (kappa[i] * special.iv(1, a * xi[i])/xi[i] - special.iv(0, a * xi[i])) * np.exp((-a) * kappa[i]) 
            else:
                subintegrative[i] = 0
        integral = (np.cumsum(subintegrative))[-1] * (kappa[1] - kappa[0])
        result = (p_0 * np.exp(- a * x/c) + a * p_0  * integral) / (rho * c)
    return result



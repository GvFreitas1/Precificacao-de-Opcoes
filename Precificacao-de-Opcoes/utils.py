from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import numpy as np


def black_scholes(S, X, T, r, sigma):
    d1 = (np.log(S/X) + (r + (sigma**2)/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)  # = np.log(S/X) + (r - (sigma**2)/2)*T / (sigma*np.sqrt(T))

    C_BS = S*norm.cdf(d1) - X*np.exp(-r*T)*norm.cdf(d2)

    return C_BS


def freitas(S, X, T, r, mi, sigma):
    mi_ST = np.log(S) + T*mi
    sigma_ST = np.sqrt(T)*sigma

    def ret_lin(s, C):
        return (s-(X+C))*(1/(s*sigma_ST*np.sqrt(2*np.pi)))*np.exp(-(np.log(s)-mi_ST)**2/(2*sigma_ST**2))

    def ret_ct(s, C):
        return -C*(1/(s*sigma_ST*np.sqrt(2*np.pi)))*np.exp(-(np.log(s)-mi_ST)**2/(2*sigma_ST**2))

    def diferenca_esperancas(C):
        Pt = quad(ret_ct, 0, X, args=(C))[0]
        Pp = quad(ret_lin, X, X + C, args=(C))[0]
        L = quad(ret_lin, X + C, np.inf, args=(C))[0]
        D = abs(L + Pt + Pp)
        return D

    C_freitas = minimize_scalar(diferenca_esperancas).x / (1+r)**T

    return C_freitas
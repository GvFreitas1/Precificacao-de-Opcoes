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

    C_freitas = minimize_scalar(diferenca_esperancas).x * np.exp(-r*mi_ST*T)

    return C_freitas


def bachelier(S, X, T, r, sigma):
    D = (S-X)/(sigma*np.sqrt(T))
    C_B = (S-X)*norm.cdf(D) + sigma*np.sqrt(T)*norm.pdf(D)

    return C_B

def bachelier_option_price(S, K, T, sigma, r):
    """
    Calcula o preço de uma opção usando o modelo de Bachelier.

    Parâmetros:
        S (float): Preço atual do ativo subjacente.
        K (float): Preço de exercício (strike price).
        T (float): Tempo até o vencimento (em anos).
        sigma (float): Volatilidade do ativo (desvio padrão absoluto do preço).
        r (float): Taxa de juros livre de risco.
        option_type (str): Tipo da opção ('call' ou 'put').

    Retorno:
        float: Preço da opção calculado pelo modelo de Bachelier.
    """
    # Diferença entre preço do ativo e strike
    d = S - K
    # Variância ajustada pelo tempo
    sigma_t = sigma * np.sqrt(T)
    
    # Cálculo de d1 (normalizado pela volatilidade ajustada)
    d1 = d / sigma_t
    
    price = np.exp(-r * T) * (d * norm.cdf(d1) + sigma_t * norm.pdf(d1))
    
    return price
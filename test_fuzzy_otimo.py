import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Visualização e utilidades
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# ==== Funções Fuzzy ====
def trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Função de pertinência triangular.
    """
    if x < a or x > c:
        return 0.0
    elif a <= x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return (c - x) / (c - b)

# ==== Dinâmica do sistema ====
def dinamica(u: np.ndarray, x: float, t: float, T: float) -> tuple:
    """
    Calcula a dinâmica fuzzy superior e inferior.
    """
    tv = np.linspace(0, T, len(u))
    hs = tv[1]
    aux = np.interp(t, tv, hs * np.cumsum(u))
    return x + aux, x - aux

# ==== Construção da solução fuzzy ====
def construir_sol(t: np.ndarray, z: np.ndarray, u: np.ndarray, fuz: tuple) -> np.ndarray:
    """
    Constrói a matriz de solução fuzzy.
    """
    rs = np.zeros((len(z), len(t)))
    u = t[1] * np.cumsum(u)
    for i, vt in enumerate(t):
        for j, vz in enumerate(z):
            rs[j, i] = trimf(vz - u[i], *fuz)
    return rs

# ==== Cálculo dos níveis fuzzy ====
def sol_levels(u: np.ndarray, fuz: tuple, nniv: int) -> list:
    """
    Retorna os limites inferior e superior dos níveis fuzzy.
    """
    niveis = np.linspace(0, 1, nniv)
    xniveis = np.sort(np.concatenate([np.linspace(fuz[0], fuz[2], 10 * nniv), [fuz[1]]])).reshape(-1, 1)
    xpert = np.array([trimf(x[0], *fuz) for x in xniveis]).reshape(-1, 1)
    hs = 1 / len(u)
    sol = hs * np.concatenate([[0], np.cumsum(u)]) + xniveis

    Sup = np.zeros((nniv, len(u) + 1))
    Inf = np.zeros((nniv, len(u) + 1))

    for i in range(nniv):
        candidatos = [sol[j] for j in range(len(xniveis)) if xpert[j] >= niveis[i]]
        if candidatos:
            Sup[i, :] = np.max(candidatos, axis=0)
            Inf[i, :] = np.min(candidatos, axis=0)
        else:
            Sup[i, :] = np.nan
            Inf[i, :] = np.nan

    return [Inf, Sup]

# ==== Funcional objetivo ====
def funcional(u: np.ndarray, gamma: float, beta: float, e: float, x0: float, T: float, alpha: float) -> float:
    """
    Calcula o funcional de custo do controle ótimo fuzzy.
    """
    hs = T / len(u)
    w = hs * np.cumsum(u)
    phit = e * (6 * w**2 - 12 * gamma * w + 12 * x0 * w + e**2 + 6 * gamma**2 - 12 * gamma * x0 + 6 * x0**2) / 6
    Iu2 = hs * np.sum(u**2)
    ju = 0.5 * (1 - alpha) * phit[-1] + 0.5 * alpha * hs * np.sum(phit) + 0.5 * beta * Iu2
    return ju

# ==== Métricas customizadas com PyTorch ====
def minhas_metricas(gamma: float, beta: float, e: float, x0: float, T: float, alpha: float, len_input: int):
    """
    Retorna três funções para usar como métricas ou funções de perda com PyTorch.
    """
    nniv = 100
    xniveis = np.sort(np.concatenate([np.linspace(x0 - e, x0 + e, nniv), [x0]])).reshape(-1, 1)
    xpert = np.array([trimf(x[0], x0 - e, x0, x0 + e) for x in xniveis]).reshape(-1, 1)
    t = np.linspace(0, T, len_input)
    hs = T / len_input

    xniveis_torch = torch.tensor(xniveis.T, dtype=torch.float32)
    xpert_torch = torch.tensor(xpert, dtype=torch.float32)
    dD = torch.tensor(np.abs(xniveis[1:] - xniveis[:-1]), dtype=torch.float32).T
    ap = torch.tensor((xpert[:-1] + xpert[1:]) / 2, dtype=torch.float32).T

    def funcional_metric(y_pred):
        D = hs * torch.cumsum(y_pred, dim=1) + xniveis_torch
        aD = (D[:, :-1] + D[:, 1:]) / 2
        auxD = (aD - gamma) ** 2 * ap
        phit = torch.sum(auxD * dD, dim=1, keepdim=True)
        Iu2 = hs * torch.sum(y_pred ** 2, dim=1, keepdim=True)
        Iphit = hs * torch.sum(phit, dim=1, keepdim=True)
        ju = 0.5 * (1 - alpha) * phit[:, -1] + 0.5 * alpha * Iphit + 0.5 * beta * Iu2
        return torch.mean(ju)

    def loss_u(y_pred):
        t_torch = torch.tensor(t, dtype=torch.float32).unsqueeze(0)
        intu = hs * torch.sum(y_pred, dim=1, keepdim=True)
        aux2 = hs * torch.cumsum(y_pred, dim=1)
        H1 = alpha * e * (T - t_torch) * (gamma - x0)
        H2 = -alpha * e * hs * (torch.sum(aux2, dim=1, keepdim=True) - torch.cumsum(aux2, dim=1))
        H3 = e * (1 - alpha) * (gamma - x0 - intu)
        diff = H1 + H2 + H3 - beta * y_pred
        norma = torch.sqrt(torch.sum(diff ** 2, dim=1)) / len_input
        return torch.mean(norma)

    def funcional2(y_pred):
        w = hs * torch.cumsum(y_pred, dim=1)
        phit = e * (6 * w ** 2 - 12 * gamma * w + 12 * x0 * w + e ** 2 + 6 * gamma ** 2 - 12 * gamma * x0 + 6 * x0 ** 2) / 6
        Iphit = hs * torch.sum(phit, dim=1, keepdim=True)
        Iu2 = hs * torch.sum(y_pred ** 2, dim=1, keepdim=True)
        ju = 0.5 * (1 - alpha) * phit[:, -1] + 0.5 * alpha * Iphit + 0.5 * beta * Iu2
        return torch.mean(ju)

    return funcional_metric, loss_u, funcional2
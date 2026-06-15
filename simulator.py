"""
simulator.py
------------
Simulador de carregamento slot a slot.

Em cada slot t:
  1. Identifica os EVs ativos (chegaram, não partiram, não satisfeitos)
  2. Chama o algoritmo de alocação
  3. Entrega energia a cada EV
  4. Registra métricas

Métricas coletadas (alinhadas com Seção 7 do artigo):
  - Corrente total por fase (para análise de desequilíbrio)
  - Taxa de satisfação de demanda por EV
  - Índice de fairness de Jain
  - Pico de carga (peak demand)
  - Tempo de execução do algoritmo por slot
"""

import time
import numpy as np
import pandas as pd
from copy import deepcopy

from models import EVSession, ChargingInfrastructure
from algorithms import phase_currents, phase_imbalance


def jain_fairness_index(values: list[float]) -> float:
    """
    Índice de fairness de Jain (Jain et al., 1984).
    Varia de 1/n (máxima injustiça) a 1.0 (perfeita equidade).
    
    J = (Σ x_i)² / (n · Σ x_i²)
    
    Usado amplamente em literatura de redes e smart grid para
    avaliar distribuição equitativa de recursos.
    """
    v = np.array([x for x in values if x > 0])
    if len(v) == 0:
        return 1.0
    return float((v.sum() ** 2) / (len(v) * (v ** 2).sum()))


def run_simulation(
    sessions: list[EVSession],
    infra: ChargingInfrastructure,
    charger,
    n_slots: int = 144,
    verbose: bool = False,
) -> dict:
    """
    Executa a simulação completa.

    Parâmetros
    ----------
    sessions : lista de sessões (serão deep-copied para não alterar originais)
    infra    : infraestrutura de carregamento
    charger  : instância de GreedyCharger, BranchBoundCharger, ou Uncontrolled
    n_slots  : número de slots de simulação
    verbose  : imprime progresso

    Retorna
    -------
    dict com métricas agregadas e séries temporais
    """
    # Deep copy para não modificar os originais
    evs = deepcopy(sessions)
    ev_map = {ev.session_id: ev for ev in evs}
    dt_hours = infra.dt_minutes / 60.0

    # Séries temporais
    ts_phase_A = []
    ts_phase_B = []
    ts_phase_C = []
    ts_total_current = []
    ts_imbalance = []
    ts_n_active = []
    ts_algo_time_ms = []

    for t in range(n_slots):
        # Seleciona EVs ativos no slot t
        active = [
            ev for ev in evs
            if ev.t_arrival <= t < ev.t_departure and not ev.is_satisfied
        ]

        t0 = time.perf_counter()
        allocations = charger.allocate(active, t)
        algo_time_ms = (time.perf_counter() - t0) * 1000

        # Entrega energia
        for ev in active:
            i_pilot = allocations.get(ev.session_id, 0.0)
            ev.deliver_energy(i_pilot, dt_hours)

        # Registro de métricas de fase
        phase_sum = np.zeros(3)
        for ev in active:
            phase_sum[ev.phase] += allocations.get(ev.session_id, 0.0)

        ts_phase_A.append(phase_sum[0])
        ts_phase_B.append(phase_sum[1])
        ts_phase_C.append(phase_sum[2])
        ts_total_current.append(phase_sum.sum())
        ts_imbalance.append(phase_imbalance(phase_sum))
        ts_n_active.append(len(active))
        ts_algo_time_ms.append(algo_time_ms)

        if verbose and t % 24 == 0:
            print(f"  t={t:3d} | ativos={len(active):3d} | "
                  f"I=[{phase_sum[0]:.1f}, {phase_sum[1]:.1f}, {phase_sum[2]:.1f}]A | "
                  f"imbal={ts_imbalance[-1]:.2f}A | {algo_time_ms:.2f}ms")

    # --- Métricas por EV ---
    satisfaction_ratios = [ev.satisfaction_ratio for ev in evs]
    satisfied_count = sum(1 for ev in evs if ev.is_satisfied)

    # --- Métricas agregadas ---
    results = {
        "algorithm": charger.name(),
        # Satisfação de demanda
        "n_sessions": len(evs),
        "n_satisfied": satisfied_count,
        "satisfaction_rate": satisfied_count / max(1, len(evs)),
        "mean_satisfaction_ratio": float(np.mean(satisfaction_ratios)),
        "min_satisfaction_ratio": float(np.min(satisfaction_ratios)),
        # Fairness
        "jain_fairness_index": jain_fairness_index(satisfaction_ratios),
        # Fase e desequilíbrio
        "mean_imbalance_A": float(np.mean(ts_imbalance)),
        "max_imbalance_A": float(np.max(ts_imbalance)),
        "peak_total_current_A": float(np.max(ts_total_current)),
        "mean_total_current_A": float(np.mean(ts_total_current)),
        # Eficiência computacional
        "mean_algo_time_ms": float(np.mean(ts_algo_time_ms)),
        "max_algo_time_ms": float(np.max(ts_algo_time_ms)),
        # Séries temporais (para gráficos)
        "ts": {
            "phase_A": ts_phase_A,
            "phase_B": ts_phase_B,
            "phase_C": ts_phase_C,
            "total":   ts_total_current,
            "imbalance": ts_imbalance,
            "n_active":  ts_n_active,
            "algo_time_ms": ts_algo_time_ms,
        },
        # EVs resultantes
        "evs": evs,
    }

    return results


def compare_algorithms(results_list: list[dict]) -> pd.DataFrame:
    """Gera tabela comparativa das métricas principais."""
    rows = []
    for r in results_list:
        rows.append({
            "Algoritmo":             r["algorithm"],
            "Taxa Satisfação (%)":   f"{r['satisfaction_rate']*100:.1f}",
            "Sat. Média (%)":        f"{r['mean_satisfaction_ratio']*100:.1f}",
            "Sat. Mínima (%)":       f"{r['min_satisfaction_ratio']*100:.1f}",
            "Fairness (Jain)":       f"{r['jain_fairness_index']:.4f}",
            "Deseq. Médio (A)":      f"{r['mean_imbalance_A']:.2f}",
            "Deseq. Máx. (A)":       f"{r['max_imbalance_A']:.2f}",
            "Pico Corrente (A)":     f"{r['peak_total_current_A']:.1f}",
            "Tempo Médio (ms)":      f"{r['mean_algo_time_ms']:.3f}",
        })
    return pd.DataFrame(rows).set_index("Algoritmo")

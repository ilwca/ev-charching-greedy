"""
data_generator.py
-----------------
Gerador de sessões sintéticas de carregamento.

As distribuições estatísticas são baseadas em:
  - Lee et al. (2019) ACN-Data — chegada, duração e demanda energética
  - Hegele et al. (2023) — variação de comportamento não-linear (~15% dos EVs)

O gerador replica as características do Caltech ACN:
  - Chegadas concentradas entre 8h e 10h (weekday)
  - Duração: 2–9 horas (log-normal)
  - Demanda: 5–60 kWh (log-normal, mediana ~15 kWh)
"""

import numpy as np
import pandas as pd
from models import EVSession, ChargingInfrastructure


def generate_sessions(
    infra: ChargingInfrastructure,
    n_sessions: int = 50,
    n_slots: int = 144,       # 144 slots × 5 min = 12 horas (7h–19h)
    nonlinear_fraction: float = 0.15,
    seed: int = 42,
) -> list[EVSession]:
    """
    Gera uma lista de sessões sintéticas de carregamento.

    Parâmetros
    ----------
    infra               : infraestrutura de carregamento
    n_sessions          : número de EVs a gerar
    n_slots             : horizonte de simulação em slots
    nonlinear_fraction  : fração de EVs com comportamento não-linear
    seed                : semente aleatória para reprodutibilidade

    Retorna
    -------
    Lista de EVSession ordenada por tempo de chegada.
    """
    rng = np.random.default_rng(seed)
    sessions = []

    # --- Distribuição de chegadas (concentrada pela manhã) ---
    # Modela chegadas entre slots 0–48 (primeiras 4 horas) com pico no slot 12
    arrival_probs = np.exp(-0.5 * ((np.arange(n_slots) - 24) / 18) ** 2)
    arrival_probs[:5] = 0           # ninguém chega nos primeiros 25 min
    arrival_probs[n_slots // 2:] *= 0.1  # menos chegadas na tarde
    arrival_probs /= arrival_probs.sum()

    # Garante no máximo uma sessão por estação ao mesmo tempo (simplificação)
    station_ids = rng.choice(infra.n_stations, size=n_sessions, replace=True)

    for i in range(n_sessions):
        # Chegada
        t_arr = int(rng.choice(n_slots, p=arrival_probs))

        # Duração da sessão: log-normal, mediana ~3h → ~36 slots de 5 min
        duration_slots = max(6, int(rng.lognormal(mean=3.6, sigma=0.5)))
        t_dep = min(t_arr + duration_slots, n_slots - 1)

        # Capacidade máxima do EV (variação de mercado)
        i_max_ev = rng.choice([16.0, 20.0, 32.0], p=[0.3, 0.2, 0.5])

        # Demanda energética: log-normal, mediana ~12 kWh
        # Limitada ao máximo fisicamente entregável na janela de estacionamento
        e_demand_kwh = max(1.0, rng.lognormal(mean=2.5, sigma=0.6))
        dt_hours = infra.dt_minutes / 60.0
        max_deliverable_wh = i_max_ev * 230.0 * (t_dep - t_arr) * dt_hours
        e_demand_wh = min(e_demand_kwh * 1000, max_deliverable_wh * 0.95)

        # Comportamento não-linear
        is_nonlinear = rng.random() < nonlinear_fraction

        station_id = int(station_ids[i])
        phase = infra.phases[station_id]

        sessions.append(EVSession(
            session_id=i,
            station_id=station_id,
            phase=phase,
            t_arrival=t_arr,
            t_departure=t_dep,
            e_demand=e_demand_wh,
            i_max=i_max_ev,
            nonlinear=is_nonlinear,
        ))

    sessions.sort(key=lambda s: s.t_arrival)
    return sessions


def sessions_to_dataframe(sessions: list[EVSession]) -> pd.DataFrame:
    """Converte lista de sessões em DataFrame para análise."""
    return pd.DataFrame([{
        "session_id":    s.session_id,
        "station_id":    s.station_id,
        "phase":         ["A", "B", "C"][s.phase],
        "t_arrival":     s.t_arrival,
        "t_departure":   s.t_departure,
        "duration_slots": s.t_departure - s.t_arrival,
        "e_demand_wh":   s.e_demand,
        "e_demand_kwh":  s.e_demand / 1000,
        "i_max_ev":      s.i_max,
        "nonlinear":     s.is_satisfied,  # será preenchido pós-simulação
    } for s in sessions])

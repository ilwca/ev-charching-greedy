"""
algorithms.py
-------------
Implementação dos algoritmos do artigo de Hegele et al. (2023).

Algoritmos implementados:
  1. GreedyCharger  — algoritmo guloso (Seção 6.2 do artigo)
  2. BranchBoundCharger — branch-and-bound exato (Seção 6.1)
  3. UncontrolledCharger — baseline: carrega na potência máxima (sem controle)

Formulação (Seção 5 do artigo):
  - O smart charging é modelado como um KNAPSACK DISCRETO no tempo.
  - Em cada slot t, para os EVs ativos A_t, escolhemos correntes i_j >= 0:
      Maximizar:    sum_j  w_j * i_j          (utilidade ponderada = fairness)
      Sujeito a:   sum_{j in fase_k} i_j <= C_k   for k in {A,B,C}
                   0 <= i_j <= i_max_j          for each j
                   |sum_fase_k - sum_fase_l| <= eps  (simetria)

  Onde w_j = e_rem_j / sum_k e_rem_k  é o peso de fairness proporcional.
"""

import numpy as np
from copy import deepcopy
from models import EVSession, ChargingInfrastructure


# ---------------------------------------------------------------------------
# Utilidades compartilhadas
# ---------------------------------------------------------------------------

def compute_fair_share(ev, t_current, dt_hours):
    """
    Corrente ideal para atender a demanda restante uniformemente
    nos slots restantes. Eq. (3) do artigo.
    """
    slots_rem = max(1, ev.t_departure - t_current)
    i_ideal = ev.e_remaining / (ev.v_nominal * slots_rem * dt_hours)
    return min(i_ideal, ev.i_max)


def phase_currents(active_evs, allocations):
    """Soma de corrente por fase [I_A, I_B, I_C]."""
    phase_sum = np.zeros(3)
    for ev in active_evs:
        phase_sum[ev.phase] += allocations.get(ev.session_id, 0.0)
    return phase_sum


def phase_imbalance(phase_sum):
    """Desvio padrão das correntes por fase (0 = equilíbrio perfeito)."""
    return float(np.std(phase_sum))


# ---------------------------------------------------------------------------
# 1. Algoritmo Guloso (Greedy) — Algoritmo principal do artigo
# ---------------------------------------------------------------------------

class GreedyCharger:
    """
    Algoritmo guloso para alocação de potência em cada slot de tempo.

    Estratégia (Seção 6.2):
      1. Calcula o 'fair share' de cada EV ativo.
      2. Ordena os EVs por fair share DECRESCENTE (quem precisa mais, primeiro).
      3. Aloca corrente a cada EV respeitando budget de fase e simetria.

    Complexidade: O(n log n) por slot.
    """

    def __init__(self, infra, symmetry_tolerance=5.0):
        self.infra = infra
        self.eps = symmetry_tolerance

    def allocate(self, active_evs, t_current):
        if not active_evs:
            return {}

        dt_hours = self.infra.dt_minutes / 60.0

        # Passo 1: fair share e ordenação
        ev_with_share = [
            (ev, compute_fair_share(ev, t_current, dt_hours))
            for ev in active_evs
        ]
        ev_with_share.sort(key=lambda x: x[1], reverse=True)

        # Passo 2: budget por fase
        phase_budget = np.full(3, self.infra.phase_budget)
        allocations = {ev.session_id: 0.0 for ev in active_evs}
        active_phases = set(ev.phase for ev in active_evs)

        # Passo 3: alocação gulosa com restrição de simetria adaptativa
        for ev, i_share in ev_with_share:
            phase = ev.phase
            available = phase_budget[phase]
            i_alloc = min(i_share, ev.i_max, available)

            # Restrição de simetria: aplica apenas quando fases diferentes
            # já receberam alocação. Evita deadlock mútuo onde nenhuma fase
            # consegue iniciar por esperar que a outra inicie primeiro.
            other_active = active_phases - {phase}
            if other_active:
                ps = phase_currents(active_evs, allocations)
                allocated_others = [p for p in other_active if ps[p] > 0]
                if allocated_others:
                    max_other = max(ps[p] for p in allocated_others)
                    i_sym_limit = max_other + self.eps - ps[phase]
                    i_alloc = min(i_alloc, max(0.0, i_sym_limit))

            if i_alloc < ev.i_min:
                i_alloc = 0.0

            allocations[ev.session_id] = i_alloc
            phase_budget[phase] -= i_alloc

        return allocations

    def name(self):
        return "Greedy"


# ---------------------------------------------------------------------------
# 2. Branch-and-Bound (solução exata para fins de comparação)
# ---------------------------------------------------------------------------

class BranchBoundCharger:
    """
    Branch-and-bound para o knapsack de carregamento.

    O artigo (Seção 6.1) usa B&B para a solução ótima em simulações offline.
    Esta implementação usa discretização da corrente em passos de
    `current_step` Ampères para tornar o problema tratável.

    Limitado a N_MAX_EXACT EVs simultâneos para viabilidade computacional.
    """

    N_MAX_EXACT = 8

    def __init__(self, infra, current_step=8.0, symmetry_tolerance=5.0):
        self.infra = infra
        self.step = current_step
        self.eps = symmetry_tolerance

    def _upper_bound(self, evs_remaining, phase_budgets, current_value):
        ub = current_value
        for ev, weight, i_share in evs_remaining:
            available = phase_budgets[ev.phase]
            ub += weight * min(i_share, ev.i_max, available)
        return ub

    def allocate(self, active_evs, t_current):
        if not active_evs:
            return {}

        dt_hours = self.infra.dt_minutes / 60.0
        total_e = sum(ev.e_remaining for ev in active_evs) or 1.0
        weights = {ev.session_id: ev.e_remaining / total_e for ev in active_evs}

        ev_shares = [
            (ev, weights[ev.session_id], compute_fair_share(ev, t_current, dt_hours))
            for ev in active_evs
        ]
        ev_shares.sort(key=lambda x: x[1], reverse=True)

        # Limitar para viabilidade
        ev_shares = ev_shares[:self.N_MAX_EXACT]

        # Opções de corrente para cada EV
        current_options = {}
        for ev, w, i_share in ev_shares:
            max_i = min(ev.i_max, max(i_share * 1.2, ev.i_min + self.step))
            opts = [0.0]
            i = ev.i_min
            while i <= max_i + 1e-6:
                opts.append(min(i, ev.i_max))
                i += self.step
            current_options[ev.session_id] = sorted(set(opts), reverse=True)

        best = {
            "value": -1.0,
            "alloc": {ev.session_id: 0.0 for ev, _, _ in ev_shares}
        }

        def branch(idx, phase_budgets, alloc, value):
            if idx == len(ev_shares):
                ps = np.array([
                    sum(alloc.get(ev.session_id, 0) for ev, _, _ in ev_shares
                        if ev.phase == p)
                    for p in range(3)
                ])
                active_p = [p for p in range(3) if ps[p] > 0 or
                            any(ev.phase == p for ev, _, _ in ev_shares
                                if alloc.get(ev.session_id, 0) > 0)]
                ok = True
                if len(active_p) > 1:
                    ok = (np.max(ps) - np.min(ps[ps > 0] if any(ps > 0) else ps)) <= self.eps + 1e-6
                if ok and value > best["value"]:
                    best["value"] = value
                    best["alloc"] = alloc.copy()
                return

            ub = self._upper_bound(ev_shares[idx:], phase_budgets.copy(), value)
            if ub <= best["value"] + 1e-9:
                return

            ev, w, i_share = ev_shares[idx]
            for i_alloc in current_options[ev.session_id]:
                if i_alloc > phase_budgets[ev.phase] + 1e-6:
                    continue
                new_budgets = phase_budgets.copy()
                new_budgets[ev.phase] -= i_alloc
                alloc[ev.session_id] = i_alloc
                branch(idx + 1, new_budgets, alloc,
                       value + w * i_alloc)

            alloc[ev.session_id] = 0.0

        initial_alloc = {ev.session_id: 0.0 for ev, _, _ in ev_shares}
        branch(0, np.full(3, self.infra.phase_budget), initial_alloc, 0.0)

        # EVs não incluídos no B&B recebem 0
        full_alloc = {ev.session_id: 0.0 for ev in active_evs}
        full_alloc.update(best["alloc"])
        return full_alloc

    def name(self):
        return "Branch-and-Bound"


# ---------------------------------------------------------------------------
# 3. Baseline: Carregamento não-controlado (FCFS)
# ---------------------------------------------------------------------------

class UncontrolledCharger:
    """
    Baseline: cada EV carrega na máxima corrente, sem coordenação.
    Representa o comportamento sem smart charging (Seção 7 do artigo).
    """

    def __init__(self, infra):
        self.infra = infra

    def allocate(self, active_evs, t_current):
        if not active_evs:
            return {}

        phase_budget = np.full(3, self.infra.phase_budget)
        allocations = {}

        for ev in sorted(active_evs, key=lambda e: e.t_arrival):
            available = phase_budget[ev.phase]
            i_alloc = min(ev.i_max, available)
            if i_alloc < ev.i_min:
                i_alloc = 0.0
            allocations[ev.session_id] = i_alloc
            phase_budget[ev.phase] -= i_alloc

        return allocations

    def name(self):
        return "Uncontrolled (FCFS)"

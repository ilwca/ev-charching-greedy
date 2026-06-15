"""
models.py
---------
Estruturas de dados centrais do estudo.

Baseado em: Hegele et al. (2023) "An Efficient Greedy Algorithm for
Real-World Large-Scale Electric Vehicle Charging", ACM e-Energy '23.

Conceitos-chave do artigo:
  - Cada EV ocupa uma fase (A, B ou C) do sistema trifásico AC
  - A corrente de carregamento é limitada por um budget global (capacity)
  - Fairness: cada EV deve receber uma parcela proporcional ao que precisa
  - Phase symmetry: a diferença de corrente entre as fases deve ser minimizada
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Configuração da infraestrutura
# ---------------------------------------------------------------------------

@dataclass
class ChargingInfrastructure:
    """
    Representa um estacionamento com N pontos de carregamento AC trifásico.

    Parâmetros (inspirados no artigo, Seção 3):
      n_stations  : número de EVSEs (pontos de carregamento)
      i_max       : corrente máxima por estação [A]
      i_budget    : budget total de corrente disponível [A] — representa
                    a capacidade do transformador local
      phases      : lista de fase (0=A, 1=B, 2=C) para cada estação,
                    atribuída ciclicamente se não fornecida
      dt_minutes  : duração de cada slot de tempo [min]
    """
    n_stations: int = 20
    i_max: float = 32.0          # A (típico para AC tipo 2 / SAE J1772)
    i_budget: float = 200.0      # A total (soma das 3 fases)
    phases: list = field(default_factory=list)
    dt_minutes: float = 5.0

    def __post_init__(self):
        if not self.phases:
            # Atribuição cíclica de fases: A, B, C, A, B, C, ...
            self.phases = [i % 3 for i in range(self.n_stations)]

    @property
    def phase_budget(self) -> float:
        """Budget por fase (assume distribuição equilibrada ideal)."""
        return self.i_budget / 3.0


# ---------------------------------------------------------------------------
# Sessão de carregamento (EV)
# ---------------------------------------------------------------------------

@dataclass
class EVSession:
    """
    Representa uma sessão de carregamento de um EV.

    Campos baseados nos dados ACN-Data (Lee et al., ACM e-Energy 2019)
    e na formulação do artigo (Seção 4, Definição 1):
      session_id   : identificador único
      station_id   : EVSE ao qual está conectado
      phase        : fase elétrica (0=A, 1=B, 2=C)
      t_arrival    : slot de chegada
      t_departure  : slot de saída (estimado pelo usuário)
      e_demand     : energia demandada [Wh]
      v_nominal    : tensão nominal [V] (tipicamente 230 V em AC monofásico)
      i_min        : corrente mínima de operação do EV [A]
      i_max        : corrente máxima aceita pelo EV [A]
      nonlinear    : se True, o EV tem comportamento não-linear (tapering)
      i_current    : corrente sendo fornecida no slot atual [A]
      e_delivered  : energia já entregue [Wh]
    """
    session_id: int
    station_id: int
    phase: int                    # 0, 1, ou 2
    t_arrival: int
    t_departure: int
    e_demand: float               # Wh
    v_nominal: float = 230.0      # V
    i_min: float = 6.0            # A (mínimo IEC 61851)
    i_max: float = 32.0           # A
    nonlinear: bool = False       # EV com comportamento não-linear
    i_current: float = 0.0        # corrente no slot corrente
    e_delivered: float = 0.0      # energia já entregue

    @property
    def e_remaining(self) -> float:
        """Energia ainda necessária [Wh]."""
        return max(0.0, self.e_demand - self.e_delivered)

    @property
    def slots_remaining(self) -> int:
        """Slots de tempo até a partida estimada."""
        return self.t_departure - self.t_arrival  # será atualizado externamente

    @property
    def i_demand(self) -> float:
        """
        Corrente ideal para satisfazer a demanda restante no tempo disponível.
        Segue a Eq. (2) do artigo: i* = E_rem / (V * T_rem * dt)
        """
        return 0.0  # calculado externamente no simulador (precisa t_current)

    def charging_efficiency(self, i_pilot: float) -> float:
        """
        Modela a resposta não-linear de alguns EVs ao sinal pilot.

        O artigo (Seção 5.1) observa que certos EVs não seguem linearmente
        o sinal de controle. Aqui, simulamos um efeito de tapering simples:
        acima de 24A, a eficiência cai para 85% (comportamento de alguns BEVs).
        """
        if not self.nonlinear:
            return 1.0
        if i_pilot <= 24.0:
            return 1.0
        else:
            # Queda gradual de eficiência acima de 24A
            return 1.0 - 0.015 * (i_pilot - 24.0)

    def actual_current(self, i_pilot: float) -> float:
        """Corrente realmente absorvida (considerando não-linearidade)."""
        eff = self.charging_efficiency(i_pilot)
        return max(0.0, min(i_pilot * eff, self.i_max))

    def deliver_energy(self, i_pilot: float, dt_hours: float):
        """Atualiza energia entregue dado o pilot signal e duração do slot."""
        i_real = self.actual_current(i_pilot)
        delta_e = i_real * self.v_nominal * dt_hours  # Wh
        self.e_delivered = min(self.e_demand, self.e_delivered + delta_e)
        self.i_current = i_real

    @property
    def is_satisfied(self) -> bool:
        """True se a demanda foi completamente atendida."""
        return self.e_delivered >= self.e_demand * 0.999  # tolerância 0.1%

    @property
    def satisfaction_ratio(self) -> float:
        """Fração da demanda entregue (0.0 a 1.0)."""
        if self.e_demand == 0:
            return 1.0
        return min(1.0, self.e_delivered / self.e_demand)

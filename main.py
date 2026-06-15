"""
main.py
-------
Script principal do estudo acadêmico.

Reproduz os experimentos centrais de:
  Hegele et al. (2023) "An Efficient Greedy Algorithm for Real-World
  Large-Scale Electric Vehicle Charging", ACM e-Energy '23.

Experimentos realizados:
  1. Simulação com 3 algoritmos (Greedy, Branch-and-Bound, Uncontrolled)
  2. Análise de escalabilidade (N = 10, 20, 50, 100 EVs)
  3. Impacto de EVs não-lineares na performance do Greedy
  4. Geração de tabelas e figuras para artigo acadêmico

Uso:
  python main.py

Resultados salvos em: results/
"""

import time
import sys
import os
import numpy as np
import pandas as pd

# Garante que o diretório do projeto está no path
sys.path.insert(0, os.path.dirname(__file__))

from models import ChargingInfrastructure
from data_generator import generate_sessions, sessions_to_dataframe
from algorithms import GreedyCharger, BranchBoundCharger, UncontrolledCharger
from simulator import run_simulation, compare_algorithms
from visualization import save_all_figures


# =============================================================================
# Configurações do experimento
# =============================================================================

INFRA = ChargingInfrastructure(
    n_stations=30,
    i_max=32.0,
    i_budget=240.0,   # 80A por fase × 3 fases
    dt_minutes=5.0,
)

N_SESSIONS   = 50     # EVs no experimento principal
N_SLOTS      = 144    # 144 × 5min = 12 horas (7h às 19h)
SEED         = 42
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")


# =============================================================================
# Experimento 1: Comparação principal dos 3 algoritmos
# =============================================================================

def experiment_main():
    print("\n" + "="*65)
    print("  EXPERIMENTO 1: Comparação de Algoritmos")
    print("="*65)

    sessions = generate_sessions(INFRA, n_sessions=N_SESSIONS,
                                 n_slots=N_SLOTS, seed=SEED)
    print(f"\n  Sessões geradas: {len(sessions)}")
    print(f"  Demanda total:   {sum(s.e_demand/1000 for s in sessions):.1f} kWh")
    print(f"  EVs não-lineares: {sum(s.nonlinear for s in sessions)}")

    chargers = [
        GreedyCharger(INFRA),
        BranchBoundCharger(INFRA, current_step=4.0),
        UncontrolledCharger(INFRA),
    ]

    results = []
    for charger in chargers:
        print(f"\n  Rodando: {charger.name()} ...")
        t0 = time.time()
        res = run_simulation(sessions, INFRA, charger, n_slots=N_SLOTS, verbose=True)
        elapsed = time.time() - t0
        print(f"  Concluído em {elapsed:.2f}s")
        results.append(res)

    # Tabela comparativa
    df_compare = compare_algorithms(results)
    print("\n" + "="*65)
    print("  RESULTADOS COMPARATIVOS")
    print("="*65)
    print(df_compare.to_string())

    # Salvar tabela CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_compare.to_csv(f"{RESULTS_DIR}/tabela_comparativa.csv")
    print(f"\n  Tabela salva em: {RESULTS_DIR}/tabela_comparativa.csv")

    return results


# =============================================================================
# Experimento 2: Escalabilidade (N EVs variando)
# =============================================================================

def experiment_scalability():
    print("\n" + "="*65)
    print("  EXPERIMENTO 2: Escalabilidade")
    print("="*65)

    ns = [10, 20, 50, 100]
    rows = []

    for n in ns:
        sessions = generate_sessions(INFRA, n_sessions=n, n_slots=N_SLOTS, seed=SEED)
        charger = GreedyCharger(INFRA)
        res = run_simulation(sessions, INFRA, charger, n_slots=N_SLOTS)

        rows.append({
            "N EVs":                n,
            "Satisfação Média (%)": f"{res['mean_satisfaction_ratio']*100:.1f}",
            "Fairness (Jain)":      f"{res['jain_fairness_index']:.4f}",
            "Deseq. Máx (A)":      f"{res['max_imbalance_A']:.2f}",
            "Tempo Médio (ms)":     f"{res['mean_algo_time_ms']:.3f}",
            "Tempo Máx (ms)":       f"{res['max_algo_time_ms']:.3f}",
        })
        print(f"  N={n:3d}: sat={res['mean_satisfaction_ratio']*100:.1f}%  "
              f"jain={res['jain_fairness_index']:.4f}  "
              f"t_max={res['max_algo_time_ms']:.3f}ms")

    df_scale = pd.DataFrame(rows).set_index("N EVs")
    df_scale.to_csv(f"{RESULTS_DIR}/escalabilidade.csv")
    print(f"\n  Tabela salva em: {RESULTS_DIR}/escalabilidade.csv")
    print(df_scale.to_string())
    return df_scale


# =============================================================================
# Experimento 3: Impacto de EVs não-lineares
# =============================================================================

def experiment_nonlinear():
    print("\n" + "="*65)
    print("  EXPERIMENTO 3: Impacto de EVs Não-Lineares")
    print("="*65)

    fractions = [0.0, 0.1, 0.2, 0.3, 0.5]
    rows = []

    for frac in fractions:
        sessions = generate_sessions(INFRA, n_sessions=N_SESSIONS,
                                     n_slots=N_SLOTS, seed=SEED,
                                     nonlinear_fraction=frac)
        charger = GreedyCharger(INFRA)
        res = run_simulation(sessions, INFRA, charger, n_slots=N_SLOTS)

        rows.append({
            "Fração Não-lineares": f"{frac*100:.0f}%",
            "Satisfação Média (%)": f"{res['mean_satisfaction_ratio']*100:.1f}",
            "Fairness (Jain)":     f"{res['jain_fairness_index']:.4f}",
            "Deseq. Máx (A)":     f"{res['max_imbalance_A']:.2f}",
        })
        print(f"  NL={frac*100:.0f}%: sat={res['mean_satisfaction_ratio']*100:.1f}%  "
              f"jain={res['jain_fairness_index']:.4f}")

    df_nl = pd.DataFrame(rows).set_index("Fração Não-lineares")
    df_nl.to_csv(f"{RESULTS_DIR}/nao_lineares.csv")
    print(f"\n  Tabela salva em: {RESULTS_DIR}/nao_lineares.csv")
    print(df_nl.to_string())
    return df_nl


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\n" + "█"*65)
    print("  Estudo Acadêmico: Smart Charging de Veículos Elétricos")
    print("  Baseado em: Hegele et al. (2023), ACM e-Energy '23")
    print("█"*65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Experimento 1: comparação principal
    main_results = experiment_main()

    # Figuras
    print("\n  Gerando figuras...")
    saved = save_all_figures(main_results, output_dir=RESULTS_DIR,
                             dt_minutes=INFRA.dt_minutes)

    # Experimento 2: escalabilidade
    df_scale = experiment_scalability()

    # Experimento 3: não-linearidade
    df_nl = experiment_nonlinear()

    print("\n" + "="*65)
    print(f"  ✅ Todos os resultados salvos em: {RESULTS_DIR}/")
    print("="*65 + "\n")

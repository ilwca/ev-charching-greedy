"""
visualization.py
----------------
Geração de figuras para o estudo acadêmico.

Figuras produzidas (inspiradas nas Figs. 3–6 do artigo):
  Fig 1 — Corrente por fase ao longo do tempo (3 algoritmos)
  Fig 2 — Desequilíbrio de fase ao longo do tempo
  Fig 3 — Distribuição do índice de satisfação por EV (histograma)
  Fig 4 — Tempo de execução por slot (boxplot)
  Fig 5 — Sessões ativas ao longo do tempo
  Fig 6 — Comparação de métricas (radar/bar)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import pandas as pd


# --- Paleta e estilo ---
COLORS = {
    "Greedy":               "#2196F3",   # azul
    "Branch-and-Bound":     "#4CAF50",   # verde
    "Uncontrolled (FCFS)":  "#F44336",   # vermelho
}
PHASE_COLORS = ["#1565C0", "#2E7D32", "#F57F17"]  # A=azul, B=verde, C=âmbar

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 120,
})


def _slot_to_hour_label(slot, dt_minutes=5, start_hour=7):
    """Converte slot em label de hora (ex: '09:30')."""
    total_min = int(start_hour * 60 + slot * dt_minutes)
    h = total_min // 60
    m = total_min % 60
    return f"{h:02d}:{m:02d}"


def fig1_phase_currents(results_list: list[dict], dt_minutes: float = 5.0,
                        save_path: str = None):
    """
    Fig 1: Corrente por fase para cada algoritmo.
    Mostra se o algoritmo mantém equilíbrio entre as três fases.
    """
    fig, axes = plt.subplots(len(results_list), 1,
                             figsize=(12, 3.5 * len(results_list)),
                             sharex=True)
    if len(results_list) == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        ts = res["ts"]
        n = len(ts["phase_A"])
        slots = np.arange(n)
        labels = [_slot_to_hour_label(s, dt_minutes) for s in slots]

        ax.fill_between(slots, ts["phase_A"], alpha=0.35, color=PHASE_COLORS[0], label="Fase A")
        ax.fill_between(slots, ts["phase_B"], alpha=0.35, color=PHASE_COLORS[1], label="Fase B")
        ax.fill_between(slots, ts["phase_C"], alpha=0.35, color=PHASE_COLORS[2], label="Fase C")
        ax.plot(slots, ts["phase_A"], color=PHASE_COLORS[0], linewidth=1.2)
        ax.plot(slots, ts["phase_B"], color=PHASE_COLORS[1], linewidth=1.2)
        ax.plot(slots, ts["phase_C"], color=PHASE_COLORS[2], linewidth=1.2)

        # Linha de budget por fase
        phase_budget = max(ts["phase_A"] + ts["phase_B"] + ts["phase_C"]) / 3 * 1.05
        ax.set_title(f"{res['algorithm']}", fontweight="bold", fontsize=11)
        ax.set_ylabel("Corrente [A]")
        ax.legend(loc="upper right", fontsize=8)

        # Ticks de hora
        tick_slots = slots[::12]
        ax.set_xticks(tick_slots)
        ax.set_xticklabels([_slot_to_hour_label(s, dt_minutes) for s in tick_slots],
                           rotation=30, fontsize=8)

    axes[-1].set_xlabel("Horário")
    fig.suptitle("Corrente por Fase ao Longo do Tempo", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def fig2_imbalance(results_list: list[dict], dt_minutes: float = 5.0,
                   save_path: str = None):
    """
    Fig 2: Desequilíbrio de fase (std das correntes A, B, C) ao longo do tempo.
    Menor = melhor simetria entre fases.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    slots = np.arange(len(results_list[0]["ts"]["imbalance"]))

    for res in results_list:
        color = COLORS.get(res["algorithm"], "gray")
        imb = res["ts"]["imbalance"]
        ax.plot(slots, imb, label=res["algorithm"], color=color, linewidth=1.8, alpha=0.85)

    tick_slots = slots[::12]
    ax.set_xticks(tick_slots)
    ax.set_xticklabels([_slot_to_hour_label(s, dt_minutes) for s in tick_slots],
                       rotation=30, fontsize=8)
    ax.set_xlabel("Horário")
    ax.set_ylabel("Desequilíbrio de Fase [A] (std)")
    ax.set_title("Desequilíbrio de Fase ao Longo do Tempo\n"
                 "(menor = melhor simetria entre fases A, B, C)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def fig3_satisfaction_hist(results_list: list[dict], save_path: str = None):
    """
    Fig 3: Histograma da satisfação de demanda por EV.
    Mostra a distribuição de quão bem cada EV foi atendido.
    """
    fig, axes = plt.subplots(1, len(results_list),
                             figsize=(5.5 * len(results_list), 4),
                             sharey=True)
    if len(results_list) == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        ratios = [ev.satisfaction_ratio * 100 for ev in res["evs"]]
        color = COLORS.get(res["algorithm"], "steelblue")
        ax.hist(ratios, bins=20, range=(0, 100), color=color, alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(ratios), color="black", linestyle="--", linewidth=1.5,
                   label=f"Média: {np.mean(ratios):.1f}%")
        ax.set_title(res["algorithm"], fontweight="bold", fontsize=10)
        ax.set_xlabel("Satisfação de Demanda (%)")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Número de EVs")
    fig.suptitle("Distribuição da Satisfação de Demanda por EV", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def fig4_algo_time(results_list: list[dict], save_path: str = None):
    """
    Fig 4: Boxplot do tempo de execução por slot para cada algoritmo.
    Demonstra a viabilidade computacional do Greedy para hardware embarcado.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    data = [res["ts"]["algo_time_ms"] for res in results_list]
    labels = [res["algorithm"] for res in results_list]
    colors = [COLORS.get(l, "gray") for l in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Tempo de Execução por Slot [ms]")
    ax.set_title("Tempo de Execução por Algoritmo\n"
                 "(escala log — viabilidade para tempo real)",
                 fontweight="bold")
    ax.set_yscale("log")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def fig5_active_evs(results_list: list[dict], dt_minutes: float = 5.0,
                    save_path: str = None):
    """
    Fig 5: Número de EVs ativos ao longo do tempo.
    (Igual para todos os algoritmos — mostra o perfil de demanda.)
    """
    fig, ax = plt.subplots(figsize=(12, 3.5))
    # Usa o primeiro resultado (n_active é igual para todos)
    ts = results_list[0]["ts"]
    slots = np.arange(len(ts["n_active"]))
    ax.fill_between(slots, ts["n_active"], alpha=0.4, color="#9C27B0")
    ax.plot(slots, ts["n_active"], color="#6A1B9A", linewidth=2)

    tick_slots = slots[::12]
    ax.set_xticks(tick_slots)
    ax.set_xticklabels([_slot_to_hour_label(s, dt_minutes) for s in tick_slots],
                       rotation=30, fontsize=8)
    ax.set_xlabel("Horário")
    ax.set_ylabel("EVs Ativos")
    ax.set_title("Perfil de Demanda: Número de EVs Conectados", fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def fig6_summary_bars(results_list: list[dict], save_path: str = None):
    """
    Fig 6: Comparação resumida das métricas principais (gráfico de barras).
    """
    metrics = {
        "Satisfação\nMédia (%)":    [r["mean_satisfaction_ratio"] * 100 for r in results_list],
        "Fairness\n(Jain × 100)":   [r["jain_fairness_index"] * 100 for r in results_list],
        "Deseq. Máx.\n(A, inv.)":   [1 / (r["max_imbalance_A"] + 0.1) * 10
                                       for r in results_list],  # invertido: maior = melhor
    }

    names = [r["algorithm"] for r in results_list]
    colors = [COLORS.get(n, "gray") for n in names]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)
        ax.set_title(metric_name, fontweight="bold", fontsize=10)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Comparação de Métricas entre Algoritmos\n"
                 "(maior = melhor em todos os eixos)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def save_all_figures(results_list: list[dict], output_dir: str = "results",
                     dt_minutes: float = 5.0):
    """Salva todas as figuras na pasta especificada."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "fig1_phase_currents.png": lambda: fig1_phase_currents(
            results_list, dt_minutes, f"{output_dir}/fig1_phase_currents.png"),
        "fig2_imbalance.png": lambda: fig2_imbalance(
            results_list, dt_minutes, f"{output_dir}/fig2_imbalance.png"),
        "fig3_satisfaction.png": lambda: fig3_satisfaction_hist(
            results_list, f"{output_dir}/fig3_satisfaction.png"),
        "fig4_algo_time.png": lambda: fig4_algo_time(
            results_list, f"{output_dir}/fig4_algo_time.png"),
        "fig5_active_evs.png": lambda: fig5_active_evs(
            results_list, dt_minutes, f"{output_dir}/fig5_active_evs.png"),
        "fig6_summary.png": lambda: fig6_summary_bars(
            results_list, f"{output_dir}/fig6_summary.png"),
    }

    saved = []
    for name, fn in paths.items():
        fig = fn()
        plt.close(fig)
        saved.append(f"{output_dir}/{name}")
        print(f"  ✓ {name}")

    return saved

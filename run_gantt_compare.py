"""
Mk01 간트 차트 비교: seed 62 (makespan=40) vs seed 0 (makespan=42)
세대별 fitness 변화 비교 포함
"""

import random
import os
import numpy as np
import plotly.graph_objects as go
from io_mk import load_mk
from ga import run_ga
from viz_plotly import plot_gantt_plotly

INSTANCE_PATH = "Dataset/Brandimarte_Data/Text/Mk01.fjs"
POP_SIZE    = 100
GENERATIONS = 100
PC, PM      = 0.7, 0.01

TARGETS = [
    (37, 40),
    (0,  42),
]

COLORS = ["#1f77b4", "#d62728"]  # 파랑(40), 빨강(42)

def plot_fitness_history(histories, save_dir):
    fig = go.Figure()

    for (seed, expected), history, color in zip(TARGETS, histories, COLORS):
        final = history[-1]
        # 처음 최솟값에 도달하는 세대 찾기
        first_gen = next(i for i, v in enumerate(history) if v == final)

        fig.add_trace(go.Scatter(
            x=list(range(len(history))),
            y=history,
            mode="lines",
            name=f"seed={seed}  best={final}",
            line=dict(color=color, width=2),
        ))
        fig.add_annotation(
            x=first_gen, y=final,
            text=f"Gen {first_gen}<br>MS={final}",
            showarrow=True, arrowhead=2,
            ax=30, ay=-30,
            font=dict(color=color, size=11),
            arrowcolor=color,
        )

    fig.update_layout(
        title="세대별 Best Makespan (Mk01)",
        xaxis_title="Generation",
        yaxis_title="Makespan",
        legend=dict(x=0.75, y=0.95),
        template="plotly_white",
        width=900, height=500,
    )

    out_path = os.path.join(save_dir, "fitness_history_compare.html")
    fig.write_html(out_path)
    print(f"[fitness history] 저장 완료: {out_path}")
    fig.show()


def main():
    inst = load_mk(INSTANCE_PATH)
    save_dir = "results/gantt_compare_Mk01"
    os.makedirs(save_dir, exist_ok=True)

    histories = []
    for seed, expected in TARGETS:
        random.seed(seed)
        np.random.seed(seed)
        schedule, mspan, history, _ = run_ga(inst, POP_SIZE, GENERATIONS, PC, PM)
        histories.append(history)

        final_gen = next(i for i, v in enumerate(history) if v == mspan)
        print(f"seed={seed:3d}  makespan={mspan}  (예상 {expected})  →  Gen {final_gen}에서 최초 달성")

        fname = f"gantt_seed{seed:03d}_ms{mspan}.html"
        plot_gantt_plotly(inst, schedule, os.path.join(save_dir, fname), show=True)

    plot_fitness_history(histories, save_dir)

if __name__ == "__main__":
    main()

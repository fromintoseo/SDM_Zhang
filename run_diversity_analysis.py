"""
Mk01 population diversity 분석
최종 makespan 그룹별 (40/41/42) 초기분포 + 세대별 unique 수 비교
"""

import random
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io_mk import load_mk
from ga import run_ga

INSTANCE_PATH = "Dataset/Brandimarte_Data/Text/Mk01.fjs"
POP_SIZE      = 100
GENERATIONS   = 100
PC, PM        = 0.7, 0.01
NUM_SEEDS     = 100

GROUP_COLORS = {40: "#1f77b4", 41: "#ff7f0e", 42: "#d62728"}
GENS = list(range(GENERATIONS))


def main():
    inst = load_mk(INSTANCE_PATH)
    save_dir = "results/diversity_analysis_Mk01"
    os.makedirs(save_dir, exist_ok=True)

    # seed별 결과 수집
    groups = {40: [], 41: [], 42: []}   # mspan -> list of div_history

    for seed in range(NUM_SEEDS):
        random.seed(seed)
        np.random.seed(seed)
        _, mspan, _, div_hist = run_ga(inst, POP_SIZE, GENERATIONS, PC, PM, track_diversity=True)
        if mspan in groups:
            groups[mspan].append(div_hist)
        print(f"seed={seed:3d} -> {mspan}", flush=True)

    # ── 차트 구성 ──────────────────────────────────────────────
    # 2행 × 3열
    # (1,1) 초기 population makespan 분포 (violin)
    # (1,2) unique_ms over gens
    # (1,3) unique_os over gens
    # (2,1) unique_ind over gens
    # (2,2) unique_ms 초반 25세대 확대
    # (2,3) unique_os 초반 25세대 확대

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "초기 Population Makespan 분포",
            "unique MS  (전체 세대)",
            "unique OS  (전체 세대)",
            "unique Individual (전체 세대)",
            "unique MS  (초반 25세대)",
            "unique OS  (초반 25세대)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    for ms_val, hist_list in sorted(groups.items()):
        if not hist_list:
            continue
        color = GROUP_COLORS[ms_val]
        label = f"MS={ms_val}  (n={len(hist_list)})"

        # ── (1,1) 초기 분포 violin ─────────────────────────────
        init_fits = [f for h in hist_list for f in h[0]['all_fitnesses']]
        fig.add_trace(go.Violin(
            y=init_fits,
            name=label,
            box_visible=True,
            meanline_visible=True,
            line_color=color,
            fillcolor=color,
            opacity=0.55,
            legendgroup=str(ms_val),
            showlegend=True,
        ), row=1, col=1)

        # ── 세대별 평균 diversity ─────────────────────────────
        def mean_metric(key):
            curves = [[d[key] for d in h] for h in hist_list]
            return np.mean(curves, axis=0)

        u_ms  = mean_metric("unique_ms")
        u_os  = mean_metric("unique_os")
        u_ind = mean_metric("unique_ind")

        shared = dict(line=dict(color=color, width=2), legendgroup=str(ms_val), showlegend=False)

        # (1,2) unique_ms 전체
        fig.add_trace(go.Scatter(x=GENS, y=u_ms,  name=label, **shared), row=1, col=2)
        # (1,3) unique_os 전체
        fig.add_trace(go.Scatter(x=GENS, y=u_os,  name=label, **shared), row=1, col=3)
        # (2,1) unique_ind 전체
        fig.add_trace(go.Scatter(x=GENS, y=u_ind, name=label, **shared), row=2, col=1)
        # (2,2) unique_ms 초반 25
        fig.add_trace(go.Scatter(x=GENS[:25], y=u_ms[:25],  name=label, **shared), row=2, col=2)
        # (2,3) unique_os 초반 25
        fig.add_trace(go.Scatter(x=GENS[:25], y=u_os[:25],  name=label, **shared), row=2, col=3)

    # 5, 10세대 수직선 (초반 확대 패널)
    for xval, xtxt in [(5, "Gen 5"), (10, "Gen 10")]:
        for rc in [(2, 2), (2, 3)]:
            fig.add_vline(x=xval, line_dash="dot", line_color="gray",
                          annotation_text=xtxt, annotation_position="top",
                          row=rc[0], col=rc[1])

    fig.update_layout(
        title="Population Diversity 분석 — Mk01 (100 seeds, 그룹별 평균)",
        template="plotly_white",
        width=1200, height=750,
        legend=dict(title="최종 Makespan 그룹", x=1.01, y=0.95),
    )
    for r in [1, 2]:
        for c in [2, 3]:
            fig.update_xaxes(title_text="Generation", row=r, col=c)
            fig.update_yaxes(title_text="Unique count", row=r, col=c)
    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_yaxes(title_text="Unique count", row=2, col=1)
    fig.update_yaxes(title_text="Makespan", row=1, col=1)

    out_path = os.path.join(save_dir, "diversity_analysis.html")
    fig.write_html(out_path)
    print(f"\n저장: {out_path}")
    fig.show()


if __name__ == "__main__":
    main()

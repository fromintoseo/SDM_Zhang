"""
Mk06 seed 분석 스크립트 - 100개 seed(0~99)로 각 1회 실행하여
논문 결과 대비 성능 분포를 확인 (체리피킹 여부 점검)
"""

import random
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from io_mk import load_mk
from ga import run_ga
from decoder import decode_calc, makespan as calc_makespan

# Mk06 논문 기준값
PAPER_BEST = 58
PAPER_AVG  = 58.0

INSTANCE_PATH = "Dataset/Brandimarte_Data/Text/Mk06.fjs"
POP_SIZE      = 200
GENERATIONS   = 100
PC            = 0.7
PM            = 0.01
NUM_SEEDS     = 100   # seed 0 ~ 99


def run_once_with_seed(inst, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    _, mspan, history, _ = run_ga(inst, POP_SIZE, GENERATIONS, PC, PM)
    return mspan, history


def main():
    if not os.path.exists(INSTANCE_PATH):
        print(f"파일 없음: {INSTANCE_PATH}")
        return

    inst = load_mk(INSTANCE_PATH)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", f"seed_analysis_Mk06_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    records = []
    all_histories = []

    print(f"Mk06 seed 분석 시작 (seed 0 ~ {NUM_SEEDS - 1})")
    print(f"논문 Best={PAPER_BEST}, 논문 Avg={PAPER_AVG}")
    print("=" * 60)

    t0 = time.time()
    for seed in range(NUM_SEEDS):
        mspan, history = run_once_with_seed(inst, seed)
        all_histories.append(history)
        gap_best = round((mspan - PAPER_BEST) / PAPER_BEST * 100, 2)
        gap_avg  = round((mspan - PAPER_AVG)  / PAPER_AVG  * 100, 2)
        records.append({
            "seed":        seed,
            "makespan":    mspan,
            "gap_vs_best%": gap_best,
            "gap_vs_avg%":  gap_avg,
            "paper_best":  PAPER_BEST,
            "paper_avg":   PAPER_AVG,
        })
        print(f"seed {seed:3d} -> makespan={mspan:4d}  "
              f"gap_best={gap_best:+.1f}%  gap_avg={gap_avg:+.1f}%")

    elapsed = time.time() - t0

    # --- 통계 ---
    makespans = [r["makespan"] for r in records]
    my_best   = min(makespans)
    my_worst  = max(makespans)
    my_mean   = np.mean(makespans)
    my_std    = np.std(makespans)
    beat_best = sum(1 for m in makespans if m <= PAPER_BEST)
    beat_avg  = sum(1 for m in makespans if m <= PAPER_AVG)

    print("\n" + "=" * 60)
    print(f"[결과 요약] 총 {NUM_SEEDS}회 실행, 소요 {elapsed:.1f}s")
    print(f"  내 GA Best  : {my_best}   (논문 Best={PAPER_BEST})")
    print(f"  내 GA Worst : {my_worst}")
    print(f"  내 GA Mean  : {my_mean:.2f}  Std={my_std:.2f}")
    print(f"  논문 Best 이하 달성 횟수 : {beat_best}/{NUM_SEEDS}")
    print(f"  논문 Avg  이하 달성 횟수 : {beat_avg}/{NUM_SEEDS}")
    print("=" * 60)

    # --- CSV 저장 ---
    df = pd.DataFrame(records)
    csv_path = os.path.join(save_dir, "seed_analysis_Mk06.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장: {csv_path}")

    # --- 히스토그램 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(makespans, bins=15, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(PAPER_BEST, color="red",    linestyle="--", linewidth=1.5, label=f"Paper Best={PAPER_BEST}")
    ax.axvline(PAPER_AVG,  color="orange", linestyle="--", linewidth=1.5, label=f"Paper Avg={PAPER_AVG}")
    ax.axvline(my_mean,    color="green",  linestyle="-",  linewidth=1.5, label=f"My Mean={my_mean:.1f}")
    ax.set_title("Makespan Distribution across 100 Seeds (Mk06)", fontsize=12)
    ax.set_xlabel("Makespan")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    # 수렴 곡선 (평균 ± std)
    ax2 = axes[1]
    histories = np.array(all_histories)
    mean_h = np.mean(histories, axis=0)
    std_h  = np.std(histories, axis=0)
    gens   = np.arange(1, GENERATIONS + 1)
    ax2.fill_between(gens, mean_h - std_h, mean_h + std_h,
                     color="steelblue", alpha=0.2, label="±1 std")
    ax2.plot(gens, mean_h, color="steelblue", linewidth=2, label="Mean Best Makespan")
    ax2.axhline(PAPER_BEST, color="red",    linestyle="--", linewidth=1.5, label=f"Paper Best={PAPER_BEST}")
    ax2.axhline(PAPER_AVG,  color="orange", linestyle="--", linewidth=1.5, label=f"Paper Avg={PAPER_AVG}")
    ax2.set_title("Convergence across 100 Seeds (Mk06)", fontsize=12)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Makespan")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    png_path = os.path.join(save_dir, "seed_analysis_Mk06.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"그래프 저장: {png_path}")

    # --- 개별 수렴 곡선 (plotly) ---
    COLOR_MAP = {58: "#1f77b4", 59: "#ff7f0e", 60: "#d62728"}
    DEFAULT_COLOR = "#aaaaaa"

    fig2 = go.Figure()
    shown = set()
    for rec, history in zip(records, all_histories):
        seed  = rec["seed"]
        mspan = rec["makespan"]
        color = COLOR_MAP.get(mspan, DEFAULT_COLOR)
        first_gen = next(i for i, v in enumerate(history) if v == mspan)
        show_leg  = mspan not in shown
        shown.add(mspan)

        fig2.add_trace(go.Scatter(
            x=list(range(len(history))),
            y=history,
            mode="lines",
            name=f"MS={mspan}",
            legendgroup=str(mspan),
            showlegend=show_leg,
            line=dict(color=color, width=1.2),
            opacity=0.55,
            hovertemplate=f"seed={seed}  MS={mspan}<br>Gen %{{x}}  fitness %{{y}}<extra></extra>",
        ))

    fig2.add_hline(y=PAPER_BEST, line_dash="dash", line_color="black",
                   annotation_text=f"Paper Best={PAPER_BEST}", annotation_position="bottom right")

    fig2.update_layout(
        title=f"세대별 Best Makespan – 전체 {NUM_SEEDS}개 시드 (Mk06)",
        xaxis_title="Generation",
        yaxis_title="Makespan",
        template="plotly_white",
        width=1000, height=550,
        legend_title="최종 Makespan",
    )

    conv_path = os.path.join(save_dir, "convergence_all_seeds.html")
    fig2.write_html(conv_path)
    print(f"수렴 곡선 저장: {conv_path}")
    fig2.show()


if __name__ == "__main__":
    main()

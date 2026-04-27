#4
# run_stage.py
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io_mk import load_mk
from ga import run_ga, generate_initial_population
from decoder import decode_calc, makespan
from viz_plotly import plot_gantt_plotly

# 논문(Zhang, 2011) Table 3
PAPER_REFERENCE = {
    "Mk01": [36, 42, 40, 40.0],
    "Mk02": [24, 32, 26, 26.0],
    "Mk03": [204, 211, 204, 204.0],
    "Mk04": [48, 81, 60, 60.0],
    "Mk05": [168, 186, 173, 173.0],
    "Mk06": [33, 86, 58, 58.0],
    "Mk07": [133, 157, 144, 145.0],
    "Mk08": [523, 523, 523, 523.0],
    "Mk09": [299, 369, 307, 307.0],
    "Mk10": [165, 296, 198, 199.0]
}

POPSIZE_MAP = {
    "Mk01": 100, "Mk02": 300, "Mk03": 50, "Mk04": 100, "Mk05": 200,
    "Mk06": 200, "Mk07": 200, "Mk08": 50, "Mk09": 300, "Mk10": 300
}


def plot_convergence_matplotlib(all_histories, dataset_name, out_png):
    histories_array = np.array(all_histories)
    mean_history = np.mean(histories_array, axis=0)
    std_history = np.std(histories_array, axis=0)
    n_runs = len(all_histories)
    ci_margin = 1.96 * (std_history / np.sqrt(n_runs))
    generations = np.arange(1, len(mean_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(generations, mean_history - ci_margin, mean_history + ci_margin,
                     color='teal', alpha=0.2, label='95% Confidence Interval')
    plt.plot(generations, mean_history, color='teal', linewidth=2, label='Average Best Makespan')
    plt.title(f"GA Convergence Statistics ({n_runs} runs) - {dataset_name}", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Makespan", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def run_experiment(instance_path, base_save_dir, num_runs=5):
    dataset_name = os.path.splitext(os.path.basename(instance_path))[0]
    target_pop_size = POPSIZE_MAP.get(dataset_name, 100)
    ref_data = PAPER_REFERENCE.get(dataset_name, ["-", "-", "-", "-"])

    print(f"\n{'=' * 60}")
    print(f" [실험 시작] {dataset_name} (Pop:{target_pop_size}, Runs:{num_runs})")
    print(f"{'=' * 60}")

    try:
        inst = load_mk(instance_path)
    except Exception as e:
        return None

    save_dir = os.path.join(base_save_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # 1. 초기해 20개 통계
    init_pop_20 = generate_initial_population(inst, pop_size=20)
    init_makespans = [makespan(decode_calc(inst, os_arr, ms_arr)) for ms_arr, os_arr in init_pop_20]
    init_20_best = min(init_makespans)
    init_20_avg = np.mean(init_makespans)

    # 2. GA 반복 실행 및 통계 수집
    all_histories = []
    ga_final_makespans = []
    best_results = []

    start_time = time.time()
    for i in range(num_runs):
        schedule, mspan, history = run_ga(inst, target_pop_size, 100, 0.7, 0.01)
        all_histories.append(history)
        ga_final_makespans.append(mspan)
        best_results.append((mspan, schedule))
        print(f"  👉 Run {i + 1:02d}/{num_runs} 완료 - Makespan: {mspan}")
    total_duration = time.time() - start_time
    avg_time_per_run = total_duration / num_runs

    # 통계 계산
    ga_best = min(ga_final_makespans)
    ga_avg = np.mean(ga_final_makespans)
    ga_std = np.std(ga_final_makespans)
    ci_margin = 1.96 * (ga_std / np.sqrt(num_runs))

    # 최적 스케줄 선택
    best_results.sort(key=lambda x: x[0])
    overall_best_schedule = best_results[0][1]

    # 시각화 저장
    conv_path = os.path.join(save_dir, f"{dataset_name}_convergence.png")
    plot_convergence_matplotlib(all_histories, dataset_name, conv_path)
    gantt_path = os.path.join(save_dir, f"{dataset_name}_best_gantt.html")
    plot_gantt_plotly(inst, overall_best_schedule, gantt_path)

    return {
        "Problem": dataset_name,
        "Paper_LB": ref_data[0],
        "Paper_UB": ref_data[1],
        "Paper_Best": ref_data[2],
        "Paper_Avg": ref_data[3],
        "My_GA_Best": ga_best,
        "My_GA_Avg": round(ga_avg, 2),
        "CI_Lower": round(ga_avg - ci_margin, 2),
        "CI_Upper": round(ga_avg + ci_margin, 2),
        "Avg_Time(s)": round(avg_time_per_run, 2),
        "Init_20_Best": init_20_best,
        "Init_20_Avg": round(init_20_avg, 2)
    }


def main():
    test_instances = [
        f"Dataset/Brandimarte_Data/Text/Mk{str(i).zfill(2)}.fjs" for i in range(1, 11)
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join("results", f"Zhang_Final_Exp_{timestamp}")
    os.makedirs(base_save_dir, exist_ok=True)

    summary_results = []

    for instance_path in test_instances:
        if os.path.exists(instance_path):
            result = run_experiment(instance_path, base_save_dir, num_runs=5)
            if result:
                summary_results.append(result)

    if summary_results:
        df = pd.DataFrame(summary_results)

        # CSV 파일로 저장
        csv_file = os.path.join(base_save_dir, "experiment_summary.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        # 전체 요약 출력
        print("-" * 100)
        print(df.to_string(index=False))
        print("-" * 100)

        print(f"\nCSV로 저장됨")

if __name__ == "__main__":
    main()


'''
# run_stage_barnes.py
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io_mk import load_mk
from ga import run_ga, generate_initial_population
from decoder import decode_calc, makespan
from viz_plotly import plot_gantt_plotly

# 논문(Zhang, 2011) Table 4 (Results of BCdata) 명시 정보
PAPER_REFERENCE = {
    "mt10c1": [655, 927, 927, 928.0],
    "mt10cc": [655, 914, 910, 910.0],
    "mt10x": [655, 929, 918, 918.0],
    "mt10xx": [655, 929, 918, 918.0],
    "mt10xxx": [655, 936, 918, 918.0],
    "mt10xy": [655, 913, 905, 906.0],
    "mt10xyz": [655, 849, 847, 847.0],
    "setb4c9": [857, 924, 914, 914.0],
    "setb4cc": [857, 909, 909, 910.0],
    "setb4x": [846, 937, 925, 925.0],
    "setb4xx": [847, 930, 925, 925.0],
    "setb4xxx": [846, 925, 925, 925.0],
    "setb4xy": [845, 924, 916, 916.0],
    "setb4xyz": [838, 914, 905, 908.1],
    "seti5c12": [1027, 1185, 1174, 1174.0],
    "seti5cc": [955, 1136, 1136, 1136.2],
    "seti5x": [955, 1218, 1209, 1209.0],
    "seti5xx": [955, 1204, 1204, 1204.0],
    "seti5xxx": [955, 1213, 1204, 1204.0],
    "seti5xy": [955, 1148, 1136, 1136.3],
    "seti5xyz": [955, 1127, 1125, 1126.5]
}

POPSIZE_MAP = {
    "mt10c1": 200, "mt10cc": 200, "mt10x": 1000, "mt10xx": 1000, "mt10xxx": 1000,
    "mt10xy": 300, "mt10xyz": 1000, "setb4c9": 1000, "setb4cc": 1000, "setb4x": 200,
    "setb4xx": 300, "setb4xxx": 1000, "setb4xy": 1000, "setb4xyz": 1000, "seti5c12": 1000,
    "seti5cc": 1000, "seti5x": 1000, "seti5xx": 1000, "seti5xxx": 1000, "seti5xy": 1000,
    "seti5xyz": 1000
}


def plot_convergence_matplotlib(all_histories, dataset_name, out_png):
    histories_array = np.array(all_histories)
    mean_history = np.mean(histories_array, axis=0)
    std_history = np.std(histories_array, axis=0)
    n_runs = len(all_histories)
    ci_margin = 1.96 * (std_history / np.sqrt(n_runs))
    generations = np.arange(1, len(mean_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(generations, mean_history - ci_margin, mean_history + ci_margin,
                     color='teal', alpha=0.2, label='95% Confidence Interval')
    plt.plot(generations, mean_history, color='teal', linewidth=2, label='Average Best Makespan')
    plt.title(f"GA Convergence Statistics ({n_runs} runs) - {dataset_name}", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Makespan", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def run_experiment(instance_path, base_save_dir, num_runs=5):
    dataset_name = os.path.splitext(os.path.basename(instance_path))[0]
    target_pop_size = POPSIZE_MAP.get(dataset_name, 100)
    ref_data = PAPER_REFERENCE.get(dataset_name, ["-", "-", "-", "-"])

    print(f"\n{'=' * 60}")
    print(f"🚀 [실험 시작] {dataset_name} (Pop:{target_pop_size}, Runs:{num_runs})")
    print(f"{'=' * 60}")

    try:
        inst = load_mk(instance_path)
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return None

    save_dir = os.path.join(base_save_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    init_pop_20 = generate_initial_population(inst, pop_size=20)
    init_makespans = [makespan(decode_calc(inst, os_arr, ms_arr)) for ms_arr, os_arr in init_pop_20]
    init_20_best = min(init_makespans)
    init_20_avg = np.mean(init_makespans)

    all_histories = []
    ga_final_makespans = []
    best_results = []

    start_time = time.time()
    for i in range(num_runs):
        schedule, mspan, history = run_ga(inst, target_pop_size, 100, 0.7, 0.01)
        all_histories.append(history)
        ga_final_makespans.append(mspan)
        best_results.append((mspan, schedule))
        print(f"  👉 Run {i + 1:02d}/{num_runs} 완료 - Makespan: {mspan}")
    total_duration = time.time() - start_time
    avg_time_per_run = total_duration / num_runs

    ga_best = min(ga_final_makespans)
    ga_avg = np.mean(ga_final_makespans)
    ga_std = np.std(ga_final_makespans)
    ci_margin = 1.96 * (ga_std / np.sqrt(num_runs))

    best_results.sort(key=lambda x: x[0])
    overall_best_schedule = best_results[0][1]

    conv_path = os.path.join(save_dir, f"{dataset_name}_convergence.png")
    plot_convergence_matplotlib(all_histories, dataset_name, conv_path)
    gantt_path = os.path.join(save_dir, f"{dataset_name}_best_gantt.html")
    plot_gantt_plotly(inst, overall_best_schedule, gantt_path)

    return {
        "Problem": dataset_name,
        "Paper_LB": ref_data[0],
        "Paper_UB": ref_data[1],
        "Paper_Best": ref_data[2],
        "Paper_Avg": ref_data[3],
        "My_GA_Best": ga_best,
        "My_GA_Avg": round(ga_avg, 2),
        "CI_Lower": round(ga_avg - ci_margin, 2),
        "CI_Upper": round(ga_avg + ci_margin, 2),
        "Avg_Time(s)": round(avg_time_per_run, 2),
        "Init_20_Best": init_20_best,
        "Init_20_Avg": round(init_20_avg, 2)
    }


def main():
    test_instances = [
        "Dataset/Barnes/Text/mt10c1.fjs", "Dataset/Barnes/Text/mt10cc.fjs",
        "Dataset/Barnes/Text/mt10x.fjs", "Dataset/Barnes/Text/mt10xx.fjs",
        "Dataset/Barnes/Text/mt10xxx.fjs", "Dataset/Barnes/Text/mt10xy.fjs",
        "Dataset/Barnes/Text/mt10xyz.fjs", "Dataset/Barnes/Text/setb4c9.fjs",
        "Dataset/Barnes/Text/setb4cc.fjs", "Dataset/Barnes/Text/setb4x.fjs",
        "Dataset/Barnes/Text/setb4xx.fjs", "Dataset/Barnes/Text/setb4xxx.fjs",
        "Dataset/Barnes/Text/setb4xy.fjs", "Dataset/Barnes/Text/setb4xyz.fjs",
        "Dataset/Barnes/Text/seti5c12.fjs", "Dataset/Barnes/Text/seti5cc.fjs",
        "Dataset/Barnes/Text/seti5x.fjs", "Dataset/Barnes/Text/seti5xx.fjs",
        "Dataset/Barnes/Text/seti5xxx.fjs", "Dataset/Barnes/Text/seti5xy.fjs",
        "Dataset/Barnes/Text/seti5xyz.fjs"
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join("results", f"Zhang_Barnes_Exp_{timestamp}")
    os.makedirs(base_save_dir, exist_ok=True)

    summary_results = []

    for instance_path in test_instances:
        if os.path.exists(instance_path):
            result = run_experiment(instance_path, base_save_dir, num_runs=5)
            if result:
                summary_results.append(result)

    if summary_results:
        df = pd.DataFrame(summary_results)
        csv_file = os.path.join(base_save_dir, "barnes_experiment_summary.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        print("\n📊 [Barnes 전체 실험 결과 요약 표]")
        print("-" * 100)
        print(df.to_string(index=False))
        print("-" * 100)
        print(f"\n✅ 전체 실험 통계가 CSV로 저장되었습니다: {csv_file}")
    else:
        print("\n❌ 실험 결과가 없습니다. 파일 경로를 다시 확인해주세요.")


if __name__ == "__main__":
    main()



'''

'''
# run_stage_dauzere.py
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io_mk import load_mk
from ga import run_ga, generate_initial_population
from decoder import decode_calc, makespan
from viz_plotly import plot_gantt_plotly

# 논문(Zhang, 2011) Table 5 (Results of DPdata) 명시 정보
PAPER_REFERENCE = {
    "01a": [2505, 2530, 2516, 2518.0],
    "02a": [2228, 2244, 2231, 2231.0],
    "03a": [2228, 2235, 2232, 2232.3],
    "04a": [2503, 2565, 2515, 2515.0],
    "05a": [2189, 2229, 2208, 2210.0],
    "06a": [2162, 2216, 2174, 2175.0],
    "07a": [2187, 2408, 2217, 2218.4],
    "08a": [2061, 2093, 2073, 2073.0],
    "09a": [2061, 2074, 2066, 2066.0],
    "10a": [2178, 2362, 2189, 2191.0],
    "11a": [2017, 2078, 2063, 2065.0],
    "12a": [1969, 2047, 2019, 2022.0],
    "13a": [2161, 2302, 2194, 2194.0],
    "14a": [2161, 2183, 2167, 2168.2],
    "15a": [2161, 2171, 2165, 2166.0],
    "16a": [2148, 2301, 2211, 2212.6],
    "17a": [2088, 2168, 2109, 2110.0],
    "18a": [2057, 2139, 2089, 2089.0]
}

POPSIZE_MAP = {
    "01a": 300, "02a": 1000, "03a": 1000, "04a": 300, "05a": 300,
    "06a": 500, "07a": 200, "08a": 500, "09a": 300, "10a": 500,
    "11a": 300, "12a": 1000, "13a": 500, "14a": 200, "15a": 1000,
    "16a": 500, "17a": 200, "18a": 300
}

def plot_convergence_matplotlib(all_histories, dataset_name, out_png):
    histories_array = np.array(all_histories)
    mean_history = np.mean(histories_array, axis=0)
    std_history = np.std(histories_array, axis=0)
    n_runs = len(all_histories)
    ci_margin = 1.96 * (std_history / np.sqrt(n_runs))
    generations = np.arange(1, len(mean_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(generations, mean_history - ci_margin, mean_history + ci_margin,
                     color='teal', alpha=0.2, label='95% Confidence Interval')
    plt.plot(generations, mean_history, color='teal', linewidth=2, label='Average Best Makespan')
    plt.title(f"GA Convergence Statistics ({n_runs} runs) - {dataset_name}", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Makespan", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def run_experiment(instance_path, base_save_dir, num_runs=5):
    dataset_name = os.path.splitext(os.path.basename(instance_path))[0]
    target_pop_size = POPSIZE_MAP.get(dataset_name, 100)
    ref_data = PAPER_REFERENCE.get(dataset_name, ["-", "-", "-", "-"])

    print(f"\n{'=' * 60}")
    print(f"🚀 [실험 시작] {dataset_name} (Pop:{target_pop_size}, Runs:{num_runs})")
    print(f"{'=' * 60}")

    try:
        inst = load_mk(instance_path)
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return None

    save_dir = os.path.join(base_save_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    init_pop_20 = generate_initial_population(inst, pop_size=20)
    init_makespans = [makespan(decode_calc(inst, os_arr, ms_arr)) for ms_arr, os_arr in init_pop_20]
    init_20_best = min(init_makespans)
    init_20_avg = np.mean(init_makespans)

    all_histories = []
    ga_final_makespans = []
    best_results = []

    start_time = time.time()
    for i in range(num_runs):
        schedule, mspan, history = run_ga(inst, target_pop_size, 100, 0.7, 0.01)
        all_histories.append(history)
        ga_final_makespans.append(mspan)
        best_results.append((mspan, schedule))
        print(f"  👉 Run {i + 1:02d}/{num_runs} 완료 - Makespan: {mspan}")
    total_duration = time.time() - start_time
    avg_time_per_run = total_duration / num_runs

    ga_best = min(ga_final_makespans)
    ga_avg = np.mean(ga_final_makespans)
    ga_std = np.std(ga_final_makespans)
    ci_margin = 1.96 * (ga_std / np.sqrt(num_runs))

    best_results.sort(key=lambda x: x[0])
    overall_best_schedule = best_results[0][1]

    conv_path = os.path.join(save_dir, f"{dataset_name}_convergence.png")
    plot_convergence_matplotlib(all_histories, dataset_name, conv_path)
    gantt_path = os.path.join(save_dir, f"{dataset_name}_best_gantt.html")
    plot_gantt_plotly(inst, overall_best_schedule, gantt_path)

    return {
        "Problem": dataset_name,
        "Paper_LB": ref_data[0],
        "Paper_UB": ref_data[1],
        "Paper_Best": ref_data[2],
        "Paper_Avg": ref_data[3],
        "My_GA_Best": ga_best,
        "My_GA_Avg": round(ga_avg, 2),
        "CI_Lower": round(ga_avg - ci_margin, 2),
        "CI_Upper": round(ga_avg + ci_margin, 2),
        "Avg_Time(s)": round(avg_time_per_run, 2),
        "Init_20_Best": init_20_best,
        "Init_20_Avg": round(init_20_avg, 2)
    }


def main():
    test_instances = [
        "Dataset/Dauzere_Data/Text/01a.fjs", "Dataset/Dauzere_Data/Text/02a.fjs",
        "Dataset/Dauzere_Data/Text/03a.fjs", "Dataset/Dauzere_Data/Text/04a.fjs",
        "Dataset/Dauzere_Data/Text/05a.fjs", "Dataset/Dauzere_Data/Text/06a.fjs",
        "Dataset/Dauzere_Data/Text/07a.fjs", "Dataset/Dauzere_Data/Text/08a.fjs",
        "Dataset/Dauzere_Data/Text/09a.fjs", "Dataset/Dauzere_Data/Text/10a.fjs",
        "Dataset/Dauzere_Data/Text/11a.fjs", "Dataset/Dauzere_Data/Text/12a.fjs",
        "Dataset/Dauzere_Data/Text/13a.fjs", "Dataset/Dauzere_Data/Text/14a.fjs",
        "Dataset/Dauzere_Data/Text/15a.fjs", "Dataset/Dauzere_Data/Text/16a.fjs",
        "Dataset/Dauzere_Data/Text/17a.fjs", "Dataset/Dauzere_Data/Text/18a.fjs"
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join("results", f"Zhang_Dauzere_Exp_{timestamp}")
    os.makedirs(base_save_dir, exist_ok=True)

    summary_results = []

    for instance_path in test_instances:
        if os.path.exists(instance_path):
            result = run_experiment(instance_path, base_save_dir, num_runs=5)
            if result:
                summary_results.append(result)

    if summary_results:
        df = pd.DataFrame(summary_results)
        csv_file = os.path.join(base_save_dir, "dauzere_experiment_summary.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        print("\n📊 [Dauzere 전체 실험 결과 요약 표]")
        print("-" * 100)
        print(df.to_string(index=False))
        print("-" * 100)
        print(f"\n✅ 전체 실험 통계가 CSV로 저장되었습니다: {csv_file}")
    else:
        print("\n❌ 실험 결과가 없습니다. 파일 경로를 다시 확인해주세요.")


if __name__ == "__main__":
    main()
'''
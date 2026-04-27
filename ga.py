import random
from decoder import decode_calc, makespan
# ㅇ
def init_GS(instance):
    # Global Selection: 전체 누적 시간을 고려하여 기계 할당
    MS = [0] * instance.total_operations
    time_array = [0] * len(instance.machines)

    jobs_seq = list(range(len(instance.jobs))) # Job 순서 랜덤하게 결정
    random.shuffle(jobs_seq) # Job 순서 랜덤하게 결정
    for j in jobs_seq:
        job = instance.jobs[j]
        for op in job.operations:
            best_alt_idx = 0
            min_time = float('inf')

            for alt_idx, (machine_id, processing_time) in enumerate(op.alternatives):
                estimated_time = time_array[machine_id] + processing_time
                if estimated_time < min_time:
                    min_time = estimated_time
                    best_alt_idx = alt_idx
                elif estimated_time == min_time:
                    if random.random() < 0.5:
                        best_alt_idx = alt_idx

            # [개선점] offset을 이용하여 즉시 위치 배정
            flat_op_idx = instance.job_index[j] + op.op_id
            MS[flat_op_idx] = best_alt_idx

            selected_machine_id, selected_ptime = op.alternatives[best_alt_idx]
            time_array[selected_machine_id] += selected_ptime

    return MS


def init_LS(instance):
    """Local Selection: 개별 Job마다 기계 누적 시간을 리셋하여 할당 (MS)"""
    MS = [0] * instance.total_operations

    jobs_seq = list(range(len(instance.jobs)))
    random.shuffle(jobs_seq)

    for j in jobs_seq:
        time_array = [0] * len(instance.machines)  # Job 바뀔 때마다 리셋
        job = instance.jobs[j]
        for op in job.operations:
            best_alt_idx = 0
            min_time = float('inf')
            for alt_idx, (machine_id, processing_time) in enumerate(op.alternatives):
                estimated_time = time_array[machine_id] + processing_time
                if estimated_time < min_time:
                    min_time = estimated_time
                    best_alt_idx = alt_idx

            flat_op_idx = instance.job_index[j] + op.op_id
            MS[flat_op_idx] = best_alt_idx

            selected_machine_id, selected_ptime = op.alternatives[best_alt_idx]
            time_array[selected_machine_id] += selected_ptime

    return MS


def init_RS(instance):
    MS = []
    for job in instance.jobs:
        for op in job.operations:
            MS.append(random.randint(0, len(op.alternatives) - 1))
    return MS


def init_OS_random(instance):
    OS = []
    for job in instance.jobs:
        OS.extend([job.job_id] * len(job.operations))
    random.shuffle(OS)
    return OS


def generate_initial_population(instance, pop_size):
    population = []
    num_gs = int(pop_size * 0.6)
    num_ls = int(pop_size * 0.3)
    num_rs = pop_size - num_gs - num_ls

    for _ in range(num_gs): population.append((init_GS(instance), init_OS_random(instance)))
    for _ in range(num_ls): population.append((init_LS(instance), init_OS_random(instance)))
    for _ in range(num_rs): population.append((init_RS(instance), init_OS_random(instance)))

    return population


def select_tournament(population, fitnesses, k=3):
    selected_indices = random.sample(range(len(population)), k)

    # 선택된 인덱스 중 fitness가 가장 좋은(낮은) 인덱스 찾기
    best_idx = min(selected_indices, key=lambda idx: fitnesses[idx])

    best_ms, best_os = population[best_idx]
    return list(best_ms), list(best_os)

def crossover_MS(p1_ms, p2_ms):
    c1_ms, c2_ms = list(p1_ms), list(p2_ms)
    if random.random() < 0.5:
        pt1, pt2 = sorted(random.sample(range(len(p1_ms)), 2))
        c1_ms[pt1:pt2], c2_ms[pt1:pt2] = c2_ms[pt1:pt2], c1_ms[pt1:pt2]
    else:
        for i in range(len(p1_ms)):
            if random.random() < 0.5:
                c1_ms[i], c2_ms[i] = c2_ms[i], c1_ms[i]
    return c1_ms, c2_ms


def crossover_OS_POX(p1_os, p2_os, num_jobs):
    jobs = list(range(num_jobs))
    random.shuffle(jobs)
    js1 = set(jobs[:len(jobs) // 2])

    c1_os = [-1] * len(p1_os)
    c2_os = [-1] * len(p2_os)

    for i in range(len(p1_os)):
        if p1_os[i] in js1: c1_os[i] = p1_os[i]
        if p2_os[i] in js1: c2_os[i] = p2_os[i]

    p2_remain = [x for x in p2_os if x not in js1]
    p1_remain = [x for x in p1_os if x not in js1]

    idx1, idx2 = 0, 0
    for i in range(len(c1_os)):
        if c1_os[i] == -1:
            c1_os[i] = p2_remain[idx1]
            idx1 += 1
        if c2_os[i] == -1:
            c2_os[i] = p1_remain[idx2]
            idx2 += 1

    return c1_os, c2_os


def mutate_MS(ms, instance):
    mut_idx = random.randint(0, len(ms) - 1)

    # [수정됨] 매번 이중 for문으로 리스트를 생성하지 않고, 미리 생성된 flat_ops를 O(1)로 가져옴
    target_op = instance.ops_instances[mut_idx]

    best_alt_idx = 0
    min_time = float('inf')
    for alt_idx, (machine_id, processing_time) in enumerate(target_op.alternatives):
        if processing_time < min_time:
            min_time = processing_time
            best_alt_idx = alt_idx

    ms[mut_idx] = best_alt_idx
    return ms


def mutate_OS(os):
    idx1, idx2 = random.sample(range(len(os)), 2)
    os[idx1], os[idx2] = os[idx2], os[idx1]
    return os


def run_ga(instance, pop_size=100, generations=100, pc=0.7, pm=0.01):
    population = generate_initial_population(instance, pop_size)

    best_makespan_history = []
    global_best_makespan = float('inf')
    global_best_schedule = None

    for gen in range(generations):
        fitnesses = [makespan(decode_calc(instance, os, ms)) for ms, os in population]

        current_best = min(fitnesses)
        best_idx = fitnesses.index(current_best)
        if current_best < global_best_makespan:
            global_best_makespan = current_best
            global_best_schedule = decode_calc(instance, population[best_idx][1], population[best_idx][0])

        best_makespan_history.append(global_best_makespan)

        new_population = []

        best_ms, best_os = population[best_idx]
        new_population.append((list(best_ms), list(best_os)))

        while len(new_population) < pop_size:
            p1_ms, p1_os = select_tournament(population, fitnesses)
            p2_ms, p2_os = select_tournament(population, fitnesses)

            if random.random() < pc:
                c1_ms, c2_ms = crossover_MS(p1_ms, p2_ms)
                c1_os, c2_os = crossover_OS_POX(p1_os, p2_os, len(instance.jobs))
            else:
                c1_ms, c1_os = list(p1_ms), list(p1_os)
                c2_ms, c2_os = list(p2_ms), list(p2_os)

            if random.random() < pm: c1_ms = mutate_MS(c1_ms, instance)
            if random.random() < pm: c1_os = mutate_OS(c1_os)
            if random.random() < pm: c2_ms = mutate_MS(c2_ms, instance)
            if random.random() < pm: c2_os = mutate_OS(c2_os)

            new_population.append((c1_ms, c1_os))
            if len(new_population) < pop_size:
                new_population.append((c2_ms, c2_os))

        population = new_population[:pop_size]

    return global_best_schedule, global_best_makespan, best_makespan_history
from model import ScheduledOp

def decode_calc(instance, OS, MS):
    num_jobs = len(instance.jobs)
    num_machines = len(instance.machines)
    op_progress = [0] * num_jobs  # 현재까지 진행된 operation 번호 저장
    job_ready = [0] * num_jobs
    machine_schedules = {m: [] for m in range(num_machines)}
    schedule = []

    # OS
    for job_id in OS:
        op_id = op_progress[job_id]
        op_idx = instance.job_index[job_id] + op_id
        alternative_set_index = MS[op_idx]

        operation = instance.jobs[job_id].operations[op_id]
        machine_id, processing_time = operation.alternatives[alternative_set_index]

        ready_time_of_job = job_ready[job_id]  # Job 내부에서 직전 Operation이 끝난 시간
        machine_ops = machine_schedules[machine_id]  # 할당된 기계의 현재 스케줄 가져오기

        start_time = ready_time_of_job

        inserted = False # 수정
        insert_idx = -1 # 수정

        if machine_ops == []:
            pass
        else:
            if ready_time_of_job + processing_time <= machine_ops[0].start:  # 가장 빠른 machine 스케쥴의 시작 시간과 비교
                insert_idx = 0
                inserted = True
            else:
                # 이미 스케줄된 작업들 사이의 빈 공간들 확인
                for i in range(len(machine_ops) - 1):
                    gap_start = machine_ops[i].end
                    gap_end = machine_ops[i + 1].start

                    # 작업 준비 시간(ready_time_of_job)과 앞 작업 끝난 시간(gap_start) 중 늦은 시간에 시작
                    possible_start = max(ready_time_of_job, gap_start)

                    if possible_start + processing_time <= gap_end:
                        start_time = possible_start
                        inserted = True
                        insert_idx = i + 1
                        break

                if not inserted:
                    start_time = max(ready_time_of_job, machine_ops[-1].end)

        end_time = start_time + processing_time
        new_op = ScheduledOp(job_id, op_id, machine_id, start_time, end_time)

        schedule.append(new_op)

        if inserted: # 수정
            machine_schedules[machine_id].insert(insert_idx, new_op)  # 수정
        else:  # 수정
            machine_schedules[machine_id].append(new_op)  # 수정

        job_ready[job_id] = end_time
        op_progress[job_id] += 1

    return schedule


def makespan(schedule):
    if not schedule:
        return 0
    return max(op.end for op in schedule)
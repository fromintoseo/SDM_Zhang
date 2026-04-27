from model import Instance, Job, Machine, Operation

def load_mk(path: str) -> Instance:
    with open(path, 'r') as f:
        mk = f.read().split() # 모든 숫자를 리스트에 넣어서 인덱싱

    # 여러 줄로 나누지 않고 리스트 인덱싱으로 한 번에 깔끔하게
    num_jobs, num_machines, average_machines_per_op = int(mk[0]), int(mk[1]), float(mk[2])
    idx = 3

    instance = Instance()
    for m in range(num_machines):
        instance.machines.append(Machine(machine_id=m, name=f"M{m+1}"))

    op_count = 0
    for j in range(num_jobs):
        job = Job(job_id=j)
        num_ops = int(mk[idx])
        idx += 1

        for o in range(num_ops):
            op = Operation(job_id=j, op_id=o)
            num_alt_machines = int(mk[idx])
            idx += 1

            for _ in range(num_alt_machines):
                machine_id = int(mk[idx]) - 1
                idx += 1
                processing_time = int(mk[idx])
                idx += 1

                op.add_alternative(machine_id, processing_time)

            job.add_operation(op)
            op_count += 1

        instance.jobs.append(job)

    instance.total_operations = op_count
    instance.compute_job_idx() # MS 배열 탐색용
    return instance
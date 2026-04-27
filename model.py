class Machine:
    def __init__(self, machine_id, name):
        self.machine_id = machine_id
        self.name = name

class Operation:
    def __init__(self, job_id, op_id):
        self.job_id = job_id
        self.op_id = op_id
        self.alternatives = []

    def add_alternative(self, machine_id, processing_time):
        self.alternatives.append((machine_id, processing_time))

class Job:
    def __init__(self, job_id):
        self.job_id = job_id
        self.operations = []

    def add_operation(self, op):
        self.operations.append(op)

class Instance:
    def __init__(self):
        self.jobs = []
        self.machines = []
        self.total_operations = 0
        self.job_index = [] # MS 배열 내에서 각 Job의 시작 인덱스
        self.ops_instances = [] # 모든 operation 객체를 MS배열 순서로 저장한 리스트

    def compute_job_idx(self):  # 각 Job의 Operation이 MS 배열에서 시작하는 위치 계산
        idx = 0
        for job in self.jobs:
            self.job_index.append(idx) # 각 Job의 시작 index 저장
            idx += len(job.operations)
            for op in job.operations:
                self.ops_instances.append(op)

class ScheduledOp:
    def __init__(self, job_id, op_id, machine_id, start, end):
        self.job_id = job_id
        self.op_id = op_id
        self.machine_id = machine_id
        self.start = start
        self.end = end


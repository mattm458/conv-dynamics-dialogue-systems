import time
from functools import partial

from loky import get_reusable_executor


def _done_callback(job_device_id, args, executor, results, fn, future):
    if len(args) > 0:
        job_args = args.pop(0)
        job_args["device"] = job_device_id
        result = executor.submit(fn, **job_args)
        result.add_done_callback(
            partial(_done_callback, job_device_id, args, executor, results, fn)
        )
        results.append(result)


def run_distributed(fn, args, devices, processes_per_device):
    # Copy args to prevent modifying the original
    args = args[:]

    max_workers = len(devices) * processes_per_device

    device_ids = []
    for device_id in devices:
        for _ in range(processes_per_device):
            device_ids.append(device_id)

    _results = []
    executor = get_reusable_executor(max_workers=max_workers)

    for _ in range(max_workers):
        job_device_id = device_ids.pop(0)
        job_args = args.pop(0)
        job_args["device"] = job_device_id

        result = executor.submit(fn, **job_args)
        result.add_done_callback(
            partial(_done_callback, device_id, args, executor, _results, fn)
        )
        _results.append(result)

    while len(args):
        time.sleep(1)

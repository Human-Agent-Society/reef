import base64
import dataclasses
import multiprocessing
import re
import time
import os
import sys
import math
from pathlib import Path
from typing import Any, Optional

import torch.cuda

from utils import set_seed
try:
    from task import TestSpec
except ImportError:
    TestSpec = dict

from reference import check_implementation, generate_input


class PopcornOutput:
    def __init__(self, fd: int):
        self.file = os.fdopen(fd, 'w')
        os.set_inheritable(fd, False)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
    
    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.file, flush=True)
    
    def log(self, key, value):
        self.print(f"{key}: {value}")


@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str


def _combine(a: int, b: int) -> int:
    # combine two integers into one:
    # we need this to generate a secret seed based on the test-level seed and
    # the global secret seed.
    # the test-level seeds are public knowledge, and typically relatively small numbers,
    # so we need to make sure they don't provide any useful info for the full seed.
    # This Cantor construction ensures that if the secret seed is a large number,
    # then so is the overall seed.
    return int(a + (a+b)*(a+b+1)//2)


def get_test_cases(file_name: str, seed: Optional[int]) -> list[TestCase]:
    try:
        content = Path(file_name).read_text()
    except Exception as E:
        print(f"Could not open test file`{file_name}`: {E}", file=sys.stderr)
        exit(113)

    tests = []
    lines = content.splitlines()
    match = r"\s*([a-zA-Z]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*"
    for line in lines:
        parts = line.split(";")
        case = {}
        for part in parts:
            matched = re.match(match, part)
            if not re.fullmatch(match, part):
                print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                exit(113)
            key = matched[1]
            val = matched[2]
            try:
                val = int(val)
            except ValueError:
                pass

            case[key] = val
        tests.append(TestCase(spec=line, args=case))

    if seed is not None:
        for test in tests:
            if "seed" in test.args:
                test.args["seed"] = _combine(test.args["seed"], seed)

    return tests


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def calculate_stats(durations: list[int]):
    """
    Calculate statistical data from a list of durations.

    @param durations: A list of durations in nanoseconds.
    @return: A Stats object containing the number of runs, mean, standard deviation, error, best, and worst durations.
    """
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    variance = sum(map(lambda x: (x - avg)**2, durations))
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)

    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best),
                 worst=float(worst))


def _clone_data(data):
    """
    Recursively goes through data and clones all tensors.
    """
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data


def wrap_check_implementation(data, submission_output):
    # Old version returned just a single string, new version
    # returns (bool, str); this function ensures compatibility with old
    # problem definitions.
    result = check_implementation(data, submission_output)
    if isinstance(result, tuple):
        return result
    else:
        return not bool(result), result


def _run_single_test(test: TestCase):
    """
    Runs a single test case. Do not call directly
    """
    from submission import custom_kernel
    data = generate_input(**test.args)
    torch.cuda.synchronize()
    submission_output = custom_kernel(_clone_data(data))
    torch.cuda.synchronize()
    return wrap_check_implementation(data, submission_output)


def run_single_test(pool: multiprocessing.Pool, test: TestCase):
    """
    Runs a single test in another process.
    """
    return pool.apply(_run_single_test, (test,))


def run_testing(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]):
    """
    Executes the actual test case code and checks for correctness.

    @param logger: A PopcornOutput object used for logging test results.
    @param tests: A list of TestCase objects representing the test cases to be executed.
    @return: An integer representing the exit status: 0 if all tests pass, otherwise 112.
    """
    passed = True
    logger.log("test-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"test.{idx}.spec", test.spec)
        good, message = run_single_test(pool, test)
        if not good:
            logger.log(f"test.{idx}.status", "fail")
            logger.log(f"test.{idx}.error", message)
            passed = False
        else:
            logger.log(f"test.{idx}.status", "pass")
            if message:
                logger.log(f"test.{idx}.message", message)

    if passed:
        logger.log("check", "pass")
        return 0
    else:
        logger.log("check", "fail")
        return 112


def _run_single_benchmark(test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float) -> Stats | Any:
    """
    Runs one benchmark. Do not call directly.
    """
    from submission import custom_kernel

    durations = []
    # generate input data once
    data = generate_input(**test.args)
    check_copy = _clone_data(data)
    #  first, one obligatory correctness check
    output = custom_kernel(data)
    good, message = wrap_check_implementation(check_copy, output)
    if not good:
        return message

    # now, do multiple timing runs without further correctness testing
    # there is an upper bound of 100 runs, and a lower bound of 3 runs;
    # otherwise, we repeat until we either measure at least 10 full seconds,
    # or the relative error of the mean is below 1%.

    bm_start_time = time.perf_counter_ns()
    for i in range(max_repeats):
        if recheck:
            # ensure we use a different seed for every benchmark
            if "seed" in test.args:
                test.args["seed"] += 13

            data = generate_input(**test.args)
            check_copy = _clone_data(data)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = custom_kernel(data)
        end_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event) * 1e6  # Convert ms to ns

        if recheck:
            good, message = check_implementation(check_copy, output)
            if not good:
                return message

        del output
        durations.append(duration)

        if i > 1:
            total_bm_duration = time.perf_counter_ns() - bm_start_time
            stats = calculate_stats(durations)
            # stop if either
            # a) relative error dips below 0.1%
            # b) we exceed the total time limit for benchmarking the kernel
            # c) we exceed 2 minutes of total wallclock time.
            if stats.err / stats.mean < 0.001 or stats.mean * stats.runs > max_time_ns or total_bm_duration > 120e9:
                break

    return calculate_stats(durations)


def run_single_benchmark(pool: multiprocessing.Pool, test: TestCase, recheck: bool, max_repeats: int,
                         max_time_ns: float):
    """
    For a particular test case, check correctness (if applicable) and grab runtime results.

    @param pool: Process on which the benchmark will be launched.
    @param test: TestCase object.
    @param recheck: Flag for whether to explicitly check functional correctness.
    @param max_repeats: Number of trials to repeat.
    @param max_time_ns: Timeout time in nanoseconds.
    @return: A Stats object for this particular benchmark case or an error if the test fails.
    """
    return pool.apply(_run_single_benchmark, (test, recheck, max_repeats, max_time_ns))


def run_benchmarking(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]):
    """
    Executes benchmarking code for a CUDA Kernel and logs runtimes.

    @param logger: A PopcornOutput object used for logging benchmark results.
    @param pool: Process on which the benchmarks will be launched.
    @param tests: A list of TestCase objects representing the test cases to be benchmarked.
    @return: An integer representing the exit status: 0 if all benchmarks pass, otherwise 112.
    """
    # warm up
    run_single_benchmark(pool, tests[0], False, 100, 10e7)

    passed = True
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        result = run_single_benchmark(pool, test, False, 100, 10e9)
        if isinstance(result, Stats):
            for field in dataclasses.fields(Stats):
                logger.log(f"benchmark.{idx}.{field.name}", getattr(result, field.name))
        else:
            passed = False
            logger.log(f"benchmark.{idx}.status", "fail")
            logger.log(f"benchmark.{idx}.error", result)

    if passed:
        logger.log("check", "pass")
        return 0
    else:
        logger.log("check", "fail")
        return 112


def run_single_profile(test: TestCase) -> str:
    """
    Runs a single test case. Do not call directly
    """
    from submission import custom_kernel
    from torch.profiler import profile, record_function, ProfilerActivity
    data = generate_input(**test.args)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        submission_output = custom_kernel(_clone_data(data))
        torch.cuda.synchronize()
    return prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)


def run_profiling(logger: PopcornOutput, tests: list[TestCase]):
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        report = run_single_profile(test)
        logger.log(f"benchmark.{idx}.report", base64.b64encode(report.encode("utf-8"), b"+*").decode("utf-8"))
    logger.log("check", "pass")
    return 0


def main():
    fd = os.getenv("POPCORN_FD")
    if not fd:
        return 111

    if len(sys.argv) < 3:
        return 2

    mode = sys.argv[1]
    seed = os.getenv("POPCORN_SEED")
    os.unsetenv("POPCORN_SEED")
    seed = int(seed) if seed else None
    set_seed(seed or 42)
    tests = get_test_cases(sys.argv[2], seed)

    with PopcornOutput(int(fd)) as logger:
        import multiprocessing
        mp_context = multiprocessing.get_context('spawn')
        with mp_context.Pool(1) as pool:
            if mode == "test":
                return run_testing(logger, pool, tests)
            if mode == "benchmark":
                return run_benchmarking(logger, pool, tests)

            if mode == "leaderboard":
                # warmup
                run_single_benchmark(pool, tests[0], False, 100, 1e7)
                logger.log("benchmark-count", len(tests))
                passed = True
                for i in range(len(tests)):
                    result = run_single_benchmark(pool, tests[i], True, 100, 30e9)
                    logger.log(f"benchmark.{i}.spec", tests[i].spec)
                    if isinstance(result, Stats):
                        for field in dataclasses.fields(Stats):
                            logger.log(f"benchmark.{i}.{field.name}", getattr(result, field.name))
                    else:
                        passed = False
                        logger.log(f"benchmark.{i}.status", "fail")
                        logger.log(f"benchmark.{i}.error", str(result))  # TODO: Make sure result implements __str__?
                        break

                logger.log("check", "pass" if passed else "fail")
            elif mode == "profile":
                run_profiling(logger, tests)
            else:
                # TODO: Implement script mode
                return 2


if __name__ == "__main__":
    sys.exit(main())
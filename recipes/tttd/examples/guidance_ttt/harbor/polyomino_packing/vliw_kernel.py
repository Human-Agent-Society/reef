from __future__ import annotations

from importlib.resources import files

from ..state import LibraryNode, make_root_node

EDGE_BENCH_TASK_ID = "vliw_kernel_optimization"
EDGE_BENCH_WORK_IMAGE = "seededge/edgebench.work.vliw_kernel_optimization:9fa380a0ebef"
EDGE_BENCH_JUDGE_IMAGE = "seededge/edgebench.judge.vliw_kernel_optimization:5cdef0021634"

_ASSET_DIR = files(__package__).joinpath("assets", EDGE_BENCH_TASK_ID)
VLIW_BASELINE_SOLUTION = _ASSET_DIR.joinpath("baseline_solution.py").read_text(encoding="utf-8")


VLIW_KERNEL_PROBLEM_PROMPT = """You are solving the official EdgeBench VLIW Kernel Optimization task.

Task:
- Modify only solution.py, principally KernelBuilder.build_kernel, so it emits a
  correct and faster instruction program for the frozen custom VLIW/SIMD simulator.
- solution.py must define KernelBuilder.build_kernel(self, forest_height,
  n_nodes, batch_size, rounds). The evaluator instantiates KernelBuilder, calls
  build_kernel, and executes kb.instrs.
- The generated program must produce the same final input indices and values as
  reference_kernel2 for every public and hidden case.
- Do not modify problem.py, runner.py, verifier.py, or test cases. Do not read or
  hard-code public or hidden seeds. Network access is unavailable during judging.

Reference computation:
- Memory words are unsigned 32-bit values.
- mem[0:7] stores rounds, n_nodes, batch_size, forest_height,
  forest_values_p, inp_indices_p, and inp_values_p.
- For every round and batch element, load idx and val, load
  forest_values[idx], replace val by the six-stage HASH_STAGES hash of
  val XOR forest_values[idx], then set idx to 2*idx plus 1 for an even val or
  plus 2 for an odd val. Reset idx to zero when idx >= n_nodes. Store the final
  idx and val back to memory.

Machine model:
- One core, SIMD width VLEN=8, and 1536 32-bit scratch words.
- One dynamically executed instruction bundle costs one simulator cycle.
- A bundle may contain at most 12 scalar ALU slots, 6 vector ALU slots,
  2 load slots, 2 store slots, and 1 flow slot.
- All slots in a bundle read the old scratch/memory state. Scratch and memory
  writes become visible only at the end of the cycle, so dependent operations
  must be placed in later bundles.
- Scalar and vector ALU operations support +, -, *, //, cdiv, XOR, AND, OR,
  shifts, modulo, less-than, and equality. Vector operations additionally
  support vbroadcast and multiply_add.
- Loads support load, load_offset, vload, and const. Stores support store and
  vstore. Flow supports scalar/vector select, add_imm, halt, pause, and jumps.
- vload and vstore operate on eight contiguous words. Indirect forest gathers
  are not contiguous and therefore require scalar loads unless reorganized by
  another correct mechanism.

Evaluation:
- Correctness is mandatory. A candidate failing any hidden case is invalid and
  receives zero training reward.
- The raw score is score_cycles, the maximum simulated cycle count over all
  correct performance cases. Lower raw cycles are better.
- The benchmark reports a monotonic normalized score in [0, 100], where higher
  is better. Guidance-TTT records that official score for reporting and uses a
  dense inverse-cycle reward for RL so every correct speedup has learning signal.
- The primary workload has forest_height=10, rounds=16, and batch_size=256, but
  correctness cases can use other valid argument values.
"""


VLIW_BASELINE_SUMMARY = """The official starter is a scalar, fully unrolled implementation. It allocates a
small shared set of scratch temporaries, processes every round and batch item in
sequence, and emits each scalar ALU, load, store, or flow operation in its own
instruction bundle. It is correctness-oriented but leaves almost all VLIW slots
idle and does not use SIMD, dependency-aware bundle packing, software
pipelining, or overlap between independent batch elements."""


def create_root_node(*, seed: int | None = None) -> LibraryNode:
    _ = seed
    root = make_root_node(problem_id=EDGE_BENCH_TASK_ID, raw_score=None, reward=0.0)
    root.metadata.update(
        {
            "task": EDGE_BENCH_TASK_ID,
            "initialization": "official_edgebench_baseline_bootstrap",
            "work_image": EDGE_BENCH_WORK_IMAGE,
            "judge_image": EDGE_BENCH_JUDGE_IMAGE,
        }
    )
    return root

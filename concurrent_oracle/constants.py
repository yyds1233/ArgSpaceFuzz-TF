"""Constants for the TensorFlow concurrent oracle module."""
from __future__ import annotations

ORACLE_IDS = ("O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9")

ORACLE_NAMES = {
    "O1": "invalid_input_and_safety",
    "O2": "serial_parallel_consistency",
    "O3": "call_independence",
    "O4": "input_and_shared_state_preservation",
    "O5": "exception_isolation_and_recovery",
    "O6": "tfdata_integrity_and_progress",
    "O7": "callback_and_resource_lifecycle",
    "O8": "cpu_threadpool_progress",
    "O9": "cpu_kernel_memory_safety",
}

SCENARIO_IDS = (
    "independent_api_calls",
    "shared_readonly_input",
    "shared_resource_access",
    "parallel_kernel_execution",
    "tfdata_parallel_map",
    "tfdata_interleave",
    "tfdata_early_stop",
    "python_callback_lifecycle",
    "exception_recovery",
    "invalid_input_cpu",
)

SCENARIO_NAMES = {
    "independent_api_calls": "独立 API 并发调用",
    "shared_readonly_input": "共享只读输入",
    "shared_resource_access": "共享资源访问",
    "parallel_kernel_execution": "CPU Kernel 内部并行",
    "tfdata_parallel_map": "Dataset 并行 map",
    "tfdata_interleave": "Dataset interleave",
    "tfdata_early_stop": "Pipeline 提前终止",
    "python_callback_lifecycle": "Python Callback 生命周期",
    "exception_recovery": "异常隔离与恢复",
    "invalid_input_cpu": "CPU 非法输入",
}

STATE_MODELS = (
    "INDEPENDENT_INPUTS",
    "SHARED_READONLY",
    "SHARED_MUTABLE",
    "SHARED_ITERATOR",
    "SHARED_GENERATOR",
    "THREAD_LOCAL_CONTEXT",
    "INTERNAL_RUNTIME",
)

DECISIONS = ("READY", "CONDITIONAL", "LOW_PRIORITY", "INSUFFICIENT_INFO")
CONTRACT_STATUSES = ("SUPPORTED", "UNSUPPORTED", "UNKNOWN")
EXECUTION_STATUSES = ("COMPLETED", "PARTIAL", "TIMEOUT", "CRASH", "SETUP_ERROR", "CAPTURE_ERROR")
ORACLE_STATUSES = ("PASS", "VIOLATION", "INCONCLUSIVE", "NOT_APPLICABLE")
GENERATION_STATUSES = ("GENERATED", "FAILED", "REPAIRED")
VALIDATION_STATUSES = ("PASSED", "FAILED", "PENDING")
INPUT_CLASSES = (
    "VALID", "BOUNDARY_VALID", "INVALID_TYPE", "INVALID_RANK",
    "INVALID_SHAPE", "INVALID_VALUE", "EXTREME_RESOURCE",
)

# Default scenario -> oracle mapping.  The scenario generator may narrow this
# based on the contract registry and the harness analysis.
SCENARIO_ORACLES = {
    "independent_api_calls": ["O2", "O3", "O4"],
    "shared_readonly_input": ["O3", "O4"],
    "shared_resource_access": ["O3", "O4", "O5"],
    "parallel_kernel_execution": ["O2", "O8"],
    "tfdata_parallel_map": ["O2", "O3", "O6"],
    "tfdata_interleave": ["O3", "O6"],
    "tfdata_early_stop": ["O6", "O7"],
    "python_callback_lifecycle": ["O7", "O8"],
    "exception_recovery": ["O5", "O7"],
    "invalid_input_cpu": ["O1", "O9"],
}

# C00..C07 from the design document.
RUNTIME_MATRIX = {
    "C00": {"external_threads": 1, "intra_op_threads": 1, "inter_op_threads": 1, "tf_data_parallel_calls": 1},
    "C01": {"external_threads": 1, "intra_op_threads": 4, "inter_op_threads": 1, "tf_data_parallel_calls": 1},
    "C02": {"external_threads": 1, "intra_op_threads": 1, "inter_op_threads": 4, "tf_data_parallel_calls": 1},
    "C03": {"external_threads": 2, "intra_op_threads": 1, "inter_op_threads": 1, "tf_data_parallel_calls": 1},
    "C04": {"external_threads": 4, "intra_op_threads": 1, "inter_op_threads": 1, "tf_data_parallel_calls": 1},
    "C05": {"external_threads": 4, "intra_op_threads": 2, "inter_op_threads": 2, "tf_data_parallel_calls": 1},
    "C06": {"external_threads": 1, "intra_op_threads": 1, "inter_op_threads": 1, "tf_data_parallel_calls": 4},
    "C07": {"external_threads": 4, "intra_op_threads": 1, "inter_op_threads": 1, "tf_data_parallel_calls": 4},
}

# Scenarios that are meaningful with the first-version CPU implementation.
P0_SCENARIOS = ("independent_api_calls", "shared_readonly_input", "parallel_kernel_execution")

# Capability manifest: a scenario is generated only when its execution backend
# and at least one observation path are implemented.  Unimplemented scenarios
# remain in the suitability report but are skipped explicitly.
SCENARIO_CAPABILITIES = {
    "independent_api_calls": {
        "implemented": True,
        "backend": "captured_target_replay_independent",
        "oracles": ["O2", "O3"],
    },
    "shared_readonly_input": {
        "implemented": True,
        "backend": "shared_target_args",
        "oracles": ["O3"],
    },
    "parallel_kernel_execution": {
        "implemented": True,
        "backend": "captured_target_replay_internal_threads",
        "oracles": ["O2", "O3"],
    },
    "shared_resource_access": {
        "implemented": False,
        "reason": "shared mutable TensorFlow resource backend is P1",
    },
    "tfdata_parallel_map": {
        "implemented": False,
        "reason": "tf.data consumption backend is P2",
    },
    "tfdata_interleave": {
        "implemented": False,
        "reason": "tf.data consumption backend is P2",
    },
    "tfdata_early_stop": {
        "implemented": False,
        "reason": "tf.data lifecycle backend is P2",
    },
    "python_callback_lifecycle": {
        "implemented": False,
        "reason": "callback lifecycle backend is P2",
    },
    "exception_recovery": {
        "implemented": False,
        "reason": "exception isolation/recovery backend is P2",
    },
    "invalid_input_cpu": {
        "implemented": False,
        "reason": "invalid-input backend is P2",
    },
}

DEFAULT_INPUT_CASES = (
    b"\x00" * 64,
    b"\x01" * 64,
    b"\x02" * 64,
    b"\x03" * 64,
    b"\x04" * 64,
    b"\x05" * 64,
    b"\x10" * 64,
    b"\x20" * 64,
)

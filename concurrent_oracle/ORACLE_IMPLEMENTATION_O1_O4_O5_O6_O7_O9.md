# concurrent_oracle O1/O4/O5/O6/O7/O9 实现说明

> 模块：`/root/concurrent_oracle`  
> 日期：2026-11-05  
> 范围：在已有 O2（串并行语义一致）、O3（调用独立性）、O8（CPU ThreadPool 进展）的基础上，实现 O1、O4、O5、O6、O7、O9。  
> 验证：`python3 -m unittest discover -s concurrent_oracle/tests -q` → **88 tests, OK (skipped=5)**；另做了 TensorFlow 2.21 + ASan 的端到端集成运行。

---

## 1. 完成状态

| Oracle | 名称 | 状态 | 执行后端 | 主要场景 |
|---|---|---|---|---|
| **O1** | 非法输入与安全拒绝 | ✅ 已实现 | `invalid_input_replay` | `invalid_input_cpu` |
| **O2** | 串并行语义一致 | ✅ 原有 | `captured_target_replay_*` | `independent_api_calls`、`parallel_kernel_execution` |
| **O3** | 调用独立性与任务关联 | ✅ 原有 | `captured_target_replay_*`、`shared_target_args` | `independent_api_calls`、`shared_readonly_input`、`parallel_kernel_execution` |
| **O4** | 输入与共享只读状态保持 | ✅ 已实现 | `captured_target_replay_*` + input digest | `independent_api_calls`、`shared_readonly_input` |
| **O5** | 异常隔离与恢复 | ✅ 已实现 | `exception_recovery_replay` | `exception_recovery` |
| **O6** | tf.data 数据完整性与进度 | ✅ 已实现 | `tfdata_recipe` | `tfdata_parallel_map`、`tfdata_interleave`、`tfdata_early_stop` |
| **O7** | Python callback/generator 生命周期 | ✅ 已实现 | `callback_recipe` | `python_callback_lifecycle` |
| **O8** | CPU ThreadPool 进展 | ✅ 原有（基础版） | `captured_target_replay_internal_threads` | `parallel_kernel_execution` |
| **O9** | CPU Kernel 内存安全 | ✅ 已实现（解析/判定/结果路径） | 任意执行后端 + sanitizer | `independent_api_calls`、`shared_readonly_input`、`parallel_kernel_execution`、`invalid_input_cpu` |

> `shared_resource_access`（共享可变 Variable/资源）仍是 **P1 未实现**；O4 当前只覆盖输入和共享只读状态，不对共享可变资源做状态契约判定。

---

## 2. 新增/修改的代码

### 2.1 新增 Oracle

| 文件 | 类 | Oracle ID |
|---|---|---|
| `oracles/invalid_input.py` | `InvalidInputOracle` | O1 |
| `oracles/input_preservation.py` | `InputPreservationOracle` | O4 |
| `oracles/exception_recovery.py` | `ExceptionRecoveryOracle` | O5 |
| `oracles/tfdata_integrity.py` | `TFDataIntegrityOracle` | O6 |
| `oracles/callback_lifecycle.py` | `CallbackLifecycleOracle` | O7 |
| `oracles/memory_safety.py` | `MemorySafetyOracle` | O9 |

`oracle_engine.py` 默认注册全部 9 个 Oracle；未注册的 plan oracle 仍会显式返回 `INCONCLUSIVE: oracle X is not implemented in P0`。

### 2.2 新增执行模板

- `templates/recipe_harness.py`：tf.data / callback 场景的独立 recipe harness，不依赖原始 API harness，直接构造有限 Dataset 并记录完整性/生命周期观测。

### 2.3 修改的现有文件

| 文件 | 修改 |
|---|---|
| `constants.py` | C08/C09/C10；O4/O1/O5/O6/O7/O9 加入场景能力；`DEFAULT_INVALID_INPUT_EXCEPTIONS` |
| `suitability_rules.py` | O1/O5 通用场景按 `enabled_oracles` 显式开启，避免默认 O2/O3 报告变化；已有 dataset/callback 规则继续生成 O6/O7 场景 |
| `scenario_generator.py` | plan.metadata 增加 `backend`、`recipe`、`invalid_arg`；recipe 场景使用 `case_source="recipe"` |
| `templates/concurrent_harness.py` | O4 输入哈希；`invalid_input` / `exception_recovery` 两种 mode；`auto_tensor` 非法参数变异 |
| `transformer.py` | recipe backend 直接复制 `recipe_harness.py`；普通 backend 支持 `cases=` 参数 |
| `runner.py` | invalid/exception/recipe mode 分派；baseline 3 次资格检查；`process_repetitions`；sanitizer 解析 |
| `observation.py` | 透传 `input_preservation`、`invalid_input`、`phase`、`dataset(s)`、`callbacks`、`recipe` |
| `pipeline.py` | corpus cases 传给 transformer；recipe 场景跳过 capture/静态 target 校验；runner 传入 sanitizer/process_repetitions |
| `config.py` | `corpus_dir`、`cases_per_api`、`process_repetitions`、sanitizer `build_id/instrumented/options/preload_path` |
| `configs/scan_extended.yaml` | 启用 O1–O8（不含 O9）的扫描配置 |
| `configs/scan_sanitizer.yaml` | 启用 O1/O2/O3/O8/O9 + ASan 的扫描配置 |

---

## 3. 数据契约

### 3.1 `TaskObservation.metadata`

O1：

```json
{
  "invalid_input": {
    "mutation": {"kind": "auto_tensor", "path": "args[0]", "value_repr": "None"},
    "accepted": false,
    "observed_exception": "ValueError",
    "observed_message": "Attempt to convert a value (None) ..."
  }
}
```

O4：

```json
{
  "input_preservation": {
    "checked": 1,
    "preserved": true,
    "diffs": [],
    "before": {"args[0]": "sha256:..."},
    "after": {"args[0]": "sha256:..."},
    "records": [{"checked": 1, "preserved": true, "diffs": [], "...": "..."}]
  }
}
```

O5：

```json
{"phase": "A"}
{"phase": "B", "invalid_input": {"observed_exception": "ValueError", "accepted": false}}
{"phase": "C"}
```

O6：

```json
{
  "dataset": {
    "kind": "tfdata_parallel_map",
    "expected_cardinality": 64,
    "observed_cardinality": 64,
    "expected_ids": [0, 1, "..."],
    "observed_ids": [0, 1, "..."],
    "missing_ids": [],
    "extra_ids": [],
    "mismatched_ids": [],
    "deterministic": true,
    "order_preserved": true,
    "early_stop": {"take": 8, "subsequent_pipeline_ok": true}
  }
}
```

O7：

```json
{
  "callbacks": {
    "kind": "python_callback_lifecycle",
    "callback_count": 10,
    "entered_ids": [0, 1, "..."],
    "exited_ids": [0, 1, "..."],
    "expected_exception_at": 8,
    "exception_propagated": true,
    "subsequent_pipeline_ok": true,
    "consumed_count": 8,
    "events": [{"phase": "enter", "element": 0}, {"phase": "exit", "element": 0}]
  }
}
```

### 3.2 `ExecutionObservation.diagnostics["sanitizer"]`

```json
{
  "enabled": true,
  "type": "asan",
  "detected": true,
  "instrumented": true,
  "build_id": "tf_fuzz-selfbuilt-2.21",
  "report_type": "AddressSanitizer",
  "summary": "ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1",
  "stack": ["#0 0x1 in foo", "#1 0x2 in bar"],
  "raw_path": "/.../stderr.log",
  "exit_code": 1,
  "signal": null
}
```

---

## 4. 各 Oracle 的判定逻辑

### O1 非法输入与安全拒绝

- 后端：`invalid_input_replay`，主模板 `mode="invalid_input"`。
- 对每个 captured case，把选中的参数替换为非法值：
  - `auto_tensor`：选择第一个 tensor/ndarray 参数并置 `None`；
  - `none` / `wrong_type` / `negative_int` / `empty_list`：显式指定变异。
- 模板记录 `observed_exception`、`accepted`；harness 捕获的异常不影响记录。
- 期望异常默认集合：`InvalidArgumentError`、`ValueError`、`TypeError`、`IndexError`、`NotFoundError`、`FailedPreconditionError`、`OutOfRangeError`、`UnimplementedError`、`ResourceExhaustedError`、`InternalError`、`AbortedError`；可通过 `plan.metadata["expected_exception_types"]` 覆盖。
- 判定：
  - 所有任务都以期望输入异常拒绝 → **PASS**；
  - 输入被接受、或异常类型不在期望集合 → 合同 SUPPORTED 时 **VIOLATION**，否则 **INCONCLUSIVE**；
  - TIMEOUT/CRASH/SETUP_ERROR → **INCONCLUSIVE**（原生安全交给 O9）。

### O4 输入与共享只读状态保持

- 主模板在每次 target call 前/后对重构参数计算 SHA-256（`_input_hashes` / `_preservation_record`）。
- tensor/ndarray 使用内容 digest；其他参数使用描述 JSON 的 digest。
- shared_readonly 场景对控制线程构造的共享参数对象在每次调用前后计算 digest。
- 判定：
  - 任一任务 digest 变化 → 合同 SUPPORTED 时 **VIOLATION**，否则 **INCONCLUSIVE**；
  - 全部不变 → **PASS**；
  - 没有 `input_preservation` 记录 → **INCONCLUSIVE**。
- 限制：共享可变 Variable/资源的写入契约不在 O4 中判定（P1）。

### O5 异常隔离与恢复

- 后端：`exception_recovery_replay`，主模板 `mode="exception_recovery"`。
- 同一进程内顺序执行三个阶段：
  - A：正常 captured case；
  - B：同一 case 的非法参数变异；
  - C：另一个正常 captured case。
- 判定：
  - B 必须产生期望输入异常；否则 **INCONCLUSIVE**（测试前提不成立）；
  - A/C 失败 → 合同 SUPPORTED 时 **VIOLATION**，否则 **INCONCLUSIVE**；
  - 全部满足 → **PASS**。
- 限制：A/B 当前顺序执行，不是并发交错；后续可扩展为 A/B 并发 + C 恢复。

### O6 tf.data 数据完整性与进度

- 后端：`tfdata_recipe`，`templates/recipe_harness.py`。
- 支持：
  - `tfdata_parallel_map`：`from_tensor_slices` + `map(num_parallel_calls)`；
  - `tfdata_interleave`：多来源 `interleave`，按 round-robin 期望顺序；
  - `tfdata_early_stop`：`take(k)` 后重新构建并完整消费新管线。
- 判定：
  - cardinality、ID 多重集、value/ID 关联、deterministic 时的顺序、early-stop 后续管线；
  - 任一不满足 → 合同 SUPPORTED 时 **VIOLATION**，否则 **INCONCLUSIVE**；
  - 全部满足 → **PASS**。

### O7 Python callback/generator 生命周期

- 后端：`callback_recipe`，`templates/recipe_harness.py`。
- 记录 callback enter/exit、元素 ID、指定位置的预期异常、异常是否传播、异常后新管线是否可消费。
- 判定：
  - enter/exit 多重集（允许指定异常元素缺少一次 exit）；
  - 预期异常必须传播；
  - 异常后新管线必须成功消费；
  - 任一不满足 → 合同 SUPPORTED 时 **VIOLATION**，否则 **INCONCLUSIVE**。
- 不要求 callback 次数等于消费元素数：prefetch 可能执行额外 callback。

### O9 CPU Kernel 内存安全

- Runner 根据 `SanitizerConfig` 设置 preload path、`ASAN_OPTIONS`/`TSAN_OPTIONS`/`MSAN_OPTIONS`；读取 stderr 和可选 `log_path`。
- 解析：
  - ASan：`ERROR: AddressSanitizer:`；
  - TSan：`WARNING: ThreadSanitizer:`；
  - MSan：`WARNING: MemorySanitizer:`；
  - UBSan：`runtime error:`。
- 判定：
  - 检测到报告 → **VIOLATION**（evidence 包含 report_type/summary/stack）；
  - sanitizer 未启用 → **INCONCLUSIVE**；
  - 启用但 `build_id`/`instrumented` 未确认 → **INCONCLUSIVE**；
  - 完成运行且无报告 → **PASS**；
  - 非 COMPLETED 且无报告 → **INCONCLUSIVE**。
- 限制：真正 TSan 需要独立的 TSan 插桩构建；当前 `scan_sanitizer.yaml` 使用自建 TF 2.21 的 ASan 运行时。

---

## 5. 场景与后端映射

| 场景 | backend | 模板/模式 | Oracle | 配置 |
|---|---|---|---|---|
| `independent_api_calls` | `captured_target_replay_independent` | `concurrent_harness.py`（concurrent） | O2/O3/O4/O9 | C03/C04 |
| `shared_readonly_input` | `shared_target_args` | `concurrent_harness.py`（concurrent） | O3/O4/O9 | C03/C04 |
| `parallel_kernel_execution` | `captured_target_replay_internal_threads` | `concurrent_harness.py`（concurrent） | O2/O8/O9 | C01/C02/C09 |
| `invalid_input_cpu` | `invalid_input_replay` | `concurrent_harness.py`（invalid_input） | O1/O9 | C00 |
| `exception_recovery` | `exception_recovery_replay` | `concurrent_harness.py`（exception_recovery） | O5 | C03 |
| `tfdata_parallel_map` | `tfdata_recipe` | `recipe_harness.py` | O6 | C06/C07 |
| `tfdata_interleave` | `tfdata_recipe` | `recipe_harness.py` | O6 | C06 |
| `tfdata_early_stop` | `tfdata_recipe` | `recipe_harness.py` | O6 | C06 |
| `python_callback_lifecycle` | `callback_recipe` | `recipe_harness.py` | O7 | C03/C04 |
| `shared_resource_access` | — | — | — | **P1 未实现** |

> O1/O5 的通用场景只会在 `enabled_oracles` 包含 O1/O5 时由规则生成；默认 `[O2,O3]` 的旧扫描报告不会多出这些计划。

---

## 6. 如何运行

### 6.1 单 harness 端到端

```bash
cd /root
python -m concurrent_oracle.cli pipeline \
  --harness /root/fuzz_output_experiment/llm.tf.math.log1p.py \
  --max-plans 20 \
  --allow-unknown-contract \
  --config concurrent_oracle/configs/scan_extended.yaml \
  --output-dir /root/concurrent_oracle/outputs/extended_one
```

### 6.2 新 Oracle 的覆盖

- **O1/O4/O5**：对普通 CPU 算子 harness（如 `tests/fixtures/llm.tf.math.log1p.py`）即可生成 `invalid_input_cpu`、`independent_api_calls`（含 O4）、`exception_recovery` 计划。
- **O6**：需要 harness 含 `tf.data.Dataset.*` 调用；fixture：`tests/fixtures/llm.tf.data.Dataset.map.py`。
- **O7**：需要 harness 含 `tf.py_function`/`tf.numpy_function`/generator；fixture：`tests/fixtures/llm.tf.py_function.py`。
- **O9**：使用 `configs/scan_sanitizer.yaml`，并确保 `preload_path` 与实际插桩 TF 构建匹配。

### 6.3 测试

```bash
cd /root
python3 -m unittest discover -s concurrent_oracle/tests -q
```

---

## 7. 验证记录

### 7.1 单元测试

- 88 个测试通过，5 个跳过（需要 TensorFlow 的集成测试）。
- 新增 `tests/test_new_oracles.py`、`tests/test_sanitizer.py`、`tests/test_extended_scenarios.py`。

### 7.2 TensorFlow 2.21 集成运行

| 场景 | 结果 |
|---|---|
| O1 `invalid_input_cpu`（log1p fixture） | 8 个 case 全部以 `ValueError` 拒绝 → **PASS** |
| O4 `independent_api_calls`（log1p fixture） | 2 个 task，输入 digest 不变 → **PASS** |
| O5 `exception_recovery`（log1p fixture） | A COMPLETED / B ValueError / C COMPLETED → **PASS** |
| O6 `tfdata_parallel_map` | 16/16 元素、ID/顺序/值一致 → **PASS** |
| O6 `tfdata_interleave` | 3 来源 × 4 元素、round-robin 顺序一致 → **PASS** |
| O6 `tfdata_early_stop` | take(4) 后新管线 16/16 消费 → **PASS** |
| O7 `python_callback_lifecycle` | 预期异常传播、异常后新管线成功 → **PASS** |
| O9 ASan（log1p fixture，`scan_sanitizer.yaml`） | 无 sanitizer 报告 → **PASS** |
| O9 合成 ASan/TSan 报告 | 解析出 report_type/summary/stack → **VIOLATION** |

### 7.3 扩展 pipeline 运行

在 `tests/fixtures/llm.tf.math.log1p.py` 上运行 `scan_extended` 配置，生成并执行：

- `independent_api_calls`：O2/O3/O4 PASS；
- `shared_readonly_input`：O3/O4 PASS；
- `invalid_input_cpu`：O1 PASS；
- `exception_recovery`：O5 PASS。

在 `tests/fixtures/llm.tf.data.Dataset.map.py` / `llm.tf.py_function.py` 上：

- `tfdata_parallel_map` / `tfdata_early_stop`：O6 PASS；
- `python_callback_lifecycle`：O7 PASS。

---

## 8. 限制与后续工作

1. **O4 共享可变资源**：`shared_resource_access` 仍为 P1；需要 Variable/资源配方后端和操作级契约。
2. **O1/O5 变异策略**：`auto_tensor` 只做通用非法值；不同 API 的“合法但边界”输入、期望异常类型需要合同补充。
3. **O5 并发性**：当前 A/B/C 顺序执行；如果要验证“并发异常隔离”，需要 A/B 并发 + C 恢复的模板变体。
4. **O6/O7 的测试对象**：recipe 场景验证的是 tf.data/callback 管线契约，不是原始 harness 的目标 API；需要真实 Dataset/callback harness 才能覆盖具体 API。
5. **O9 构建**：ASan 路径已实测；TSan/UBSan/MSan 解析已实现，但需要对应插桩构建和 `build_id`/`instrumented` 配置。
6. **批量扫描入口**：运行版本独有的 `scan_harnesses.py` 仍只有 pyc，当前 HEAD 没有源码；批量扫描需要恢复该文件或使用 `pipeline`/CLI 逐个运行。
7. **报告**：新 Oracle 的 evidence 已写入 `OracleEntry.evidence`；`reproduction_rate` 仍未实现（P3）。

---

## 9. 文件清单

新增：

```text
concurrent_oracle/oracles/invalid_input.py
concurrent_oracle/oracles/input_preservation.py
concurrent_oracle/oracles/exception_recovery.py
concurrent_oracle/oracles/tfdata_integrity.py
concurrent_oracle/oracles/callback_lifecycle.py
concurrent_oracle/oracles/memory_safety.py
concurrent_oracle/templates/recipe_harness.py
concurrent_oracle/configs/scan_extended.yaml
concurrent_oracle/configs/scan_sanitizer.yaml
concurrent_oracle/tests/test_new_oracles.py
concurrent_oracle/tests/test_sanitizer.py
concurrent_oracle/tests/test_extended_scenarios.py
concurrent_oracle/tests/fixtures/llm.tf.data.Dataset.map.py
concurrent_oracle/tests/fixtures/llm.tf.py_function.py
concurrent_oracle/ORACLE_IMPLEMENTATION_O1_O4_O5_O6_O7_O9.md
```

修改：

```text
concurrent_oracle/constants.py
concurrent_oracle/config.py
concurrent_oracle/suitability_rules.py
concurrent_oracle/scenario_generator.py
concurrent_oracle/templates/concurrent_harness.py
concurrent_oracle/transformer.py
concurrent_oracle/runner.py
concurrent_oracle/observation.py
concurrent_oracle/pipeline.py
concurrent_oracle/oracle_engine.py
concurrent_oracle/case_capture.py
concurrent_oracle/harness_analyzer.py
concurrent_oracle/tests/test_oracle_engine.py
```

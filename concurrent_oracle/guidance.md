# TensorFlow CPU 并发 Oracle 模块：总体设计与工程实施指南

> 版本：工程设计 v1.0  
> 适用项目：ArgSpaceFuzz-TF  
> 模块名称：`concurrent_oracle`  
> 目标运行环境：TensorFlow CPU-only / CPU 插桩构建  
> 接入方式：直接消费 `fuzz_api_one` 生成的 Harness  
> 核心原则：整体流水线、阶段接口和数据模型与 PyTorch `concurrent_oracle` 保持一致；TensorFlow Runtime 适配、并发场景和 Oracle 独立实现。

---

# 1. 设计目标与基本原则

## 1.1 模块定位

在 `ArgSpaceFuzz-TF` 中新增独立的 `concurrent_oracle/` 模块，作为与现有 `screen/` 并列的第二条测试流水线。

现有工程继续负责：

- API 参数空间/PSR 的构建；
- `fuzz_api_one` 的 Harness 生成；
- 原始 Harness 的基础语法和可执行性校验；
- Screen 原有的执行筛选工作。

新增模块负责：

- 加载 `fuzz_api_one` 生成的 Harness；
- 分析 API 是否适合并发测试；
- 生成并发测试场景和执行计划；
- 将原始 Harness 转换为并发 Harness；
- 执行 CPU 多线程、TensorFlow 内部并行和 `tf.data` 并行测试；
- 收集执行观测并运行 Oracle；
- 对候选问题进行复现、缩减和报告。

**本模块不重新生成 PSR，不替代 `fuzz_api_one`，也不接管 Screen 的筛选任务。**

## 1.2 与 PyTorch 版本保持一致的部分

TensorFlow 版本应尽量复用 PyTorch `concurrent_oracle` 已经确定的工程思想和阶段划分：

| 工程能力 | PyTorch 版本 | TensorFlow 版本 |
| --- | --- | --- |
| 输入来源 | `fuzz_api_one` Harness | `fuzz_api_one` Harness |
| 适用性分析 | 规则 + LLM | 规则 + LLM |
| 测试计划 | `ConcurrentTestPlan` | `ConcurrentTestPlan` |
| Harness 转换 | LLM + 固定模板 | LLM + TensorFlow 固定模板 |
| 校验  | 语法、依赖、串行基线 | 语法、依赖、串行基线 |
| 执行  | 独立 Runner | 独立 Runner |
| Oracle 调度 | 统一 Oracle Engine | 统一 Oracle Engine |
| 结果状态 | PASS / VIOLATION / INCONCLUSIVE / NOT_APPLICABLE | 保持一致 |
| 复现与缩减 | 独立模块 | 独立模块 |
| Screen | 完全解耦 | 完全解耦 |

两套模块不要求共用具体实现代码，但应保持相近的模块边界、核心对象名称、状态定义和 CLI 使用方式。这样便于后续统一实验、统计和维护。

## 1.3 TensorFlow 版本需要独立实现的部分

TensorFlow 的 CPU 并发模型与 PyTorch 不同，不能直接复制 PyTorch Oracle 的内部逻辑。

TensorFlow 版本重点适配：

1. **Python 外部线程并发**：多个 Python worker 同时调用 TensorFlow API。
2. **Intra-op 并行**：单个 TensorFlow 算子内部的 CPU 并行。
3. **Inter-op 并行**：多个 TensorFlow 操作之间的 Runtime 调度。
4. **tf.data 并行流水线**：`map`、`interleave`、`prefetch`、iterator 和 generator。
5. **Python Callback 与 GIL**：C++ worker 调用 Python 函数时的锁交互。
6. **Runtime 资源生命周期**：Dataset、Iterator、Generator 以及可选的 TF1 Session。
7. **CPU Kernel 内存安全**：在 CPU 插桩构建下采集 ASan/UBSan 等错误。

第一阶段只做 CPU，不实现 CUDA、cuDNN、GPU Kernel、GPU Atomic 或 GPU 内存竞态相关测试。

## 1.4 基本设计原则

- **独立接入**：只通过 Harness 文件、PSR 和可选元信息接入，不依赖 Screen。
- **原始 Harness 只读**：所有并发转换结果写入新目录。
- **先分析、后转换**：适用性分析通过后才生成并发 Harness。
- **场景驱动**：同一个 API 可以生成多个并发场景，不使用简单的“适合/不适合”二分类。
- **规则与 LLM 协作**：规则负责可确定的结构识别，LLM 负责语义分析和代码转换。
- **合同驱动判定**：并发异常不自动等于 Bug，必须结合 API 合同和可验证性质。
- **执行与判定分离**：Runner 负责运行和采集，Oracle Engine 负责性质判定。
- **进程级隔离**：可能卡死、Abort 或触发 Sanitizer 的用例必须在独立子进程中运行。
- **完整可复现**：保存原始 Harness、转换结果、测试计划、输入、运行配置、日志和 Oracle 结果。

---

# 2. 总体架构

## 2.1 系统整体流程

```text
ArgSpaceFuzz-TF
│
├── build_yaml / PSR
│
├── fuzz_api_one
│   ├── API 参数空间读取
│   ├── Harness 生成
│   └── 原始 Harness 校验
│
├── screen
│   └── 原有执行筛选流程（保持不变）
│
└── concurrent_oracle
    │
    ├── 1. Harness 加载
    │
    ├── 2. Harness 静态分析
    │
    ├── 3. API 并发适用性分析
    │      ├── 规则预筛选
    │      ├── LLM 语义分析
    │      └── Oracle 可判定性检查
    │
    ├── 4. 并发测试计划生成
    │
    ├── 5. 并发 Harness 转换
    │      ├── 固定模板转换
    │      ├── TensorFlow Adapter
    │      └── LLM 辅助转换
    │
    ├── 6. Harness 校验
    │      ├── 语法与依赖检查
    │      ├── 目标 API 调用检查
    │      └── 串行基线检查
    │
    ├── 7. CPU 并发执行
    │      ├── Python 外部线程
    │      ├── TensorFlow intra-op
    │      ├── TensorFlow inter-op
    │      └── tf.data 并行 worker
    │
    ├── 8. 结果观测与 Oracle 判定
    │
    ├── 9. 重复复现与失败缩减
    │
    └── 10. 标准化报告
```

## 2.2 模块边界

```text
fuzz_api_one
    |
    | HarnessArtifact
    | PSR / metadata（可选）
    v
concurrent_oracle.harness_loader
    |
    v
harness_analyzer
    |
    v
api_suitability_analyzer
    |
    v
scenario_generator
    |
    v
llm_transformer / templates / adapters
    |
    v
harness_validator
    |
    v
runner
    |
    v
observation
    |
    v
oracle_engine
    |
    +------> reproducer
    |
    +------> reducer
    |
    v
reporter
```

Screen 不参与以上任何调用链。

## 2.3 模块职责

| 模块  | 职责  | 是否调用 LLM |
| --- | --- | --- |
| `harness_loader` | 加载 Harness、PSR、元信息 | 否   |
| `harness_analyzer` | 提取 API 调用、输入构造、输出处理和资源线索 | 可选  |
| `api_suitability_analyzer` | 协调适用性分析 | 是   |
| `suitability_rules` | 根据 API 名称、AST 和 Harness 结构预筛选 | 否   |
| `llm_suitability` | 识别语义风险、候选场景和信息缺口 | 是   |
| `contract_registry` | 查询和维护 API 合同、支持范围及证据 | 否   |
| `scenario_generator` | 生成测试场景、并发配置和 Oracle 组合 | 可选  |
| `tensorflow_adapter` | 处理 TensorFlow Runtime、设备、输入和输出差异 | 否   |
| `llm_transformer` | 将原始 Harness 改造成指定并发场景 | 是   |
| `harness_validator` | 校验转换产物和串行基线 | 否   |
| `runner` | 子进程隔离、线程调度、超时和日志采集 | 否   |
| `observation` | 将原始执行结果转换为统一观测结构 | 否   |
| `oracle_engine` | 调用 Oracle 并统一结果状态 | 否   |
| `reproducer` | 重复运行和复现率统计 | 否   |
| `reducer` | 缩减输入、线程数和任务组合 | 可选  |
| `reporter` | 生成 JSON、Markdown 和复现目录 | 否   |

## 2.4 数据流转

一次 API 测试从输入到结果，至少产生以下中间文件：

```text
outputs/
├── artifacts/
│   └── <artifact_id>.json
├── analysis/
│   └── <artifact_id>_analysis.json
├── suitability/
│   └── <artifact_id>_suitability.json
├── plans/
│   └── <plan_id>.json
├── generated_harnesses/
│   └── <plan_id>/
│       ├── concurrent_harness.py
│       ├── transform_record.json
│       └── validation.json
├── captured_cases/
│   └── <plan_id>/
├── executions/
│   └── <plan_id>/
│       └── <config_id>/
│           ├── execution.json
│           ├── stdout.log
│           ├── stderr.log
│           ├── task_results.jsonl
│           └── diagnostics/
├── oracle_results/
│   └── <plan_id>.json
├── reproducers/
│   └── <failure_id>/
└── reports/
    └── <artifact_id>_report.md
```

每个阶段都必须将输入和输出持久化。后续阶段应能直接读取上一步产物，不要求从头重新运行 LLM 分析或 Harness 转换。

---

# 3. API 并发测试适用性分析

## 3.1 模块目标

适用性分析不是简单判断 API 是否线程安全，而是判断：

- 当前 Harness 是否有并发测试价值；
- 哪些并发场景适合该 API；
- 需要共享哪些对象、隔离哪些对象；
- 哪些 Oracle 可以给出确定判定；
- 是否需要 TensorFlow 特定 Runtime 配置；
- 当前信息是否足以生成可执行的并发 Harness。

同一个 API 可以同时具有多个候选场景。例如，Dataset API 可能适合测试数据完整性、并行进度和资源生命周期，但不适合普通 Tensor 数值串并行差分。

## 3.2 分析输入

`api_suitability_analyzer` 的输入统一为 `HarnessArtifact` 和静态分析结果。

```text
HarnessArtifact
├── artifact_id
├── framework = tensorflow
├── api_name
├── harness_path
├── psr_path
├── metadata_path
└── source_info
```

静态分析阶段需要提取：

| 信息  | 提取内容 | 用途  |
| --- | --- | --- |
| 目标 API | import 路径、调用表达式、函数名 | 判断 API 类型 |
| 输入构造 | Tensor、shape、dtype、seed、参数值 | 生成并发输入 |
| 输出处理 | `.numpy()`、返回值、断言、异常处理 | 建立结果观测 |
| 资源创建 | Dataset、Iterator、Variable、Generator | 识别共享状态 |
| 并行参数 | `num_parallel_calls`、`AUTOTUNE` | 识别内部并行 |
| 线程配置 | intra-op、inter-op | 识别 Runtime 配置 |
| Python 回调 | `py_function`、`numpy_function`、generator | 识别 GIL 交互 |
| 生命周期操作 | `close`、iterator 终止、Session 清理 | 识别析构场景 |
| 设备信息 | CPU device scope、GPU 显式指定 | 判断是否符合 CPU 范围 |
| 随机状态 | `tf.random`、Generator、stateless RNG | 判断随机 Oracle |

静态分析可以使用 Python AST。对于 AST 无法可靠恢复的动态调用、装饰器或间接调用，保留原始代码片段交给 LLM，不要在静态分析阶段强行推断。

## 3.3 规则预筛选

第一版至少实现以下规则：

| 规则类别 | 识别特征 | 候选场景 |
| --- | --- | --- |
| `PURE_TENSOR_OP` | 普通 Tensor 输入输出，无明显共享状态 | 独立调用一致性 |
| `CPU_PARALLEL_OP` | 可能进入 CPU 并行 Kernel | intra-op / inter-op |
| `DATASET_PIPELINE` | `tf.data.Dataset`、map、interleave、prefetch | Pipeline 语义与进度 |
| `PYTHON_CALLBACK` | `tf.py_function`、`tf.numpy_function` | Callback / GIL |
| `GENERATOR` | `from_generator`、Python generator | 生产者消费者与生命周期 |
| `SHARED_VARIABLE` | `tf.Variable`、ResourceVariable | 共享状态 |
| `RANDOM_STATE` | stateful RNG、Generator | 随机状态隔离 |
| `STATELESS_RANDOM` | stateless RNG | 可复现语义 |
| `THREAD_CONFIG` | intra-op/inter-op 配置 | 线程资源配置 |
| `RESOURCE_LIFECYCLE` | iterator、Session、资源 close/finalize | 生命周期 |
| `CPU_IO` | `tf.io`、文件读取等 CPU 数据路径 | 并行 IO 和 Pipeline 进度 |
| `UNKNOWN` | 无法识别 | LLM 补充分析 |

规则引擎只产生候选，不负责最终排除 API。

## 3.4 LLM 适用性分析

LLM 输入内容：

1. 原始 Harness 源码；
2. Harness AST 摘要；
3. 目标 API 名称；
4. PSR 参数约束；
5. TensorFlow 版本和 CPU-only 环境信息；
6. 规则预筛选结果；
7. API 合同和已有证据；
8. 当前支持的 Oracle 列表。

LLM 输出必须为 JSON，不允许只返回自然语言结论。

建议输出：

```json
{
  "artifact_id": "tf_api_001",
  "decision": "READY",
  "api_categories": [
    "DATASET_PIPELINE",
    "PYTHON_CALLBACK"
  ],
  "candidate_scenarios": [
    {
      "scenario_id": "tfdata_parallel_map",
      "oracles": ["O2", "O3", "O6"],
      "state_model": "INDEPENDENT_INPUTS",
      "priority": "HIGH",
      "reason": "The Harness constructs a Dataset map pipeline and can expose parallel execution behavior."
    },
    {
      "scenario_id": "callback_lifecycle",
      "oracles": ["O5", "O7"],
      "state_model": "SHARED_RESOURCE_LIFECYCLE",
      "priority": "MEDIUM",
      "reason": "The Harness contains a Python callback and resource cleanup path."
    }
  ],
  "required_runtime_config": {
    "device": "CPU",
    "intra_op_threads": [1, 2, 4],
    "inter_op_threads": [1, 2],
    "external_threads": [1, 2, 4]
  },
  "contract_status": "UNKNOWN",
  "required_information": [],
  "transformation_difficulty": "MEDIUM",
  "confidence": 0.86
}
```

字段要求：

- `decision`：`READY`、`CONDITIONAL`、`LOW_PRIORITY`、`INSUFFICIENT_INFO`。
- `candidate_scenarios`：可以为空或包含多个场景。
- `oracles`：必须来自已注册的 Oracle ID。
- `state_model`：必须来自系统支持的状态模型枚举。
- `contract_status`：`SUPPORTED`、`UNSUPPORTED`、`UNKNOWN`。
- `confidence`：仅代表 LLM 自评，不代表漏洞概率。

## 3.5 适用性分析决策

| 决策  | 含义  | 工程动作 |
| --- | --- | --- |
| `READY` | 有明确场景和可用 Oracle | 生成测试计划 |
| `CONDITIONAL` | 需要合同、配置或额外信息 | 保存待补条件 |
| `LOW_PRIORITY` | 成本较高或当前价值较低 | 暂缓，可抽样执行 |
| `INSUFFICIENT_INFO` | Harness 或合同信息不足 | 不生成正式测试，等待补充 |

不要使用永久性的 `UNSUITABLE` 把 API 从测试集合中删除。适用性应针对“API × 场景 × Oracle”判断，而不是只针对 API 本身。

## 3.6 适用性分析器接口

```python
class APISuitabilityAnalyzer:
    def __init__(
        self,
        rules,
        llm_client,
        contract_registry,
        config,
    ):
        ...

    def analyze(
        self,
        artifact: "HarnessArtifact",
        analysis: "HarnessAnalysis",
    ) -> "SuitabilityReport":
        ...

    def merge_candidates(
        self,
        rule_result: dict,
        llm_result: dict,
    ) -> dict:
        ...

    def validate_llm_result(
        self,
        result: dict,
    ) -> None:
        ...
```

执行顺序：

```text
HarnessArtifact
      |
      v
HarnessAnalyzer
      |
      v
SuitabilityRules
      |
      v
Rule Candidates
      |
      v
LLM Suitability
      |
      v
ContractRegistry
      |
      v
Candidate Merge
      |
      v
SuitabilityReport
```

实现要求：

- LLM 调用失败时保留规则结果，不能直接丢弃 Harness；
- LLM 返回非法 JSON 时最多有限重试；
- LLM 提出的未知 Oracle ID 必须校验失败；
- 每次分析保存 prompt 版本、模型配置和原始响应；
- 相同 Harness 内容和相同分析配置可以使用缓存；
- 缓存 Key 至少包含 Harness hash、PSR hash、TensorFlow 版本和 prompt 版本。

---

# 4. TensorFlow CPU 并发测试场景模型

## 4.1 统一 TestCase 模型

与 PyTorch 版本保持相同的测试抽象：

\[
\mathrm{TestCase} = (X, C, S, O)
\]

| 字段  | 含义  | TensorFlow 示例 |
| --- | --- | --- |
| `X` | 输入和参数 | Tensor、shape、dtype、seed、Dataset 输入 |
| `C` | 并发执行配置 | 外部线程数、intra-op、inter-op、parallel calls |
| `S` | 共享状态模型 | 独立输入、共享只读、共享 Variable、共享 Iterator |
| `O` | Oracle 计划 | 串并行一致性、数据完整性、生命周期、进度 |

每个测试计划必须明确四个部分，不能只记录一个线程数。

## 4.2 区分三层并发

TensorFlow CPU 测试至少区分以下三个并发维度。

### A. Python 外部线程并发

多个 Python 线程同时调用 TensorFlow API。

```text
Python Thread 0 ---> TensorFlow API(input_0)
Python Thread 1 ---> TensorFlow API(input_1)
Python Thread 2 ---> TensorFlow API(input_2)
Python Thread 3 ---> TensorFlow API(input_3)
```

配置字段：

```json
{
  "external_threads": 4
}
```

主要检测：

- 独立调用是否互相干扰；
- 输出是否串扰；
- 异常是否污染其他任务；
- 并发执行后是否能够恢复；
- 共享资源是否出现非预期状态变化。

### B. TensorFlow 内部并行

单次 API 调用内部可能使用多个 CPU worker。

配置接口：

```python
tf.config.threading.set_intra_op_parallelism_threads(n)
tf.config.threading.set_inter_op_parallelism_threads(n)
```

配置字段：

```json
{
  "intra_op_threads": 4,
  "inter_op_threads": 2
}
```

两者必须分开记录。它们不是 Python 外部线程数，也不能用 `ThreadPoolExecutor` 替代。

### C. tf.data 并行

Dataset 自己的并行配置：

```python
dataset.map(
    map_fn,
    num_parallel_calls=parallel_calls,
    deterministic=deterministic,
)
```

以及：

- `Dataset.interleave`
- `Dataset.prefetch`
- `Dataset.from_generator`
- Dataset iterator

配置字段：

```json
{
  "tf_data_parallel_calls": 4,
  "tf_data_deterministic": true
}
```

不是所有 Dataset 方法都具有相同参数，Adapter 必须按目标 API 能力生成配置，不能对不支持的接口强行传入参数。

## 4.3 第一版场景集合

第一版统一支持以下场景 ID：

| 场景 ID | 名称  | 说明  | 主要 Oracle |
| --- | --- | --- | --- |
| `independent_api_calls` | 独立 API 并发调用 | 多线程使用独立输入调用同一 API | O2、O3、O4 |
| `shared_readonly_input` | 共享只读输入 | 多线程读取同一个只读 Tensor/资源 | O3、O4 |
| `shared_resource_access` | 共享资源访问 | 多线程访问 Variable、Generator 等资源 | O3、O4、O5 |
| `parallel_kernel_execution` | CPU Kernel 内部并行 | 扫描 intra-op/inter-op 配置 | O2、O8 |
| `tfdata_parallel_map` | Dataset 并行 map | 比较串行与并行 Pipeline | O2、O3、O6 |
| `tfdata_interleave` | Dataset interleave | 测试并行读取和数据完整性 | O3、O6 |
| `tfdata_early_stop` | Pipeline 提前终止 | 检查取消、迭代结束和资源释放 | O6、O7 |
| `python_callback_lifecycle` | Python Callback 生命周期 | callback、GIL、generator 清理 | O7、O8 |
| `exception_recovery` | 异常隔离与恢复 | 混合正常任务和异常任务 | O5、O7 |
| `invalid_input_cpu` | CPU 非法输入 | 非法参数、边界值和 CPU Sanitizer | O1、O9 |

## 4.4 共享状态模型

统一使用以下枚举：

| 状态模型 | 含义  | 使用场景 |
| --- | --- | --- |
| `INDEPENDENT_INPUTS` | 每个 worker 使用独立输入 | 普通 API 并发 |
| `SHARED_READONLY` | 多个 worker 共享只读输入 | 只读 Tensor |
| `SHARED_MUTABLE` | 多个 worker 共享可变资源 | Variable 等 |
| `SHARED_ITERATOR` | 多个 worker 操作同一 Iterator | Dataset |
| `SHARED_GENERATOR` | 多个 worker 依赖同一 generator | `from_generator` |
| `THREAD_LOCAL_CONTEXT` | 观察线程局部状态 | Python callback 等 |
| `INTERNAL_RUNTIME` | 只调整 TensorFlow 内部线程配置 | CPU Kernel |

**注意：** `SHARED_MUTABLE` 不是默认的错误测试模式。只有 API 合同允许该用法，或明确要探索其未定义/未保证行为时，才创建对应测试计划。合同未知时结果必须保留为候选，不能直接判为违规。

## 4.5 并发配置矩阵

建议第一版采用以下配置：

| 配置  | External Threads | Intra-op | Inter-op | Dataset Parallel Calls | 用途  |
| --- | --- | --- | --- | --- | --- |
| C00 | 1   | 1   | 1   | 1   | 串行基线 |
| C01 | 1   | 4   | 1   | 1   | 单次算子内部并行 |
| C02 | 1   | 1   | 4   | 1   | 算子间并行 |
| C03 | 2   | 1   | 1   | 1   | 外部双线程 |
| C04 | 4   | 1   | 1   | 1   | 外部多线程 |
| C05 | 4   | 2   | 2   | 1   | 内外并发组合 |
| C06 | 1   | 1   | 1   | 4   | Dataset 并行 |
| C07 | 4   | 1   | 1   | 4   | 外部与 Dataset 并行组合 |

配置矩阵按场景裁剪，不做无条件笛卡尔积。

## 4.6 全局线程配置的执行约束

`tf.config.threading.set_intra_op_parallelism_threads()` 和 `set_inter_op_parallelism_threads()` 必须在 TensorFlow Runtime 初始化相关操作前设置。

因此：

1. 不允许在一个已初始化的 TensorFlow 进程中反复修改全局线程配置；
2. 每个不同的 intra-op/inter-op 配置组合使用独立子进程；
3. 子进程启动后先设置环境变量；
4. 导入 TensorFlow；
5. 设置线程配置；
6. 检查 CPU-only 设备；
7. 再加载并执行 Harness。

Runner 必须保证这个顺序。

---

# 5. TensorFlow CPU Oracle 体系

## 5.1 Oracle 总览

Oracle Engine 采用注册式结构。每个 Oracle 实现统一接口，Runner 只负责执行，不在 Runner 内部硬编码具体判定逻辑。

| Oracle | 名称  | 主要判定内容 | 主要场景 |
| --- | --- | --- | --- |
| O1  | 非法输入与安全拒绝 | 非法参数处理、崩溃和 Sanitizer 错误 | CPU 非法输入 |
| O2  | 串并行语义一致 | 串行基线与并发执行结果关系 | 普通 Tensor API |
| O3  | 调用独立性 | 独立任务是否发生非预期串扰 | 外部多线程 |
| O4  | 输入与共享状态保持 | 只读输入、共享状态是否异常变化 | 共享输入/资源 |
| O5  | 异常隔离与恢复 | 异常是否污染其他任务和后续调用 | 混合任务 |
| O6  | tf.data 数据完整性与进度 | 元素丢失、重复、顺序合同、Pipeline 卡住 | Dataset |
| O7  | Callback 与资源生命周期 | GIL、Generator、Iterator、析构和清理 | Python Callback |
| O8  | CPU ThreadPool 进度与资源 | 任务进度、线程资源、超时和退出 | CPU 内部并行 |
| O9  | CPU Kernel 内存安全 | ASan/UBSan/TSan 可观察错误 | CPU 插桩构建 |

第一版优先实现 O2、O3、O6、O8；随后补充 O5、O7、O1、O9。O4 可以先作为 O2/O3 的辅助检查器。

## 5.2 O1：非法输入与安全拒绝 Oracle

**目标：** 对 PSR 定义的非法输入和边界输入进行 CPU 执行检查，确认 API 是否按合同拒绝输入，并记录崩溃或 Sanitizer 错误。

工程实现：

1. 从原始 Harness 读取 PSR 参数约束；
2. 生成合法输入、边界输入和非法输入；
3. 每个输入保存独立 Case；
4. 在独立子进程中执行；
5. 收集 Python 异常、进程退出码、signal 和 Sanitizer 日志；
6. 根据 API 参数合同判定。

输入分类：

```text
VALID
BOUNDARY_VALID
INVALID_TYPE
INVALID_RANK
INVALID_SHAPE
INVALID_VALUE
EXTREME_RESOURCE
```

判定规则：

- 明确要求拒绝的非法输入被接受且产生错误结果，才形成合同违规候选；
- Python 层正常抛出预期异常属于 `PASS`；
- `SIGSEGV`、`SIGABRT` 或 Sanitizer 报告需要单独保存；
- 任意异常不能自动判为安全拒绝；
- 超大资源输入必须受限，不能让测试框架本身发生无界内存消耗。

结果字段：

```json
{
  "oracle_id": "O1",
  "input_class": "INVALID_SHAPE",
  "expected": "REJECT",
  "actual": "PROCESS_CRASH",
  "status": "INCONCLUSIVE",
  "process_signal": "SIGSEGV",
  "sanitizer_log": "diagnostics/asan.log"
}
```

## 5.3 O2：串并行语义一致 Oracle

**目标：** 在 API 合同允许比较的前提下，检查并发执行是否改变 API 的语义结果。

测试流程：

```text
固定输入 X
   |
   +----> Serial Execution ----> Expected Results
   |
   +----> Concurrent Execution -> Actual Results
                                      |
                                      v
                               Result Comparator
                                      |
                                      v
                                  O2 Result
```

执行步骤：

1. 固定输入、随机 seed 和运行配置；
2. 在同一 TensorFlow 版本和设备配置下执行串行基线；
3. 保存串行输出；
4. 使用相同输入构造并发任务；
5. 并发执行；
6. 强制物化输出；
7. 按输出结构、dtype、shape 和数值比较；
8. 生成差异报告。

比较规则：

| 输出类型 | 比较方式 |
| --- | --- |
| 整数 Tensor | 默认精确比较 |
| Bool Tensor | 精确比较 |
| 浮点 Tensor | 使用配置化 atol/rtol |
| NaN | 单独比较 NaN mask |
| Inf | 比较正负无穷位置 |
| Tuple/List/Dict | 递归比较结构和成员 |
| SparseTensor | 比较 indices、values、dense_shape |
| RaggedTensor | 比较 flat_values 和 row_splits |

不能把所有浮点结果都要求 bitwise 一致。CPU 后端可能存在合法的浮点计算顺序差异，比较器应根据 API 合同和算子特性设置容差。

对于 Dataset 的非确定性顺序和 stateful RNG，不直接套用普通 Tensor 的串行逐元素比较，应交给 O6 的专门判定逻辑。

## 5.4 O3：调用独立性 Oracle

**目标：** 多个独立任务并发执行时，任务之间不应出现非预期的输出串扰或异常关联。

场景：

```text
Worker 0: input_0 -> API -> output_0
Worker 1: input_1 -> API -> output_1
Worker 2: input_2 -> API -> output_2
Worker 3: input_3 -> API -> output_3
```

实现要求：

- 每个 worker 有独立 `task_id`；
- 每个 worker 有独立输入或明确的共享模式；
- 使用 `threading.Barrier` 同步开始；
- 每个 worker 独立捕获异常；
- 输出必须与 task_id 绑定；
- 不允许多个 worker 写入同一个未同步的结果变量；
- 每个任务完成后记录开始和结束时间。

核心检查：

1. 输出是否对应正确输入；
2. 是否出现跨任务结果覆盖；
3. 某个任务的异常是否错误地关联到其他任务；
4. 是否出现任务缺失、重复完成或结果丢失；
5. 同样的任务在串行执行时是否正常。

## 5.5 O4：输入与共享状态保持 Oracle

**目标：** 检查输入、只读 Tensor 和声明需要保持稳定的资源状态。

第一版检查：

- 输入 Tensor 的 shape、dtype 和数值是否意外改变；
- 共享只读 Tensor 是否出现非预期修改；
- API 声明为只读的状态是否被破坏；
- 并发执行结束后资源是否仍可读取；
- 共享资源状态是否符合 API 的明确合同。

执行流程：

1. 执行前对输入和资源状态快照；
2. 启动并发任务；
3. 等待所有 worker 完成；
4. 重新采集快照；
5. 比较前后状态；
6. 根据资源合同判定。

注意：

- 不要默认 TensorFlow 所有 API 都不会修改输入；
- 对 `tf.Variable` 等可变资源，必须先确认目标操作语义；
- 如果 API 允许并发状态推进，不能把状态变化直接判为错误；
- 合同未知时标记 `INCONCLUSIVE`。

## 5.6 O5：异常隔离与恢复 Oracle

**目标：** 检查一个并发任务发生异常后，其他任务是否能够正确完成，后续 API 调用是否恢复正常。

测试组合：

```text
Worker 0: 正常输入
Worker 1: 非法输入 -> 预期异常
Worker 2: 正常输入
Worker 3: 正常输入
```

执行步骤：

1. 为任务设置正常/异常标签；
2. 使用 Barrier 同步启动；
3. 每个 worker 独立捕获异常；
4. 记录异常类型和任务 ID；
5. 检查其他正常任务是否完成；
6. 并发任务结束后执行恢复调用；
7. 将恢复调用结果与串行基线比较。

判定：

- 预期异常只应归属对应任务；
- 不应把一个任务的异常错误地报告给其他任务；
- 其他任务是否继续执行，应以 API 和调度合同为准；
- 进程崩溃、永久阻塞和普通 Python 异常必须区分；
- 恢复调用异常时保留完整的前序执行证据。

## 5.7 O6：tf.data 数据完整性与进度 Oracle

**目标：** 检查 Dataset 并行 Pipeline 的数据完整性、顺序合同、输入输出关联和执行进度。

重点 API：

- `Dataset.map`
- `Dataset.interleave`
- `Dataset.prefetch`
- `Dataset.from_tensor_slices`
- `Dataset.from_generator`
- Dataset Iterator

### A. 纯函数 map

输入：

\[
X = [x_0, x_1, \ldots, x_{n-1}]
\]

纯函数：

\[
y_i = f(x_i)
\]

串行执行与并行执行分别获得输出，检查：

- 输出数量；
- 元素是否丢失；
- 元素是否重复；
- 每个输出是否对应正确输入；
- dtype/shape/数值是否满足合同；
- `deterministic=True` 时是否满足顺序要求。

### B. 非确定性顺序

当 `deterministic=False` 时：

- 允许输出顺序变化；
- 仍然检查数据集合完整性；
- 检查元素是否丢失或重复；
- 检查输入输出关联；
- 不要求输出顺序等于串行基线。

### C. 随机算子

stateful RNG 和 stateless RNG 分开处理：

- stateful RNG：记录 seed、执行配置和版本，不默认要求并行结果等于固定串行顺序；
- stateless RNG：使用显式 seed 和输入关联建立更强的确定性预期；
- 随机结果差异不自动判为 Bug。

### D. Pipeline 进度

每个输入元素设置可识别 ID，记录：

- submitted；
- started；
- completed；
- failed；
- consumed。

使用 Watchdog 检测长期无进度。超时首先标记 `TIMEOUT`，再结合线程堆栈、任务进度和重复运行判断是否为死锁候选。

## 5.8 O7：Python Callback 与资源生命周期 Oracle

**目标：** 检查 Python callback、GIL、TensorFlow worker 和 Dataset 资源生命周期之间的阻塞、异常清理和退出问题。

重点 API：

- `tf.py_function`
- `tf.numpy_function`
- `Dataset.from_generator`
- Dataset Iterator
- generator close/finalize
- `tf.compat.v1.Session`（仅当 Harness 本身使用）

测试场景：

1. callback 正常执行后释放资源；
2. callback 抛出异常后清理资源；
3. Dataset 提前停止后释放 Iterator；
4. 多个 worker 执行期间主线程发出停止信号；
5. generator 结束或 close 后进程正常退出；
6. 资源释放后再次创建并执行同类资源。

观测字段：

- callback 调用次数；
- callback 成功/失败次数；
- generator close/finalize 次数；
- Iterator 创建/释放记录；
- worker 退出时间；
- Python 线程堆栈；
- native thread stack（环境支持时）；
- 子进程退出状态。

TF1 Session 只作为可选 Adapter，不应强制接入所有 TF2 Harness。

## 5.9 O8：CPU ThreadPool 进度与资源 Oracle

**目标：** 检查 CPU 内部并行任务是否推进，以及线程资源配置是否出现异常。

测试内容：

- intra-op/inter-op 配置；
- Dataset `num_parallel_calls`；
- 多轮执行后的线程资源变化；
- 任务提交、开始、完成数量；
- 任务等待时间；
- 执行超时；
- 进程退出；
- 环境允许时记录 CPU 使用率和上下文切换。

任务进度模型：

```json
{
  "submitted": 100,
  "started": 100,
  "completed": 100,
  "failed": 0,
  "pending": 0
}
```

判定边界：

- 线程数超过配置值不自动等于 Bug；
- futex 或锁等待高属于性能诊断；
- 执行时间增加不自动等于死锁；
- 只有任务进度长期停止、堆栈和重复执行提供足够证据，才升级为死锁候选；
- 全局线程配置必须在 Runtime 初始化前设置。

## 5.10 O9：CPU Kernel 内存安全 Oracle

**目标：** 利用 CPU 插桩构建检测越界、Use-after-free、double-free 和其他可观测的 CPU 内存错误。

工具边界：

| 工具  | 主要用途 |
| --- | --- |
| ASan | 越界、Use-after-free、double-free 等 |
| UBSan | 未定义行为 |
| TSan | 数据竞争检测，需要独立 TSan 构建 |

不能假设 ASan 构建同时具备 TSan 能力。

并发归因流程：

1. 固定同一输入；
2. 先运行单线程基线；
3. 再运行并发配置；
4. 记录 Sanitizer 报告；
5. 对比触发条件；
6. 重复运行；
7. 只有能够证明与并发配置存在稳定关联时，才标记并发相关候选。

如果单线程也能稳定触发，则归类为普通 CPU Kernel 内存安全问题，不计入并发缺陷统计。

---

# 6. 从原始 Harness 到并发 Harness 的转换流程

## 6.1 输入来源

新模块不直接读取 Fuzzer 内部运行状态。它通过 `fuzz_api_one` 的输出目录或清单文件获取 Harness。

建议支持两种输入方式：

### 单个 Harness

```bash
python -m concurrent_oracle.cli analyze \
  --harness path/to/generated_harness.py
```

### Harness 目录/清单

```bash
python -m concurrent_oracle.cli analyze \
  --harness-dir path/to/fuzz_api_one/output \
  --manifest path/to/harness_manifest.json
```

如果现有 `fuzz_api_one` 没有 manifest，第一版可以根据目录结构扫描 Python 文件，并由 `harness_loader` 补充元信息。不要为了新模块强制修改原始生成流程。

## 6.2 HarnessArtifact 加载

加载时完成：

1. 检查文件是否存在；
2. 计算文件 hash；
3. 识别目标 API；
4. 读取对应 PSR（如果存在）；
5. 读取生成元信息（如果存在）；
6. 保存原始 Harness 路径；
7. 记录 TensorFlow 版本和来源信息（如果可获取）。

原始文件不复制修改。后续阶段通过 Artifact ID 追踪它。

## 6.3 Harness 静态分析

`harness_analyzer.py` 使用 AST 提取：

- import；
- 函数定义；
- 目标 API 调用；
- 参数构造；
- TensorFlow Tensor 构造；
- Dataset 构造；
- Python callback；
- 资源创建和清理；
- 返回值和断言；
- `.numpy()` 等物化位置。

建议输出 `HarnessAnalysis`：

```json
{
  "artifact_id": "tf_api_001",
  "target_calls": [
    {
      "api_name": "tf.data.Dataset.map",
      "line": 28,
      "call_expression": "dataset.map(map_fn, num_parallel_calls=4)"
    }
  ],
  "input_variables": ["input_tensor"],
  "resource_variables": ["dataset", "iterator"],
  "callback_functions": [],
  "output_expressions": ["list(dataset.as_numpy_iterator())"],
  "cleanup_expressions": [],
  "ast_parse_status": "SUCCESS"
}
```

AST 无法识别动态调用时，保留代码片段和定位信息，不要伪造调用关系。

## 6.4 固定输入捕获

并发测试不能直接把多个完整 Fuzzer 主循环放进多个 Python 线程。

正确方式：

```text
原始 Harness
    |
    v
输入构造/Case Capture
    |
    v
固定 TestCase
    |
    +----> 串行基线
    |
    +----> 并发 Harness
```

捕获内容：

- 输入 Tensor 的 dtype、shape 和值；
- API 参数；
- 随机 seed；
- Dataset 元素；
- 资源构造参数；
- 目标 API 调用所需的最小上下文。

大 Tensor 可以保存为独立 `.npz` 或其他二进制文件，JSON 只保存文件路径和元信息。

如果原始 Harness 使用 Atheris 等 Fuzzer 入口，应优先复用其输入构造函数或提取单个输入执行入口，而不是启动多个 Fuzzer 实例。

## 6.5 并发 Harness 转换策略

转换优先级：

1. **固定模板转换**：目标调用和输入构造能够被 AST 识别时使用；
2. **TensorFlow Adapter 转换**：Dataset、Variable、Generator 等资源型对象使用专用 Adapter；
3. **LLM 转换**：调用链复杂、AST 信息不足时使用；
4. **人工标记**：无法提取稳定输入或无法建立 Oracle 的 Harness 不强行生成。

转换时必须保留：

- 原始输入构造；
- 目标 API 调用；
- 原始参数值；
- 原始异常处理语义；
- 输出结构。

只增加：

- 并发 worker；
- 输入复制/共享策略；
- 同步控制；
- 串行基线；
- 结果收集；
- Oracle 所需观测。

## 6.6 并发 Harness 统一入口

所有转换后的 Harness 建议实现统一入口：

```python
def run_case(case_config: dict) -> dict:
    """
    Execute one concurrent test case.

    Returns a JSON-serializable observation.
    """
    ...
```

入口不直接负责最终 Oracle 判定，只负责执行并返回原始观测。

建议结构：

```python
def run_case(case_config):
    inputs = load_case_inputs(case_config)
    runtime = initialize_tensorflow(case_config["runtime"])

    baseline = run_serial_baseline(inputs, case_config)

    concurrent_result = run_concurrent_workers(
        inputs=inputs,
        config=case_config,
    )

    return {
        "baseline": baseline,
        "concurrent": concurrent_result,
        "runtime": collect_runtime_info(runtime),
    }
```

实际转换时根据场景选择执行路径。不要把所有 API 都强制改造成同一个 Dataset 或 ThreadPool 模板。

## 6.7 并发 Harness 校验

`harness_validator.py` 至少检查：

| 检查项 | 检查方式 |
| --- | --- |
| Python 语法 | `ast.parse` / `compile` |
| 依赖导入 | 独立子进程 import |
| 目标 API | AST marker 和运行时调用计数 |
| worker 数量 | 运行时任务记录 |
| 同步逻辑 | Barrier/Queue/Event 使用检查 |
| 异常收集 | 每个 worker 独立捕获异常 |
| 线程收尾 | worker join/退出记录 |
| 输出关联 | task_id 与结果对应 |
| 串行基线 | 与原始 Harness 结果比较 |
| Oracle 绑定 | 计划中的 Oracle 均可执行 |

如果转换代码语法错误或目标 API 未被调用，不能进入正式并发执行。

## 6.8 有限修复机制

转换失败时可以有限修复：

```text
LLM Transform
     |
     v
Syntax / Import / Structure Check
     |
     +---- PASS ---> Serial Baseline
     |
     +---- FAIL ---> Error Summary
                        |
                        v
                   LLM Repair
                        |
                        v
                  Re-validation
```

建议默认最多修复 2 次，次数可配置。

修复时必须保留：

- 原始转换代码；
- 每次修复后的代码；
- 校验错误；
- Prompt 和模型响应；
- 最终选用版本。

不能无限循环修复，也不能在修复过程中改变测试计划的 Oracle 性质。

---

# 7. 核心数据接口与 Schema

阶段之间使用 JSON 持久化，Python 内部建议使用 `dataclass` 或 Pydantic。字段可以扩展，但核心对象和状态应与 PyTorch 版本保持一致。

## 7.1 HarnessArtifact

```json
{
  "artifact_id": "tf_api_001",
  "framework": "tensorflow",
  "api_name": "tf.data.Dataset.map",
  "harness_path": "path/to/original_harness.py",
  "harness_hash": "sha256:...",
  "psr_path": "path/to/api_psr.yaml",
  "metadata_path": null,
  "source": "fuzz_api_one",
  "tensorflow_version": null
}
```

字段：

- `artifact_id`：唯一标识；
- `framework`：固定为 `tensorflow`；
- `api_name`：目标 API；
- `harness_path`：原始 Harness 路径；
- `harness_hash`：用于缓存和追踪；
- `psr_path`：可选；
- `metadata_path`：可选；
- `source`：固定记录来源；
- `tensorflow_version`：生成时版本未知可为空，运行时补充。

## 7.2 HarnessAnalysis

```json
{
  "artifact_id": "tf_api_001",
  "target_calls": [],
  "input_variables": [],
  "resource_variables": [],
  "callback_functions": [],
  "output_expressions": [],
  "cleanup_expressions": [],
  "ast_parse_status": "SUCCESS",
  "warnings": []
}
```

## 7.3 SuitabilityReport

```json
{
  "artifact_id": "tf_api_001",
  "decision": "READY",
  "api_categories": [
    "DATASET_PIPELINE"
  ],
  "rule_candidates": ["O2", "O3", "O6", "O8"],
  "llm_candidates": ["O2", "O6"],
  "candidate_scenarios": [
    "tfdata_parallel_map",
    "tfdata_early_stop"
  ],
  "contract_status": "UNKNOWN",
  "required_information": [],
  "transformation_difficulty": "MEDIUM",
  "rationale": "The Harness builds a Dataset pipeline and can be transformed into a parallel map test.",
  "confidence": 0.85
}
```

## 7.4 ConcurrentTestPlan

```json
{
  "plan_id": "tf_plan_001",
  "artifact_id": "tf_api_001",
  "scenario": "tfdata_parallel_map",
  "runtime": {
    "device": "CPU",
    "external_threads": 1,
    "intra_op_threads": 1,
    "inter_op_threads": 1,
    "tf_data_parallel_calls": 4,
    "deterministic": true
  },
  "state_model": "INDEPENDENT_INPUTS",
  "oracles": ["O2", "O3", "O6", "O8"],
  "contract_status": "SUPPORTED",
  "trials": 20,
  "timeout_seconds": 60,
  "case_source": "captured_cases/tf_plan_001",
  "expected_relation": "DATA_INTEGRITY_AND_ORDER"
}
```

## 7.5 ConcurrentHarnessArtifact

```json
{
  "artifact_id": "tf_api_001",
  "source_harness": "path/to/original_harness.py",
  "concurrent_harness": "outputs/generated_harnesses/tf_plan_001/concurrent_harness.py",
  "plan_id": "tf_plan_001",
  "generation_status": "GENERATED",
  "validation_status": "PASSED",
  "generation_attempt": 1,
  "oracle_bindings": ["O2", "O3", "O6", "O8"]
}
```

## 7.6 ExecutionObservation

```json
{
  "plan_id": "tf_plan_001",
  "config_id": "cpu_config_001",
  "execution_status": "COMPLETED",
  "process_exit_code": 0,
  "process_signal": null,
  "observation_complete": true,
  "tasks": [
    {
      "task_id": 0,
      "input_id": "case_000",
      "status": "COMPLETED",
      "output_path": "task_results/task_000.json",
      "exception_type": null,
      "exception_message": null,
      "started_at": 0.01,
      "finished_at": 0.05
    }
  ],
  "task_progress": {
    "submitted": 4,
    "started": 4,
    "completed": 4,
    "failed": 0,
    "pending": 0
  },
  "runtime": {
    "tensorflow_version": "runtime-filled",
    "device": "CPU",
    "gpu_devices": [],
    "intra_op_threads": 1,
    "inter_op_threads": 1,
    "external_threads": 4,
    "tf_data_parallel_calls": 4,
    "one_dnn_enabled": null
  },
  "diagnostics": {
    "stdout": "stdout.log",
    "stderr": "stderr.log",
    "python_stacks": null,
    "native_stacks": null,
    "sanitizer_log": null
  }
}
```

## 7.7 OracleResult

```json
{
  "plan_id": "tf_plan_001",
  "scenario": "tfdata_parallel_map",
  "execution_status": "COMPLETED",
  "oracle_results": [
    {
      "oracle_id": "O6",
      "status": "PASS",
      "reason": "All input elements were observed exactly once and the configured ordering contract was satisfied."
    },
    {
      "oracle_id": "O8",
      "status": "PASS",
      "reason": "All submitted tasks completed within the execution budget."
    }
  ],
  "evidence_level": "E1",
  "reproduction_rate": null,
  "failure_artifacts": []
}
```

## 7.8 状态枚举

执行状态：

- `COMPLETED`
- `PARTIAL`
- `TIMEOUT`
- `CRASH`
- `SETUP_ERROR`
- `CAPTURE_ERROR`

Oracle 状态：

- `PASS`
- `VIOLATION`
- `INCONCLUSIVE`
- `NOT_APPLICABLE`

合同状态：

- `SUPPORTED`
- `UNSUPPORTED`
- `UNKNOWN`

生成状态：

- `GENERATED`
- `FAILED`
- `REPAIRED`

校验状态：

- `PASSED`
- `FAILED`
- `PENDING`

执行状态和 Oracle 状态必须分开保存。进程正常退出不代表 Oracle 一定 PASS；进程崩溃也不代表所有 Oracle 都 VIOLATION。

---

# 8. Python 模块与类接口设计

## 8.1 `harness_loader.py`

```python
class HarnessLoader:
    def load(
        self,
        harness_path: str,
        psr_path: str | None = None,
        metadata_path: str | None = None,
    ) -> HarnessArtifact:
        ...

    def load_directory(
        self,
        harness_dir: str,
        manifest_path: str | None = None,
    ) -> list[HarnessArtifact]:
        ...

    def calculate_hash(self, path: str) -> str:
        ...
```

职责：

- 加载单个 Harness；
- 扫描 Harness 目录；
- 解析 manifest；
- 计算 hash；
- 不执行 Harness；
- 不修改原始文件。

## 8.2 `harness_analyzer.py`

```python
class HarnessAnalyzer:
    def analyze(
        self,
        artifact: HarnessArtifact,
    ) -> HarnessAnalysis:
        ...

    def extract_imports(self, source: str) -> list[dict]:
        ...

    def extract_target_calls(self, source: str, api_name: str) -> list[dict]:
        ...

    def extract_tensor_construction(self, source: str) -> list[dict]:
        ...

    def extract_dataset_operations(self, source: str) -> list[dict]:
        ...

    def extract_resource_lifecycle(self, source: str) -> list[dict]:
        ...
```

职责：

- AST 解析；
- API 调用定位；
- Tensor/Dataset 构造识别；
- callback 和资源生命周期识别；
- 输出静态分析结果。

## 8.3 `suitability_rules.py`

```python
class SuitabilityRules:
    def analyze(
        self,
        artifact: HarnessArtifact,
        analysis: HarnessAnalysis,
    ) -> dict:
        ...

    def classify_api(self, api_name: str) -> list[str]:
        ...

    def detect_dataset_pattern(self, analysis) -> bool:
        ...

    def detect_callback_pattern(self, analysis) -> bool:
        ...

    def detect_shared_resource(self, analysis) -> bool:
        ...
```

规则尽量可配置，避免把所有 API 名称硬编码在一个大函数里。

## 8.4 `llm_suitability.py`

```python
class LLMSuitabilityAnalyzer:
    def analyze(
        self,
        artifact: HarnessArtifact,
        analysis: HarnessAnalysis,
        rule_result: dict,
        contract_info: dict,
    ) -> dict:
        ...

    def build_prompt(self, context: dict) -> str:
        ...

    def parse_response(self, response: str) -> dict:
        ...

    def validate_response(self, result: dict) -> None:
        ...
```

LLM 只返回结构化建议，不直接执行代码，也不判定漏洞。

## 8.5 `contract_registry.py`

```python
class ContractRegistry:
    def get_contract(
        self,
        api_name: str,
        scenario: str | None = None,
    ) -> dict:
        ...

    def register_contract(self, contract: dict) -> None:
        ...

    def validate_oracle_applicability(
        self,
        api_name: str,
        scenario: str,
        oracle_id: str,
    ) -> dict:
        ...
```

合同文件建议使用 JSON/YAML，至少记录：

```json
{
  "api_name": "tf.data.Dataset.map",
  "scenario": "tfdata_parallel_map",
  "status": "SUPPORTED",
  "source": "TensorFlow API documentation",
  "properties": [
    "output_cardinality",
    "deterministic_order_when_enabled"
  ],
  "notes": [],
  "last_verified_version": null
}
```

没有可靠来源时 `status` 保持 `UNKNOWN`。

## 8.6 `scenario_generator.py`

```python
class ScenarioGenerator:
    def generate(
        self,
        suitability: SuitabilityReport,
    ) -> list[ConcurrentTestPlan]:
        ...

    def generate_runtime_configs(
        self,
        scenario: str,
        api_categories: list[str],
    ) -> list[dict]:
        ...

    def select_oracles(
        self,
        scenario: str,
        contract_info: dict,
    ) -> list[str]:
        ...
```

职责：

- 将适用性候选转换为具体测试计划；
- 设置线程配置；
- 设置共享状态；
- 选择 Oracle；
- 控制测试预算；
- 避免配置笛卡尔积爆炸。

## 8.7 `adapters/base.py`

```python
from abc import ABC, abstractmethod


class FrameworkAdapter(ABC):
    @abstractmethod
    def validate_environment(self) -> dict:
        ...

    @abstractmethod
    def configure_runtime(self, config: dict) -> dict:
        ...

    @abstractmethod
    def load_case_inputs(self, case_path: str):
        ...

    @abstractmethod
    def materialize_output(self, output):
        ...

    @abstractmethod
    def compare_outputs(self, expected, actual, config: dict) -> dict:
        ...

    @abstractmethod
    def cleanup(self) -> None:
        ...
```

## 8.8 `adapters/tensorflow.py`

```python
class TensorFlowAdapter(FrameworkAdapter):
    def validate_environment(self) -> dict:
        ...

    def configure_runtime(self, config: dict) -> dict:
        ...

    def load_case_inputs(self, case_path: str):
        ...

    def materialize_output(self, output):
        ...

    def compare_outputs(self, expected, actual, config: dict) -> dict:
        ...

    def configure_cpu_device(self):
        ...

    def get_runtime_info(self) -> dict:
        ...

    def cleanup(self) -> None:
        ...
```

TensorFlow Adapter 负责：

- CPU-only 检查；
- TensorFlow Runtime 配置；
- Tensor 输入/输出处理；
- Dataset 特有结果物化；
- Tensor 结构比较；
- Runtime 版本和设备信息采集。

## 8.9 `llm_transformer.py`

```python
class LLMHarnessTransformer:
    def transform(
        self,
        artifact: HarnessArtifact,
        analysis: HarnessAnalysis,
        plan: ConcurrentTestPlan,
    ) -> ConcurrentHarnessArtifact:
        ...

    def build_prompt(self, context: dict) -> str:
        ...

    def parse_code(self, response: str) -> str:
        ...

    def repair(
        self,
        source_code: str,
        validation_error: dict,
        plan: ConcurrentTestPlan,
    ) -> str:
        ...
```

Prompt 必须明确：

- 原始 Harness 是唯一输入语义来源；
- 不允许改变目标 API；
- 不允许删除原始输入约束；
- 必须实现指定的并发场景；
- 必须输出 task_id、异常和结果；
- 必须实现 worker 收尾；
- 必须符合 `run_case(case_config)` 入口；
- 不允许在生成代码中执行最终 Oracle 判定。

## 8.10 `harness_validator.py`

```python
class HarnessValidator:
    def validate_syntax(self, path: str) -> dict:
        ...

    def validate_imports(self, path: str) -> dict:
        ...

    def validate_target_call(self, path: str, api_name: str) -> dict:
        ...

    def validate_worker_structure(self, path: str) -> dict:
        ...

    def run_serial_baseline(self, path: str, plan: ConcurrentTestPlan) -> dict:
        ...

    def validate(
        self,
        artifact: ConcurrentHarnessArtifact,
        plan: ConcurrentTestPlan,
    ) -> dict:
        ...
```

## 8.11 `runner.py`

```python
class ConcurrentRunner:
    def run(
        self,
        artifact: ConcurrentHarnessArtifact,
        plan: ConcurrentTestPlan,
    ) -> ExecutionObservation:
        ...

    def run_one_config(
        self,
        harness_path: str,
        config: dict,
        timeout_seconds: int,
    ) -> dict:
        ...

    def collect_process_result(self, process) -> dict:
        ...

    def terminate_process(self, process) -> None:
        ...
```

Runner 必须拥有子进程生命周期，不能仅依赖线程级 Future timeout。

## 8.12 `oracle_engine.py`

```python
class OracleEngine:
    def __init__(self, oracles: list):
        ...

    def register(self, oracle) -> None:
        ...

    def evaluate(
        self,
        observation: ExecutionObservation,
        plan: ConcurrentTestPlan,
    ) -> OracleResult:
        ...
```

Oracle 接口：

```python
class BaseOracle:
    oracle_id: str

    def is_applicable(self, plan, observation) -> bool:
        ...

    def evaluate(self, plan, observation) -> dict:
        ...
```

每个 Oracle 只返回自身判定和证据，不修改执行观测。

## 8.13 `reproducer.py`、`reducer.py`、`reporter.py`

```python
class Reproducer:
    def reproduce(self, result_path: str, trials: int = 20) -> dict:
        ...


class FailureReducer:
    def reduce_input(self, failure_case: dict) -> dict:
        ...

    def reduce_threads(self, plan: dict) -> dict:
        ...

    def reduce_task_count(self, plan: dict) -> dict:
        ...


class Reporter:
    def write_json(self, result: OracleResult) -> str:
        ...

    def write_markdown(self, result: OracleResult) -> str:
        ...

    def package_reproducer(self, result: OracleResult) -> str:
        ...
```

---

# 9. TensorFlow 并发执行模板

## 9.1 模板目标

固定模板负责大多数结构明确的 Harness。LLM 不应每次从零发明线程池、Barrier 和结果收集逻辑。

建议模板位置：

```text
concurrent_oracle/templates/concurrent_harness.py
```

模板提供：

- CPU-only 环境初始化；
- TensorFlow 线程配置；
- Case 加载；
- 串行基线；
- Python worker；
- Barrier；
- 异常收集；
- 输出物化；
- task progress；
- JSON 结果输出。

## 9.2 模板结构示例

以下为模板骨架，实际目标 API 调用由 Adapter 或转换器填入：

```python
import os
import json
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor


def configure_tensorflow(runtime_config):
    # Must be called before TensorFlow runtime initialization.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(
        int(runtime_config.get("intra_op_threads", 1))
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(runtime_config.get("inter_op_threads", 1))
    )

    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        raise RuntimeError(
            f"CPU-only test expected, but GPU devices are visible: {gpu_devices}"
        )

    return tf


def materialize(value):
    import tensorflow as tf

    if isinstance(value, tf.Tensor):
        return value.numpy()

    if isinstance(value, (list, tuple)):
        return type(value)(materialize(x) for x in value)

    if isinstance(value, dict):
        return {
            key: materialize(item)
            for key, item in value.items()
        }

    return value


def worker(task_id, case, barrier, result_list, lock):
    record = {
        "task_id": task_id,
        "status": "RUNNING",
        "started_at": time.time(),
        "output": None,
        "exception_type": None,
        "exception_message": None,
        "traceback": None,
    }

    try:
        barrier.wait(timeout=30)

        # TODO: Replace with the original Harness target API call.
        output = invoke_target_api(case)

        record["output"] = materialize(output)
        record["status"] = "COMPLETED"

    except Exception as exc:
        record["status"] = "EXCEPTION"
        record["exception_type"] = type(exc).__name__
        record["exception_message"] = str(exc)
        record["traceback"] = traceback.format_exc()

    finally:
        record["finished_at"] = time.time()
        with lock:
            result_list.append(record)


def run_case(case_config):
    tf = configure_tensorflow(case_config["runtime"])

    cases = load_cases(case_config)
    worker_count = int(case_config["runtime"]["external_threads"])

    barrier = threading.Barrier(worker_count)
    results = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                worker,
                task_id,
                case,
                barrier,
                results,
                lock,
            )
            for task_id, case in enumerate(cases)
        ]

        for future in futures:
            future.result()

    return {
        "runtime": {
            "tensorflow_version": tf.__version__,
            "device": "CPU",
            "gpu_devices": [],
        },
        "task_count": len(cases),
        "results": sorted(results, key=lambda item: item["task_id"]),
    }
```

该模板是结构示例，不是所有场景的最终实现：

- Dataset 场景应由 `tensorflow_dataset` Adapter 构造 Pipeline；
- 生命周期场景必须有独立子进程和停止机制；
- 共享资源场景不能默认复制输入；
- `invoke_target_api()` 和 `load_cases()` 由转换产物提供；
- Barrier 数量必须与实际参与任务数一致；
- 如果 worker 可能在 Barrier 前失败，需要设置超时和失败释放策略；
- TensorFlow 必须在设置 Runtime 配置后再执行目标操作。

## 9.3 结果物化

TensorFlow Eager 模式下，部分操作可能延迟到结果读取时才真正执行。为了避免只记录 Tensor 对象而没有触发计算，Adapter 应在适当位置执行 `.numpy()` 或递归物化。

Dataset 场景应完整消费或按计划消费迭代器，不能只构造 Dataset 就认为 Pipeline 已执行。

## 9.4 线程与进程的职责

- Python worker：用于同一 TensorFlow 进程内的外部并发；
- 子进程：用于隔离不同 Runtime 配置、卡死和崩溃；
- TensorFlow 内部线程池：由 intra-op/inter-op 控制；
- Dataset worker：由 Dataset 并行参数和 Runtime 调度控制。

不要用多个子进程替代所有外部线程测试，否则测试到的是多进程隔离，不是同进程多线程并发。

---

# 10. Runner、Watchdog 与 CPU 执行环境

## 10.1 Runner 进程模型

建议父进程负责配置调度和结果汇总，子进程负责导入 TensorFlow 和执行 Harness。

```text
Parent Process
    |
    +-- read TestPlan
    |
    +-- create config_001
    |       |
    |       +--> Child Process
    |               |
    |               +--> Set environment
    |               +--> Import TensorFlow
    |               +--> Configure intra/inter-op
    |               +--> Check CPU-only
    |               +--> Load Harness
    |               +--> Execute
    |               +--> Write observation
    |
    +-- create config_002
    |       |
    |       +--> Child Process
    |
    +-- collect results
    +-- classify timeout/crash
    +-- call Oracle Engine
```

建议使用 `multiprocessing` 的 `spawn` 启动方式，避免父进程提前初始化 TensorFlow 后把 Runtime 状态带入子进程。

## 10.2 子进程初始化顺序

每个子进程严格按照以下顺序：

1. 设置 `CUDA_VISIBLE_DEVICES=-1`；
2. 设置必要的 CPU 环境变量；
3. 导入 TensorFlow；
4. 设置 intra-op/inter-op；
5. 检查 GPU 设备为空；
6. 记录 TensorFlow 版本和 Build Info；
7. 创建输入；
8. 执行串行基线；
9. 执行并发场景；
10. 物化输出；
11. 写入观测；
12. 正常清理并退出。

不要在第 7 步之后再尝试修改全局线程配置。

## 10.3 CPU-only 环境检查

建议检查：

```python
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

并根据当前构建支持情况记录：

- TensorFlow 版本；
- TensorFlow 安装路径；
- CPU 型号；
- CPU 指令集；
- oneDNN 状态；
- GPU 设备列表；
- 插桩类型；
- 线程配置；
- 环境变量。

目标设备可使用：

```python
with tf.device("/CPU:0"):
    output = target_call(...)
```

如果当前构建是 CPU-only，仍然需要保存构建信息和设备检查结果。

## 10.4 超时处理

父进程必须实现进程级超时：

- 正常完成：`COMPLETED`；
- 超过预算：`TIMEOUT`；
- 收到 SIGSEGV/SIGABRT 等：`CRASH`；
- 环境或配置失败：`SETUP_ERROR`；
- 输入捕获失败：`CAPTURE_ERROR`。

超时后：

1. 记录最后一次任务进度；
2. 保存 stdout/stderr；
3. 尝试采集 Python 线程堆栈；
4. 环境允许时采集 native thread stack；
5. 终止子进程；
6. 保存终止方式和退出状态。

不能只依赖 `Future.result(timeout=...)`，因为底层 C++ worker 永久阻塞时，Python 线程级超时不一定能安全回收整个执行环境。

## 10.5 Watchdog

Watchdog 主要监控：

- 子进程存活状态；
- 任务提交数量；
- 任务开始数量；
- 任务完成数量；
- 最后一次进度时间；
- Dataset 消费数量；
- callback 开始/结束数量；
- 资源关闭状态。

建议阈值配置：

```yaml
watchdog:
  process_timeout_seconds: 60
  no_progress_timeout_seconds: 20
  stack_dump_interval_seconds: 10
  terminate_grace_seconds: 3
```

不同场景可以使用不同预算，不能所有 API 固定使用同一个超时值。

## 10.6 Sanitizer 接入

Runner 支持通过配置传入 Sanitizer 环境：

```yaml
sanitizer:
  enabled: true
  type: asan
  log_path: outputs/diagnostics
```

Sanitizer 类型：

- `none`
- `asan`
- `ubsan`
- `tsan`

TSan 必须使用专门构建。若当前环境没有 TSan 构建，配置应明确为 `none`，不能将 ASan 结果当作数据竞争检测结果。

---

# 11. 工程目录结构

建议在 TensorFlow 仓库中新建以下目录。真实仓库路径以实际检查结果为准，不要为了匹配示意结构而重构现有工程。

```text
ArgSpaceFuzz-TF/
├── build_yaml/                         # 现有：PSR / API 参数空间
├── fuzz_api_one/                       # 现有：Harness 生成与校验
├── screen/                             # 现有：执行筛选，保持不变
└── concurrent_oracle/                  # 新增：独立并发测试模块
    ├── __init__.py
    ├── cli.py                          # 独立命令行入口
    ├── pipeline.py                     # 端到端流程编排
    ├── config.py                       # 默认配置、预算、超时
    ├── schemas.py                      # 核心数据模型和枚举
    ├── constants.py                    # Oracle ID、状态常量
    │
    ├── harness_loader.py               # 加载原始 Harness / PSR / metadata
    ├── harness_analyzer.py             # AST 分析和目标 API 提取
    ├── case_capture.py                 # 固定输入和可重放 Case
    │
    ├── api_suitability_analyzer.py     # 适用性分析协调器
    ├── suitability_rules.py            # 规则预筛选
    ├── llm_suitability.py              # LLM 适用性分析
    ├── contract_registry.py            # API 合同登记和查询
    ├── scenario_generator.py           # 并发场景和测试计划生成
    │
    ├── adapters/
    │   ├── __init__.py
    │   ├── base.py                     # FrameworkAdapter 抽象接口
    │   ├── tensorflow.py               # TensorFlow 通用适配
    │   ├── tensorflow_eager.py         # Eager API 适配
    │   ├── tensorflow_dataset.py       # tf.data Pipeline 适配
    │   ├── tensorflow_resource.py      # Variable / Generator / Iterator
    │   └── tensorflow_runtime.py       # CPU 线程配置和设备检查
    │
    ├── transformer.py                 # 固定模板转换
    ├── llm_transformer.py              # LLM 辅助转换
    ├── prompt_templates/
    │   ├── suitability.md              # 适用性分析 Prompt
    │   └── transform_harness.md        # Harness 转换 Prompt
    ├── templates/
    │   ├── concurrent_harness.py       # 通用并发 Harness 模板
    │   ├── dataset_harness.py          # Dataset 场景模板
    │   └── callback_harness.py         # Callback 场景模板
    ├── harness_validator.py            # 转换产物校验
    │
    ├── runner.py                       # 子进程和并发执行
    ├── process_manager.py              # 子进程生命周期管理
    ├── watchdog.py                     # 超时和无进度监控
    ├── observation.py                  # 执行观测标准化
    ├── result_comparator.py            # TensorFlow 输出比较
    │
    ├── oracle_engine.py                # Oracle 注册、调度与判定
    ├── oracles/
    │   ├── __init__.py
    │   ├── base.py                     # BaseOracle
    │   ├── safety.py                   # O1
    │   ├── consistency.py              # O2
    │   ├── independence.py             # O3
    │   ├── state.py                    # O4
    │   ├── recovery.py                 # O5
    │   ├── tfdata.py                   # O6
    │   ├── callback_lifecycle.py       # O7
    │   ├── threadpool.py               # O8
    │   └── memory_safety.py             # O9
    │
    ├── contracts/
    │   └── tensorflow/
    │       ├── threading.json
    │       ├── tf_data.json
    │       ├── callbacks.json
    │       ├── resources.json
    │       └── cpu_ops.json
    │
    ├── regression_corpus/
    │   ├── historical_issues/
    │   ├── generated_cases/
    │   └── manifests/
    │
    ├── reproducer.py                   # 重复复现
    ├── reducer.py                      # 失败用例缩减
    ├── reporter.py                     # JSON / Markdown 报告
    │
    ├── tests/
    │   ├── test_schemas.py
    │   ├── test_harness_loader.py
    │   ├── test_harness_analyzer.py
    │   ├── test_suitability_rules.py
    │   ├── test_suitability_llm.py
    │   ├── test_contract_registry.py
    │   ├── test_scenario_generator.py
    │   ├── test_transformer.py
    │   ├── test_harness_validator.py
    │   ├── test_tensorflow_adapter.py
    │   ├── test_runner.py
    │   ├── test_result_comparator.py
    │   ├── test_oracle_engine.py
    │   └── test_pipeline_smoke.py
    │
    ├── configs/
    │   ├── default.yaml
    │   ├── cpu_smoke.yaml
    │   └── sanitizer.yaml
    │
    └── outputs/                        # 运行产物，加入 .gitignore
        ├── artifacts/
        ├── analysis/
        ├── suitability/
        ├── plans/
        ├── captured_cases/
        ├── generated_harnesses/
        ├── executions/
        ├── diagnostics/
        ├── oracle_results/
        ├── reproducers/
        └── reports/
```

## 11.1 目录实施要求

- 第一版不要求一次性创建全部目录；
- `adapters/` 负责 TensorFlow Runtime 和数据类型差异；
- `oracles/` 负责可验证性质；
- `templates/` 负责稳定的并发执行代码结构；
- `contracts/` 记录 API 合同和来源；
- `outputs/` 不提交大量运行日志和临时生成文件；
- Screen 不应成为任何模块的 import 依赖；
- 不要把所有功能写进 `pipeline.py` 或 `cli.py`。

---

# 12. 独立 CLI 与完整流水线

## 12.1 CLI 设计

CLI 尽量与 PyTorch 版本保持一致。

### 分析单个 Harness

```bash
python -m concurrent_oracle.cli analyze \
  --harness path/to/original_harness.py
```

### 分析 Harness 目录

```bash
python -m concurrent_oracle.cli analyze \
  --harness-dir path/to/fuzz_api_one/output
```

### 生成测试计划和并发 Harness

```bash
python -m concurrent_oracle.cli generate \
  --artifact outputs/artifacts/tf_api_001.json
```

### 执行指定计划

```bash
python -m concurrent_oracle.cli run \
  --plan outputs/plans/tf_plan_001.json
```

### 重复复现

```bash
python -m concurrent_oracle.cli reproduce \
  --result outputs/oracle_results/tf_plan_001.json \
  --trials 50
```

### 失败缩减

```bash
python -m concurrent_oracle.cli reduce \
  --result outputs/oracle_results/tf_plan_001.json
```

### 生成报告

```bash
python -m concurrent_oracle.cli report \
  --result outputs/oracle_results/tf_plan_001.json
```

### 端到端执行

```bash
python -m concurrent_oracle.cli pipeline \
  --harness path/to/original_harness.py
```

端到端命令只是编排入口，不能让各阶段只能通过 pipeline 执行。每个阶段都必须能够单独调试和恢复。

## 12.2 CLI 参数建议

| 参数  | 命令  | 含义  |
| --- | --- | --- |
| `--harness` | analyze | 单个 Harness 路径 |
| `--harness-dir` | analyze | Harness 目录 |
| `--manifest` | analyze | Harness 清单 |
| `--psr` | analyze | PSR 文件 |
| `--artifact` | generate | Artifact JSON |
| `--plan` | run | 测试计划 JSON |
| `--result` | reproduce/reduce/report | Oracle 结果 JSON |
| `--config` | 所有阶段 | 配置文件 |
| `--max-repair` | generate | 最大修复次数 |
| `--trials` | reproduce | 重复执行次数 |
| `--timeout` | run | 子进程超时 |
| `--output-dir` | 所有阶段 | 输出目录 |

---

# 13. 配置文件设计

## 13.1 默认配置

建议 `configs/default.yaml`：

```yaml
framework: tensorflow

paths:
  output_dir: concurrent_oracle/outputs
  contract_dir: concurrent_oracle/contracts/tensorflow
  template_dir: concurrent_oracle/templates

environment:
  device: CPU
  cuda_visible_devices: "-1"
  require_cpu_only: true
  multiprocessing_start_method: spawn

runtime:
  external_threads: [1, 2, 4]
  intra_op_threads: [1, 2, 4]
  inter_op_threads: [1, 2]
  tf_data_parallel_calls: [1, 2, 4]
  deterministic: [true, false]

generation:
  max_repair_attempts: 2
  use_template_first: true
  use_llm_transform: true

execution:
  trials: 10
  timeout_seconds: 60
  no_progress_timeout_seconds: 20
  max_parallel_processes: 1

oracle:
  enabled:
    - O1
    - O2
    - O3
    - O4
    - O5
    - O6
    - O7
    - O8
    - O9

comparison:
  float_atol: 1.0e-6
  float_rtol: 1.0e-5
  exact_integer: true
  compare_nan_mask: true
  compare_inf_sign: true

sanitizer:
  enabled: false
  type: none
```

## 13.2 CPU 冒烟配置

`configs/cpu_smoke.yaml`：

```yaml
framework: tensorflow

environment:
  device: CPU
  cuda_visible_devices: "-1"
  require_cpu_only: true
  multiprocessing_start_method: spawn

runtime:
  external_threads: [1, 2]
  intra_op_threads: [1]
  inter_op_threads: [1]
  tf_data_parallel_calls: [1, 2]
  deterministic: [true]

generation:
  max_repair_attempts: 1
  use_template_first: true
  use_llm_transform: true

execution:
  trials: 3
  timeout_seconds: 30
  no_progress_timeout_seconds: 10
  max_parallel_processes: 1

oracle:
  enabled:
    - O2
    - O3
    - O6
    - O8

sanitizer:
  enabled: false
  type: none
```

## 13.3 配置约束

- 不同 Runtime 全局线程配置必须分进程；
- 不允许 `intra_op_threads=0` 等未核实配置进入默认矩阵；
- `AUTOTUNE` 单独作为一种配置，不与固定整数混淆；
- Dataset 不支持的参数不应由通用配置强行传入；
- Sanitizer 构建能力必须由环境检测确认；
- 预算和超时应可按 API 类型覆盖。

---

# 14. 第一版实施计划与验收标准

## 14.1 P0：工程接入与最小闭环

目标：先证明 TensorFlow 真实 Harness 能够从 `fuzz_api_one` 接入，并完成一次并发测试。

任务：

1. 创建 `concurrent_oracle/`；
2. 定义核心数据模型和状态枚举；
3. 实现 HarnessLoader；
4. 实现 HarnessAnalyzer；
5. 实现基础规则预筛选；
6. 实现 SuitabilityReport；
7. 实现 ScenarioGenerator；
8. 实现 TensorFlowAdapter；
9. 实现固定模板转换；
10. 实现 HarnessValidator；
11. 实现独立 Runner；
12. 实现 O2、O3；
13. 实现 OracleEngine；
14. 实现 JSON 结果保存；
15. 完成一个真实 Harness 的端到端冒烟测试。

P0 验收：

- 能加载一个真实 `fuzz_api_one` Harness；
- 原始 Harness 不被修改；
- 能生成 SuitabilityReport；
- 能生成 ConcurrentTestPlan；
- 能生成并发 Harness；
- 能执行串行基线；
- 能执行至少两个 Python worker；
- 能保存 task_id、输出和异常；
- 能完成串并行结果比较；
- 能生成 OracleResult；
- 不依赖 Screen。

## 14.2 P1：适用性分析和 LLM 转换完善

任务：

1. 完善 AST 规则；
2. 接入 LLM 适用性分析；
3. 结构化校验 LLM JSON；
4. 加入 ContractRegistry；
5. 完善 LLM Transformer；
6. 加入有限修复；
7. 加入转换缓存；
8. 支持批量 Harness 分析；
9. 记录转换成功率和串行基线通过率。

P1 验收：

- 规则和 LLM 结果分别保存；
- 不同 API 可以产生多个测试场景；
- LLM 输出非法字段时不会进入执行；
- 合同未知时不会自动判定 VIOLATION；
- 生成失败可以定位到阶段和错误原因；
- 支持批量分析和单个 Harness 重跑。

## 14.3 P2：TensorFlow 专属 Oracle

按优先级实现：

1. O6：tf.data 数据完整性与进度；
2. O8：CPU ThreadPool 进度与资源；
3. O5：异常隔离与恢复；
4. O7：Python Callback 与资源生命周期；
5. O1：非法输入与安全拒绝；
6. O9：CPU Kernel 内存安全；
7. O4：共享状态保持的扩展检查。

P2 验收：

- Dataset 并行 map 能完成数据完整性检查；
- `deterministic=True/False` 使用不同判定规则；
- 任务超时能输出任务进度；
- callback/generator 场景能记录生命周期；
- CPU Sanitizer 日志能被归档；
- 不把性能变化直接判成 Bug。

## 14.4 P3：复现、缩减和实验

任务：

1. 实现重复复现；
2. 计算候选复现率；
3. 缩减线程数；
4. 缩减任务数量；
5. 缩减输入规模；
6. 保存最小复现 Harness；
7. 生成 Markdown 报告；
8. 建立历史回归集合；
9. 统计各 Oracle 的执行成本和候选产出。

P3 验收：

- 候选问题可独立重复运行；
- 复现配置完整；
- 缩减前后都有原始产物；
- 报告可以定位到原始 Harness 和测试计划；
- 历史回归用例可以一键执行。

---

# 15. 单元测试与集成测试设计

## 15.1 单元测试

| 测试文件 | 测试内容 |
| --- | --- |
| `test_schemas.py` | JSON 序列化、字段校验和状态枚举 |
| `test_harness_loader.py` | 单文件、目录、manifest 加载 |
| `test_harness_analyzer.py` | API、输入、Dataset、callback 识别 |
| `test_suitability_rules.py` | 规则候选输出 |
| `test_suitability_llm.py` | LLM JSON 解析和非法输出处理 |
| `test_contract_registry.py` | 合同读取和状态判断 |
| `test_scenario_generator.py` | 场景、线程配置和 Oracle 组合 |
| `test_transformer.py` | 模板转换和 LLM 转换结果 |
| `test_harness_validator.py` | 语法、导入、目标 API 和串行基线 |
| `test_tensorflow_adapter.py` | CPU 检查、物化和比较 |
| `test_runner.py` | 子进程、超时和退出状态 |
| `test_oracle_engine.py` | Oracle 注册和状态合并 |

## 15.2 集成测试

第一版准备三个最小测试 Harness：

### Case A：普通 Tensor API

目标：

- 验证 Harness 加载；
- 验证独立输入构造；
- 验证两个/四个 Python worker；
- 验证串并行比较；
- 验证 O2/O3。

### Case B：tf.data map

目标：

- 验证 Dataset Adapter；
- 验证 `num_parallel_calls`；
- 验证数据完整性；
- 验证 deterministic 配置；
- 验证 O6/O8。

### Case C：Python callback/generator

目标：

- 验证 callback 调用；
- 验证异常收集；
- 验证 Iterator/generator 结束；
- 验证子进程超时和回收；
- 验证 O5/O7。

## 15.3 端到端验收流程

```text
真实原始 Harness
      |
      v
analyze
      |
      v
SuitabilityReport
      |
      v
generate
      |
      v
ConcurrentTestPlan
      |
      v
ConcurrentHarnessArtifact
      |
      v
validate
      |
      v
run
      |
      v
ExecutionObservation
      |
      v
OracleResult
      |
      v
report
```

必须保留所有阶段文件，不能只输出最终 PASS/FAIL。

---

# 16. 历史问题回归用例的工程接入

历史 TensorFlow Issue 只作为回归用例来源，不需要把 Issue 调研写进核心代码。

建议建立：

```text
regression_corpus/
└── historical_issues/
    ├── issue_932_threadpool_shutdown/
    │   ├── manifest.json
    │   ├── harness.py
    │   └── README.md
    ├── issue_21277_gil_session_lifecycle/
    ├── issue_13932_tfdata_random/
    ├── issue_29968_thread_resource/
    └── issue_88422_thread_config/
```

每个目录的 `manifest.json` 记录：

```json
{
  "issue_id": 932,
  "title": "thread pool deadlocks on shutdown",
  "category": "THREADPOOL_LIVENESS",
  "device": "CPU",
  "reproduction_status": "NOT_VERIFIED",
  "required_tensorflow_version": null,
  "required_build": null,
  "scenario": "threadpool_shutdown",
  "oracles": ["O7", "O8"],
  "notes": "Historical regression candidate. Do not treat as reproduced until verified in the current build."
}
```

回归用例状态：

- `NOT_VERIFIED`
- `REPRODUCED`
- `NOT_REPRODUCED`
- `ENVIRONMENT_DEPENDENT`
- `NOT_APPLICABLE`

不能因为创建了回归目录，就声称历史 Issue 已经复现。底层 C++ ThreadPool 缺陷如果不能通过当前公开 Harness 到达，应标记为当前 Harness 不适用，而不是伪造测试覆盖。

---

# 17. 实验统计与报告格式

## 17.1 Harness 转换统计

至少记录：

- 输入 Harness 数；
- 加载成功数；
- AST 分析成功数；
- 适用性分析完成数；
- `READY` / `CONDITIONAL` / `LOW_PRIORITY` / `INSUFFICIENT_INFO` 数；
- 并发 Harness 生成成功数；
- 语法校验通过数；
- 串行基线通过数；
- 平均转换耗时；
- 平均修复次数。

## 17.2 并发执行统计

至少记录：

- API 数量；
- 场景数量；
- Runtime 配置数量；
- Trial 数；
- `COMPLETED` 数；
- `TIMEOUT` 数；
- `CRASH` 数；
- `SETUP_ERROR` 数；
- 任务完成率；
- 平均执行耗时；
- CPU 线程配置；
- 资源回收状态。

## 17.3 Oracle 统计

至少记录：

- 每个 Oracle 的执行次数；
- `PASS` 数；
- `VIOLATION` 数；
- `INCONCLUSIVE` 数；
- `NOT_APPLICABLE` 数；
- 候选复现率；
- 最终确认问题数；
- 去重后问题数；
- 误报数量；
- 合同未知导致的不确定结果数。

## 17.4 标准报告内容

每个候选问题至少包含：

```text
1. 基本信息
   - Failure ID
   - API
   - TensorFlow version
   - CPU / Build / Sanitizer

2. 原始 Harness
   - 原始路径
   - Hash
   - PSR

3. 测试计划
   - Scenario
   - State Model
   - External Threads
   - Intra-op
   - Inter-op
   - Dataset Parallel Calls

4. 执行结果
   - Execution Status
   - Exit Code / Signal
   - Task Progress
   - Exceptions
   - Timeout

5. Oracle 结果
   - Oracle ID
   - PASS / VIOLATION / INCONCLUSIVE
   - 判定理由
   - 证据文件

6. 复现信息
   - Trial Count
   - Reproduction Rate
   - 最小输入
   - 最小线程配置
   - 复现命令

7. 根因分析
   - 相关源码位置（如已定位）
   - 是否确认并发相关
   - 是否需要人工确认
```

---

# 18. 给 Agent 的实施要求

Agent 开发时必须遵循以下顺序。

## 18.1 开发前检查

1. 检查真实 `ArgSpaceFuzz-TF` 仓库结构；
2. 找到 `fuzz_api_one` 的入口和 Harness 输出位置；
3. 找到 PSR 和元信息格式；
4. 找到当前 TensorFlow CPU-only 构建和运行命令；
5. 检查现有 Python 依赖；
6. 确认 Screen 边界；
7. 对照 PyTorch `concurrent_oracle` 的数据模型和 CLI；
8. 输出仓库接入映射文档。

如果实际仓库与本文示例目录不同，以真实仓库为准。不要为了符合示例目录而重构原有工程。

## 18.2 编码顺序

**第一批：工程骨架**

- `schemas.py`
- `constants.py`
- `config.py`
- `cli.py`
- `harness_loader.py`
- `harness_analyzer.py`

**第二批：分析和计划**

- `suitability_rules.py`
- `llm_suitability.py`
- `contract_registry.py`
- `api_suitability_analyzer.py`
- `scenario_generator.py`

**第三批：转换和校验**

- `adapters/base.py`
- `adapters/tensorflow.py`
- `templates/concurrent_harness.py`
- `transformer.py`
- `llm_transformer.py`
- `harness_validator.py`

**第四批：执行和 Oracle**

- `process_manager.py`
- `runner.py`
- `watchdog.py`
- `observation.py`
- `result_comparator.py`
- `oracle_engine.py`
- `oracles/consistency.py`
- `oracles/independence.py`
- `oracles/tfdata.py`
- `oracles/threadpool.py`

**第五批：扩展能力**

- `oracles/recovery.py`
- `oracles/callback_lifecycle.py`
- `oracles/safety.py`
- `oracles/memory_safety.py`
- `reproducer.py`
- `reducer.py`
- `reporter.py`

## 18.3 开发约束

- 不修改 `fuzz_api_one` 的核心生成逻辑；
- 不修改 Screen；
- 不让 `concurrent_oracle` import Screen 内部模块；
- 原始 Harness 只读；
- 所有转换代码写入新目录；
- 所有 Runtime 配置独立子进程；
- TensorFlow 导入和线程配置顺序必须正确；
- 所有 worker 结果必须带 `task_id`；
- 所有异常必须带类型、消息和 traceback；
- 所有超时必须有子进程级回收；
- 所有 Oracle 必须实现统一接口；
- 所有 Oracle 判定必须保留理由和证据；
- 合同未知时不得自动判 `VIOLATION`；
- LLM 不能负责最终 Oracle 判定；
- 不能把 GPU 相关逻辑混入 CPU 版本；
- 不允许仅创建空文件就认为模块完成。

## 18.4 阶段交付要求

每完成一个阶段，Agent 必须输出：

1. 新增/修改文件清单；
2. 每个文件的职责；
3. 已实现的接口；
4. 执行命令；
5. 单元测试结果；
6. 端到端测试结果；
7. 尚未实现的功能；
8. 与设计文档不一致的地方及原因。

不得一次性生成大量未验证代码。优先保证每个阶段可运行、可测试、可回滚。

---

# 19. 最终设计决策

1. TensorFlow CPU 并发 Oracle 作为独立的 `concurrent_oracle/` 模块接入。
2. 输入直接来自 `fuzz_api_one` 生成的原始 Harness 和可选 PSR/元信息。
3. Screen 保持完全独立，不修改、不调用、不依赖。
4. 工程流程、核心数据模型、状态定义和 CLI 思路与 PyTorch 版本一致。
5. TensorFlow 通过独立 Adapter 处理 Runtime、Tensor、Dataset 和资源差异。
6. 并发测试采用规则 + LLM 适用性分析，再生成场景和测试计划。
7. Harness 转换采用固定模板优先、LLM 辅助、校验和有限修复。
8. 执行层采用父进程调度、子进程隔离、子进程内 Python 多线程的结构。
9. 不同 intra-op/inter-op 配置放在独立进程中。
10. Oracle Engine 使用统一接口，第一版优先实现串并行一致性、调用独立性、tf.data 数据完整性和 CPU ThreadPool 进度。
11. 所有执行结果、输入、配置、日志和复现脚本必须持久化。
12. 先完成一个真实 Harness 的端到端闭环，再扩展更多 Oracle 和历史回归用例。

**最终交付标准：** 从 `fuzz_api_one` 的一个真实 TensorFlow Harness 出发，不经过 Screen，能够独立完成 Harness 加载、适用性分析、测试计划生成、并发 Harness 转换、校验、CPU-only 并发执行、Oracle 判定和报告生成；整个过程可重复执行、可追踪、可复现。
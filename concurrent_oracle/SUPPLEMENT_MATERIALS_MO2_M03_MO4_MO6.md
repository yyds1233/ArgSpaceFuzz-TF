# concurrent_oracle 待补充材料汇总（MO2 / M03 / MO4 / MO6）

> 模块：`/root/concurrent_oracle`  
> 依据：`guidance.md`、`fix_guidance.md`、git HEAD `475ef33`（update oracle v2）源码、`__pycache__` 中运行版本字节码、`outputs/` 下 2026-10-30 的扫描产物、`contracts/` 与 `configs/`。  
> 生成日期：2026-11-03。  
> **2026-11-05 更新**：O1、O4、O5、O6、O7、O9 已在当前工作区实现并通过单元/集成验证，详见 [ORACLE_IMPLEMENTATION_O1_O4_O5_O6_O7_O9.md](ORACLE_IMPLEMENTATION_O1_O4_O5_O6_O7_O9.md)。本文以下 2026-10-30 的扫描统计仍是**旧版本产物**，不包含这六个 Oracle；关于“当前完成度”的结论请以新实现文档为准。  
> 重要原则：本文只把**已实现并实际运行**的内容写成完成项；设计文档中的建议字段、建议参数、P1/P2 规划不当作实测结果。凡是当前没有运行记录的，明确标为 **未实现 / 占位 / 待实测**。

---

## 0. 一页结论

| 补充项 | 当前可以交付的内容 | 当前不能声称的内容 | 状态 |
|---|---|---|---|
| **MO2** LLM 适用性分析与转换提示 | 规则适用性分析、固定模板转换、LLM 接口占位与回退骨架 | 实际启用的 LLM 提示、模型版本、采样参数、输出提取、回退记录 | **未启用 LLM / P1 占位** |
| **M03** PSR 映射、fuzz 追溯、耗时、命名 | `psr_path` 引用、ArgFuzz corpus 部分接入、每 API 耗时、稳定命名规则、TensorFlow import 别名解析 | PSR 状态到调用的逐项映射、主扫描的 corpus 追溯、逐配置耗时、PSR 别名传递 | **部分实现** |
| **MO4** 数值复核、特殊值、完整性、稳定性 | 完整值逐元素比较、NaN/Inf 位置检查、逐 trial/逐任务完整性、串行 baseline 资格检查 | 大张量分块数值复核、重复串行稳定性统计、跨 PyTorch/TF 的实测兼容结论 | **部分实现** |
| **MO6** 候选复现、分布、缩减入口 | 完整的状态分布、34 个执行失败 API 清单、固定预算边界 | 候选复现清单（本轮为 0）、reproducer/reducer 入口、复现失败率和触发率 | **候选为 0 / 复现与缩减未实现** |

### 0.1 本轮扫描的关键数字

- 扫描入口：`/root/fuzz_output_experiment` 下的 **362 个 TensorFlow harness**。
- 最终合并状态（`outputs/vuln_scan_combined/FINAL_VULN_REPORT.md`）：
  - **CLEAN 131**、**WARNING 178**、**EXECUTION_ISSUE 34**、**NOT_APPLICABLE 19**。
  - **confirmed contract-gated violation = 0**。
  - **candidate findings（UNKNOWN contract 下的具体差异）= 0**。
  - **incomplete/uncertain observations = 0**（首轮 31 个 INCONCLUSIVE 经 retry 后转为 20 个 WARNING + 11 个 CLEAN）。
  - execution issue entries = **36**（分布在 **34 个 API**，其中 `tf.nn.depthwise_conv2d` 占 3 条）；warning entries = **196**（分布在 **178 个 API**）。
- 初次扫描：362 个 API / 1372 个生成计划 / **309 个实际完成计划**；总 `elapsed_seconds` = **2533.4 s**，均值 **7.0 s/API**，p50 7.7 s，max 23.1 s。
- 压力扫描（`vuln_scan_stress_full`）：48 个 API / 101 个计划 / 83 个 COMPLETED + 2 个 SETUP_ERROR；总耗时 **999.3 s**，均值 **20.8 s/API**；O8（ThreadPool 进展）PASS 9 次，NOT_APPLICABLE 76 次。
- 执行完整性：742 份 `combined_observation.json` 中 **704 份 observation_complete=True、38 份 False**；任务进度合计 submitted 2754 / started 2754 / completed 2685 / failed 67 / pending 0。
- 串行 baseline 资格检查：183 份一次通过、24 份两次通过、0 份三次通过；累计排除 **56 个** `baseline_not_complete` case。
- 输入：主扫描 343 个 capture 文件、2744 个 case、1759 个 eligible；**全部为 `deterministic_defaults`**。压力扫描 101 个 capture 文件、1212 个 case、652 个 eligible；**全部为 `argfuzz_corpus`**。

> 结论性提醒：本轮 **0 candidate** 只代表“当前计划、当前 O2/O3 判定、当前 case 范围下没有产生具体差异候选”，不能外推为 TensorFlow 并发安全结论。该模块的 `FINAL_VULN_REPORT.md` 自己也明确写了这一限制。

---

## 1. 版本与证据说明（先读这一节）

### 1.1 工作区与 git HEAD 不一致

当前工作区 `concurrent_oracle/` 下的多数 `.py` 源文件处于 **git 删除状态**（`git status` 显示 ` D`），但 git HEAD `475ef33` 仍保存完整源码。与此同时，`__pycache__/*.pyc` 和 `outputs/` 反映的是一个**比 HEAD 更晚的运行版本**，它包含 HEAD 没有的能力：

| 能力 | git HEAD `475ef33` | 运行版本（`__pycache__` + outputs） |
|---|---|---|
| `case_capture.CaseCapture.corpus_cases` | 无 | 有 |
| `transformer.transform(..., cases=...)` / `argfuzz_corpus` | 无 | 有 |
| `runner.process_repetitions` / `baseline_qualification` | 部分（HEAD 只有单次 baseline） | 有（baseline 最多 3 次尝试，test 进程可重复） |
| `scan_harnesses.py` | 无（HEAD 无此文件） | 有（pyc） |
| `ThreadpoolOracle` / O8 | oracle_engine 只注册 O2/O3 | 注册 O2/O3/O8 |
| `RUNTIME_MATRIX` C08/C09/C10 | 只有 C00–C07 | 有 C08/C09/C10 |

因此本文的引用分三类：

- **[HEAD]**：`git show HEAD:concurrent_oracle/<file>`，用于稳定的接口、数据模型、默认值。
- **[RUN-PYC]**：`__pycache__/<module>.cpython-310.pyc`，用于实际运行版本的行为（corpus、process_repetitions、O8、C08/C09 等）。
- **[OUTPUT]**：`outputs/` 下的扫描结果、报告、case、observation。

### 1.2 复现性限制

当前工作区**不能直接重新运行** `concurrent_oracle`：

- 源码 `.py` 被删除，`__pycache__/*.pyc` 不是可直接导入的包布局；
- 运行版本独有的 `scan_harnesses.py` 只存在于 pyc，HEAD 没有对应源码；
- 运行版本对 `case_capture.py`、`transformer.py`、`runner.py`、`constants.py`、`oracle_engine.py` 的修改没有进入 HEAD。

如果需要严格复现本轮产物，必须先恢复**运行版本源码**（当前仓库内没有完整副本）；仅执行 `git checkout -- concurrent_oracle` 只能恢复到 HEAD 版本，会丢失 corpus、O8、scan_harnesses、C08/C09 等运行版本特性。本文的数据因此主要来自已有产物和 pyc，而不是重新执行。

> 这一点会直接影响 MO6“独立运行候选复现清单”的可复现性，建议在补充材料中单独说明。

---

## 2. MO2：实际启用的 LLM 适用性分析与驱动转换提示

### 2.1 要求拆解

MO2 要求补充：

1. **实际启用的 LLM 适用性分析提示**（suitability prompt）；
2. **驱动/转换提示**（harness transformation prompt）；
3. **模型版本**；
4. **采样参数**（temperature、top_p、max_tokens 等）；
5. **输出提取方式**；
6. **回退记录**；
7. 第三章的生成模型设置，能够替换文章中的“可选模型调用配置”。

### 2.2 实际启用情况：本模块没有启用 LLM

这是本节最重要的结论：**本轮 `concurrent_oracle` 扫描没有实际启用 LLM 适用性分析，也没有启用 LLM 驱动的 harness 转换。** 证据如下。

#### （1）Pipeline 明确传入 `llm_client=None`

[HEAD] `pipeline.py:61-66`：

```python
self.suitability = APISuitabilityAnalyzer(
    rules=self.rules,
    llm_client=None,
    contract_registry=self.contract_registry,
    config={"model_config": {}},
)
```

运行版本 pyc 的 `ConcurrentOraclePipeline.__init__` 同样是 `llm_client=None`、`model_config={}`。

#### （2）LLM 适用性分析是 P1 占位

[HEAD] `llm_suitability.py:1-6` 明确写着：

> “LLM-assisted suitability analysis (P1 placeholder). P0 deliberately uses rules only.”

[HEAD] `llm_suitability.py:23-41`：

```python
def analyze(...):
    if self.client is None:
        return None
    raise NotImplementedError("LLM suitability analysis is scheduled for P1")

def build_prompt(...):
    raise NotImplementedError("LLM suitability analysis is scheduled for P1")

def parse_response(...):
    raise NotImplementedError("LLM suitability analysis is scheduled for P1")

def validate_response(...):
    raise NotImplementedError("LLM suitability analysis is scheduled for P1")
```

也就是说：

- 没有 client 时，`analyze()` 直接返回 `None`；
- 即使注入了 client，`analyze()` 也会抛 `NotImplementedError`，`APISuitabilityAnalyzer` 捕获后记录 `llm_error`，继续使用规则结果；
- `build_prompt` / `parse_response` / `validate_response` 全部未实现，**不存在实际启用的 suitability prompt**。

#### （3）696 份 suitability 报告全部没有 LLM 记录

`outputs/` 下全部 `*_suitability.json` 共 **696 份**，逐项统计：

| 字段 | 结果 |
|---|---|
| `llm_result` | **696/696 为 null** |
| `model_config` | **696/696 为 `{}`** |
| `prompt_version` | **696/696 为 null** |
| `raw_llm_response` | **696/696 为 null** |
| `llm_candidates` | 全部为空 |

示例：`outputs/suitability/tf.math.log1p_a0fd52a7_suitability.json` 的 `decision=READY`、`contract_status=SUPPORTED`、`rule_candidates=[O2,O3,O4]`，但 `llm_result=null`、`model_config={}`、`prompt_version=null`。这说明 **decision 来自规则，不是 LLM**。

#### （4）转换器是固定模板，不是 LLM

[HEAD] `transformer.py:1`：

> “Fixed-template transformation from a fuzz_api_one harness to a concurrent harness.”

[HEAD] `transformer.py:68-90` 使用 `CaseCapture` 生成 cases，然后把模板中的 `__ORIGINAL_HARNESS__`、`__TARGET_PATCHES__`、`__DEFAULT_CASES__` 替换成字面量；没有调用任何 LLM。

运行版本 pyc 的 `transform` 只是增加了 `cases` 参数和 `argfuzz_corpus` 来源选择：

```python
case_source = "argfuzz_corpus" if cases else "deterministic_defaults"
```

仍然没有 LLM 转换。

#### （5）配置中没有 LLM 段落

`configs/scan.yaml` 与 `configs/scan_stress.yaml` 的 `generation` 段：

```yaml
generation:
  max_repair_attempts: 1
  use_template_first: true
  use_llm_transform: false
```

[HEAD] `config.py:41-45` 的 `GenerationConfig` 只有三个字段：

```python
max_repair_attempts: int = 2
use_template_first: bool = True
use_llm_transform: bool = True
```

没有 provider、base_url、model、temperature、top_p、max_tokens、seed、prompt_version 等字段。`prompt_templates/` 目录为空；仓库没有 `llm_transformer.py`；输出目录没有保存任何对应 prompt。

### 2.3 可直接写入文章的表述

建议用下面这段替代“LLM 驱动适用性分析/转换”的表述：

> 并发预言机 P0 采用**规则预筛选 + 固定模板转换**：适用性分析由 `SuitabilityRules` 根据 harness AST 特征产生候选场景和 Oracle 候选，再由 `ContractRegistry` 聚合契约状态；harness 转换由固定模板完成。LLM 适用性分析在代码中保留为 P1 占位接口（`LLMSuitabilityAnalyzer`），本轮实验**未启用**，因此不报告 LLM prompt、模型版本、采样参数或 LLM 回退记录。若需要 LLM 辅助分析，必须在 P1 实现后重新运行并单独记录。

如果文章第三章的“生成模型”指的是 **harness 生成**而不是并发预言机，请使用 2.6 节的 `fuzz_api_one` 实际配置；两者不要混写。

### 2.4 如果后续要启用 LLM：建议记录字段

下表中的字段是**建议新增**，不是当前已有记录。设计上最适合存放的位置是 `SuitabilityReport.model_config`、`prompt_version`、`raw_llm_response`，以及 `ConcurrentHarnessArtifact.transform_record`。

| 字段 | 建议位置 | 示例 | 当前状态 |
|---|---|---|---|
| `llm.enabled` | config | `true` | 未启用 |
| `llm.provider` | config | `openai-compatible` | 未配置 |
| `llm.base_url` | config | `https://.../v1` | 未配置 |
| `llm.model` | config | 例如 `gpt-5-codex` | 未配置 |
| `llm.model_version` / snapshot | config + 每次调用 | 供应商返回的版本或快照名 | 未记录 |
| `llm.temperature` | config | `0.0` | 未配置 |
| `llm.top_p` | config | `1.0` | 未配置 |
| `llm.max_tokens` | config | `4096` / `6000` | 未配置 |
| `llm.seed` | config + 每次调用 | `null` 或固定整数 | 未配置 |
| `llm.prompt_version` | `SuitabilityReport.prompt_version` | `suitability-v1` | 占位常量存在，未使用 |
| `llm.output_format` | config | `json` | 未配置 |
| `llm.parse_strategy` | config | 严格 JSON → 提取首个对象 → 有限重试 | 未实现 |
| `llm.fallback` | config | `rules_only` | 代码骨架存在，未启用 |
| `prompt_hash` | 每次调用 | `sha256(prompt)` | 未记录 |
| `request_tokens` / `response_tokens` | 每次调用 | 供应商 `usage` | 未记录 |
| `raw_response_path` | 每次调用 | `outputs/llm_calls/<id>.json` | 未记录 |
| `parse_ok` / `validation_errors` | 每次调用 | `true` / `[]` | 未记录 |
| `fallback_used` | 每次调用 | `false` | 未记录 |

#### LLM 适用性输出的期望结构

设计文档 [guidance.md §3.4] 要求的 JSON 结构如下（当前未实现）：

```json
{
  "artifact_id": "tf_api_001",
  "decision": "READY",
  "api_categories": ["DATASET_PIPELINE", "PYTHON_CALLBACK"],
  "candidate_scenarios": [
    {
      "scenario_id": "tfdata_parallel_map",
      "oracles": ["O2", "O3", "O6"],
      "state_model": "INDEPENDENT_INPUTS",
      "priority": "HIGH",
      "reason": "..."
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

`APISuitabilityAnalyzer.validate_llm_result`（[HEAD] `api_suitability_analyzer.py:146-165`）已经实现了以下校验：

- 必须是 JSON object；
- `decision` 必须属于 `READY / CONDITIONAL / LOW_PRIORITY / INSUFFICIENT_INFO`；
- `candidate_scenarios` 必须是 list；
- `scenario_id` 必须属于 `SCENARIO_IDS`；
- `oracles` 必须属于已注册 ID；
- `state_model` 必须属于 `STATE_MODELS`；
- `confidence` 必须在 `[0,1]`。

#### LLM 回退骨架（已实现但不是“实际启用的回退记录”）

[HEAD] `api_suitability_analyzer.py:40-48`：

```python
if self.llm_analyzer is not None and self.llm_client is not None:
    try:
        ...
        llm_result = self.llm_analyzer.analyze(...)
        if llm_result is not None:
            self.validate_llm_result(llm_result)
    except Exception as exc:
        llm_error = f"{type(exc).__name__}: {exc}"
        llm_result = None
```

失败时规则结果保留，`rationale` 追加 `LLM suitability failed; rules retained: ...`。这套骨架可以作为 P1 的回退路径，但本轮没有产生任何实际 LLM 回退记录。

#### 建议的配置模板（标注为 proposed，不是当前生效配置）

```yaml
# 仅作为实现 P1 后的配置草案；当前 scan.yaml 中没有这一段。
llm_suitability:
  enabled: false                 # 当前实际值：false / 未启用
  provider: openai-compatible
  base_url: https://your-endpoint/v1
  model: your-model
  model_version: your-model-version
  temperature: 0.0
  top_p: 1.0
  max_tokens: 4096
  seed: null
  prompt_version: suitability-v1
  output_format: json
  parse: strict_json_then_extract_object
  fallback: rules_only

llm_transform:
  enabled: false                 # 当前实际值：generation.use_llm_transform=false
  model: your-model
  temperature: 0.0
  max_tokens: 32000
  repair_attempts: 2
  output_format: python_code_block
  fallback: fixed_template
```

> 仅仅在 YAML 中增加字段不会生效：`pipeline.py` 仍然传 `llm_client=None`；`LLMSuitabilityAnalyzer` 和 `build_prompt` 仍是 `NotImplementedError`；`transformer.py` 仍是固定模板。实现 P1 需要同时改代码。

### 2.5 “输出提取及回退记录”的当前边界

- **输出提取**：并发预言机没有 LLM 输出，因此没有“提取”记录。
- **回退**：规则结果的保留路径存在（`llm_error` + rules retained），但本轮没有触发，因为根本没有 LLM 调用。
- **可交付的记录**：只有 696 份 `llm_result=null` / `model_config={}` / `prompt_version=null` / `raw_llm_response=null` 的 suitability JSON，可作为“未启用 LLM”的直接证据。

### 2.6 模块外补充：如果第三章的“生成模型”指 harness 生成

`concurrent_oracle` 消费的 362 个 harness 来自 `fuzz_api_one`，其实际 LLM 配置如下（来自 `fuzz_api_one/llm_gen_harness.py`、`batch_gen_harness_summary.json`、`batch_gen_harness.log`）：

| 项目 | 实际值 |
|---|---|
| 生成模块 | `fuzz_api_one/llm_gen_harness.py` |
| 模型 | `gpt-5.4` |
| API 模式 | `responses` 优先，失败后 `chat` |
| temperature | `0` |
| max output tokens | `32000` |
| 输入 | YAML + API 文档 txt + Atheris 说明 |
| YAML 参数空间构建 | `build_yaml` 阶段使用 `gpt-5-codex`；Stage C 逐 API 调用，Stage D 在原实验中因文档名不匹配实际为纯复制（0 次 LLM 调用） |
| system prompt | 约 4,770 字符 |
| user prompt | 平均约 28.4k 字符（24.5k–35.8k） |
| repair attempts | `2`（实际 362/362 都用了 1 轮 repair） |
| 输出提取 | 正则提取 fenced Python code block；也兼容 JSON `code`/`python` 字段 |
| 校验 | 语法、`atheris` 首 import、`TestOneInput(data)`、`atheris.Setup/Fuzz`、API marker、TensorFlow import、`instrument_imports` |
| 回退 | `responses` 失败→`chat`；校验失败→repair prompt；repair 仍失败→抛错 |
| 实际调用量 | 362 个 harness × 2 次（初稿 + 1 次 repair）= **724 次** |
| 实际耗时 | **2 h 10 min**（workers=2，06:37:53–08:47:49） |

如果文章第三章描述的是“LLM 生成 harness”，应使用上表；如果描述的是“LLM 辅助并发适用性分析/转换”，应使用 2.3 节的“未启用”表述。两者属于不同模块，不能互相替代。

---

## 3. M03：PSR 状态映射、已有 fuzz 追溯、配置耗时、命名/别名、TF 资源/Dataset/callback

### 3.1 要求拆解

MO3 要求补充：

1. **PSR 状态与底层调用的映射**；
2. **已有 fuzz 追溯接入**；
3. **不同配置的耗时记录**；
4. **自动命名和别名待传方案**；
5. 明确 **TensorFlow 资源、Dataset、callback 配置需要重新实现和验证，不能作为当前完成度统计**。

### 3.2 PSR 状态与底层调用的映射：目前只有“路径引用”，没有“状态映射”

#### 当前数据流

```text
harness 头部 # source_yaml=/root/yaml/<api>.yaml
        │
        ▼
HarnessLoader.load()  ──► HarnessArtifact.psr_path
        │
        ▼
HarnessAnalyzer.analyze() ──► target_calls / input_variables / ...
        │
        ▼
CaseCapture（capture 模式运行原始 harness）
        │
        ▼
cases.json: target_calls[].args/kwargs 的 dtype/shape/value/sha256
```

- [HEAD] `harness_loader.py:81-89` 会从 harness 头部 `# source_yaml=...` 提取 PSR 路径；`HarnessArtifact.psr_path` 保存该路径。
- 初始扫描的 362 个 artifact **全部有 `psr_path`**（362/362）；报告中也打印 `psr: /root/yaml/...`。
- 但是 362 个 artifact 的 `metadata` **全部为空**（`metadata_nonempty=0`），`tensorflow_version` 字段也未填。
- `HarnessAnalysis` 的字段只有 `target_calls`、`input_variables`、`resource_variables`、`callback_functions`、`output_expressions`、`cleanup_expressions`、`imports`、`tensor_construction`、`dataset_operations`、`resource_lifecycle`；**没有 PSR 状态字段**。
- `ConcurrentTestPlan.metadata` 只有 `api_name`、`priority`、`reason`、`config_id`、`api_categories`、`unimplemented_oracles`；**没有 PSR 状态/参数别名**。
- `cases.json` 的 case 字段有 `case_id`、`data_b64`、`byte_sha256`、`origin_path`、`source`、`target_calls`、`case_source`、`eligible`、`ineligible_reason`、`arg_conversions`；**没有 `psr_state_id` 或 `psr_param`**。

因此，当前的映射是：

> **PSR 文件路径 → 原始 harness → capture 模式产生的 target call 参数描述**

而不是：

> **PSR 中某个参数状态（rank/dtype/shape/取值）→ 某次底层调用的具体参数**

#### 建议的 PSR 映射方案（待传）

| PSR YAML 字段 | 对应的 captured call 字段 | 建议新增 ID/字段 |
|---|---|---|
| `params.<name>.dtype_choices` | `target_calls[].args[].dtype` | `psr_dtype_choice` |
| `params.<name>.shape_spec_by_rank` | `target_calls[].args[].shape` | `psr_shape_spec` |
| `params.<name>.semantic_role` | 参数角色（primary/aux/index/...） | `psr_semantic_role` |
| `test_ranks` | case 的实际 rank | `psr_rank` |
| `shape_vars` | 实际 shape 绑定 | `psr_shape_binding` |
| `constraints` | 期望关系 / Oracle contract | `psr_constraint_id` |
| `tf.param_mapping` / `inverse_mapping` | 底层 raw op 调用名 | `psr_callable_alias` |
| `api_name` + `psr_path` | artifact | `psr_hash` |

建议：

- 对每个 case 计算 `psr_state_id = short_hash(canonical_json({api_name, psr_hash, rank, dtype, shape, param_values}))`；
- 在 `HarnessArtifact.metadata` 增加 `psr_hash`、`psr_version`；
- 在 `ConcurrentTestPlan.metadata` 增加 `psr_state_ids`、`psr_selection`；
- 在 `cases.json` 的每个 target call 增加 `psr_state_id`、`psr_param_map`；
- 缓存 key 至少包含 `harness_hash + psr_hash + prompt_version + tensorflow_version`（设计文档 §3.6 已提出，但当前未实现）。

### 3.3 已有 fuzz 追溯接入：主扫描未接入，压力扫描部分接入

运行版本的 `CaseCapture.corpus_cases`（[RUN-PYC] `case_capture.cpython-310.pyc`）实现了 ArgFuzz corpus 加载：

- 输入：`corpus_root=/root/corpus_experiment_save`，按 `api_name` 找子目录；
- 只保留非空普通文件；按 `(文件大小, 文件名)` 排序；
- 若文件数 > `max_cases`（`cases_per_api=12`），按大小排序后**等距抽样**，覆盖大小两端；
- 生成 case：
  - `case_id = corpus_%03d`
  - `data_b64`
  - `byte_sha256`
  - `origin_path`
  - `source = argfuzz_corpus`
- 若目录不存在或为空，回退到 `default_cases()`（8 组固定 64 字节）。

实际使用情况：

| 扫描 | capture 文件 | case 来源 | cases | eligible | ineligible | 截断参数 |
|---|---:|---|---:|---:|---:|---:|
| 初次扫描 `vuln_scan` | 343 | `deterministic_defaults` 343 | 2744 | 1759 | 913 | 0 |
| retry `vuln_scan_retry` | 223 | `deterministic_defaults` 223 | 1784 | 799 | 913 | 0 |
| retry2 | 13 | `deterministic_defaults` 13 | 104 | 52 | 52 | 0 |
| retry3 | 1 | `deterministic_defaults` 1 | 8 | 8 | 0 | 0 |
| parallel2 | 42 | `deterministic_defaults` 42 | 336 | 189 | 147 | 0 |
| stress pilot | 15 | `argfuzz_corpus` 15 | 180 | 126 | 54 | 3 |
| stress pilot2 | 15 | `argfuzz_corpus` 15 | 180 | 126 | 54 | 3 |
| stress full | 101 | `argfuzz_corpus` 101 | 1212 | 652 | 464 | 0 |

解释：

- `/root/corpus_experiment_save` 下有 **362 个 API 目录**，语料是存在的；
- 但初次 362-API 扫描发生在 corpus 接入之前（或运行的是不含 `corpus_dir` 的版本），因此 **主扫描全部使用 8 组固定字节**；
- 只有 stress 系列（48 个 API）使用了 `argfuzz_corpus`；
- 因此，**不能把主扫描的 362 个 API 计入“已有 fuzz 语料追溯覆盖率”**。如果文章要写“接入 ArgFuzz 语料”，应写成“在 48 个压力扫描 API / 101 个计划上接入”，而不是“362 个 API 全部接入”。

已保留的追溯字段：

- `origin_path`：语料原始文件路径；
- `byte_sha256`：原始字节哈希；
- `case_id`：`corpus_XXX`；
- `source`：`argfuzz_corpus`；
- `case_source`：`captured_target_call` / `no_target_call` / `bytes_fallback`；
- `eligible` / `ineligible_reason`；
- `arg_conversions`（例如 `ndarray_to_tensor`）；
- `capture_error` / `capture_exceptions` / `target_exception`（当 capture 失败时）。

仍缺的统计字段（设计文档 R02/R03 提出，当前未实现）：

- `eligible_case_count`
- `serial_case_count`
- `concurrent_case_count`
- `compared_case_count`
- `case_selection_count`
- `psr_state_count`
- `configuration_pair_count`
- 每个 case 的串行/并发参与次数
- 每个 case 的 `psr_state_id`

### 3.4 不同配置的耗时记录：只有“每 API 总耗时”，没有逐配置/逐计划耗时

`scan_harnesses` 在 `scan_results.jsonl` 中为每个 API 记录 `elapsed_seconds`，但没有为每个 plan/config 记录单独的进程总耗时。`execution.json` 有每个 task 的 `started_at` / `finished_at`，但没有进程级 `duration`。因此下表是“每 API 总耗时”，不是“每配置净耗时”。

| 扫描 | API 数 | 实际完成 result 数 | 总耗时 | 均值 | p50 | max | 实际执行配置 |
|---|---:|---:|---:|---:|---:|---:|---|
| `vuln_scan`（初次） | 362 | 309 | 2533.4 s | 7.0 s | 7.7 s | 23.1 s | C03 |
| `vuln_scan_retry` | 223 | 189 | 1574.4 s | 7.06 s | 7.65 s | 16.8 s | C03 |
| `vuln_scan_retry2` | 13 | 13 | 129.7 s | 9.98 s | 10.15 s | 10.4 s | C03 |
| `vuln_scan_retry3` | 1 | 1 | 7.5 s | 7.51 s | 7.51 s | 7.51 s | C03 |
| `vuln_scan_parallel2` | 14 | 39 | 324.9 s | 23.2 s | 23.1 s | 30.6 s | C03 / C01 / C04 |
| `vuln_scan_stress_full` | 48 | 83 完成 + 2 SETUP_ERROR | 999.3 s | 20.8 s | 20.7 s | 62.6 s | C08 / C09 |

配置定义（[RUN-PYC] `constants`，C00–C10，格式为 `external/intra/inter/tf_data`）：

| 配置 | external_threads | intra_op_threads | inter_op_threads | tf_data_parallel_calls |
|---|---:|---:|---:|---:|
| C00 | 1 | 1 | 1 | 1 |
| C01 | 1 | 4 | 1 | 1 |
| C02 | 1 | 1 | 4 | 1 |
| C03 | 2 | 1 | 1 | 1 |
| C04 | 4 | 1 | 1 | 1 |
| C05 | 4 | 2 | 2 | 1 |
| C06 | 1 | 1 | 1 | 4 |
| C07 | 4 | 1 | 1 | 4 |
| C08 | 8 | 1 | 1 | 1 |
| C09 | 1 | 8 | 1 | 1 |
| C10 | 4 | 4 | 2 | 1 |

注意：

- 初次扫描虽然为每个 API 生成了 C03/C04 两套计划，但 `max_plans=1`（`scan_harnesses` 默认）只执行第一个计划，实际执行的基本都是 C03；
- 因此不能用初次扫描的 `elapsed_seconds` 比较 C03 与 C04；
- stress 扫描同时执行 C08（external=8）和 C09（intra=8），但 `elapsed_seconds` 是每 API 总计，无法拆分到两个配置；
- 建议后续在 `ExecutionObservation` / `scan_results` 中增加 `duration_seconds`（按 plan/config/run_id），并在报告中按 config 聚合。

### 3.5 自动命名和别名：当前规则清楚，但 PSR/参数别名还没有传递字段

#### 当前命名规则

- `artifact_id = slug(api_name) + "_" + harness_hash[:8]`（[HEAD] `schemas.py:367-369`）
  - 例：`tf.math.log1p_a0fd52a7`
- `plan_id = artifact_id + "__" + scenario + "__" + %02d`（[HEAD] `schemas.py:372-373`）
  - 例：`tf.math.log1p_a0fd52a7__independent_api_calls__00`
- `run_id = prefix + "_" + YYYYMMDD_HHMMSS + "_" + uuid[:8]`
  - 例：`test_20261030_052227_b9c33621`
- `case_id = case_%03d`（默认）或 `corpus_%03d`（语料）
- 输出目录：
  - `generated_harnesses/<plan_id>/concurrent_harness.py`
  - `captured_cases/<plan_id>/cases.json`
  - `executions/<plan_id>/<mode>/<run_id>/`
  - `oracle_results/<plan_id>.json`
  - `reports/<plan_id>_report.md`

#### 当前别名解析

- `HarnessLoader.infer_api_name`（[HEAD] `harness_loader.py:31-61`）按以下顺序推断 API：
  1. harness 头部 `# api_name=` / `# target_api=`；
  2. 头部 `# source_yaml=` 的 stem；
  3. 文件名 `llm.<api>.py`；
  4. 源码中最像 API 入口的 `tf.*` 调用（排除 `tf.constant`、`tf.identity`、`tf.convert_to_tensor`、`tf.raw_ops`）；
  5. 文件 stem。
- `HarnessAnalyzer._build_alias_map`（[HEAD] `harness_analyzer.py:77-97`）解析：
  - `import tensorflow as tf` → `tf`；
  - `import tensorflow.math as m` → `m` → `tf.math`；
  - `from tensorflow import math` → `math` → `tf.math`；
  - `from tensorflow.math import reduce_sum` → `reduce_sum` → `tf.math.reduce_sum`。
- `_resolve_alias` / `_target_match`（[HEAD] `harness_analyzer.py:100-120`）允许：
  - 精确匹配 `tf.math.log1p`；
  - `tf.raw_ops.Log1p` 通过叶子名匹配；
  - 同一叶子名的 composite 调用。

#### 待传方案（PSR/参数别名）

当前没有字段把 PSR 参数名、raw op 参数名、captured call 参数名串起来。建议：

| 层级 | 当前 | 建议新增 |
|---|---|---|
| `HarnessArtifact.metadata` | 空 | `psr_path`, `psr_hash`, `psr_version`, `api_aliases` |
| `ConcurrentTestPlan.metadata` | `api_name`, `priority`, `reason`, `config_id`, `api_categories`, `unimplemented_oracles` | `psr_state_ids`, `psr_selection`, `param_aliases`, `callable_alias` |
| `cases.json` case | `case_id`, `data_b64`, `byte_sha256`, `origin_path`, `source` | `psr_state_id`, `psr_param_map`, `source_harness_hash` |
| `target_calls[]` | `call`, `args`, `kwargs` | `raw_op_name`, `psr_param_names`, `param_mapping` |
| `OracleResult.metadata` | `config_id`, `run_id`, `mode`, `trials`, `baseline_run_id`, `test_run_id`, `contract_status`, `task_progress`, `observation_complete`, `baseline_runtime` | `psr_state_ids`, `psr_hash`, `alias_resolution` |

建议的自动命名/别名规则：

```text
alias_id = slug(api_name) + ":" + psr_state_id
plan_psr_key = plan_id + ":" + short_hash(sorted(psr_state_ids))
case_psr_key = case_id + ":" + psr_state_id
```

### 3.6 TensorFlow 资源、Dataset、callback：合同存在，但后端未实现，不能计入完成度

#### 场景能力清单

[HEAD] `constants.py:98-142` 与 [RUN-PYC] `constants` 的 `SCENARIO_CAPABILITIES`：

| 场景 | backend implemented | 运行版本 Oracle | 当前状态 |
|---|---|---|---|
| `independent_api_calls` | ✅ True | O2, O3 | 已执行 |
| `shared_readonly_input` | ✅ True | O3 | 已执行 |
| `parallel_kernel_execution` | ✅ True | O2, O3, O8 | 压力扫描执行 |
| `shared_resource_access` | ❌ False | — | P1，`shared mutable TensorFlow resource backend is P1` |
| `tfdata_parallel_map` | ❌ False | — | P2，`tf.data consumption backend is P2` |
| `tfdata_interleave` | ❌ False | — | P2 |
| `tfdata_early_stop` | ❌ False | — | P2 |
| `python_callback_lifecycle` | ❌ False | — | P2，`callback lifecycle backend is P2` |
| `exception_recovery` | ❌ False | — | P2 |
| `invalid_input_cpu` | ❌ False | — | P2 |

#### 合同文件现状

`contracts/tensorflow/` 中存在以下合同，但 `last_verified_version` 全部为 `null`，且没有对应执行后端：

| 文件 | 条目 | 状态 |
|---|---|---:|
| `cpu_ops.json` | 6 条合同（覆盖 5 个 API）：`tf.math.log1p`（独立 + 共享只读）、`tf.nn.relu`、`tf.math.reduce_sum`、`tf.linalg.matmul`、`tf.argsort` | SUPPORTED |
| `callbacks.json` | `tf.py_function`、`tf.numpy_function` | SUPPORTED（合同） |
| `tf_data.json` | `Dataset.map`、`Dataset.interleave`、`Dataset.from_generator` | SUPPORTED（合同） |
| `resources.json` | `tf.Variable`、`tf.data.Iterator`、`tf.compat.v1.Session` | UNKNOWN |
| `threading.json` | `set_intra_op_parallelism_threads`、`set_inter_op_parallelism_threads` | SUPPORTED |

#### 本轮扫描的实际排除

- NOT_APPLICABLE API：**19 个**；典型原因：
  - `decision is INSUFFICIENT_INFO; no callable plan`（13 个）；
  - `no plans generated`（6 个），跳过场景包括：
    - `shared_resource_access: "shared mutable TensorFlow resource backend is P1"`
    - `tfdata_parallel_map: "tf.data consumption backend is P2"`
    - `tfdata_early_stop: "tf.data lifecycle backend is P2"`
    - `exception_recovery: "exception isolation/recovery backend is P2"`
- 最终报告明确写明：
  > “Dataset, callback, shared mutable resource and invalid-input backends are not implemented; those APIs are skipped or warned about.”

#### 可直接写入文章的表述

> 本版本完成了 `independent_api_calls`、`shared_readonly_input`、`parallel_kernel_execution` 三类 CPU 纯算子/Tensor 场景；`shared_resource_access`、`tf.data` 三类场景、`python_callback_lifecycle`、`exception_recovery`、`invalid_input_cpu` 仍只有合同/设计，没有执行后端，因此不计入当前完成度。`tf.Variable`、`tf.data.Iterator`、`tf.compat.v1.Session` 的合同状态为 UNKNOWN，不能据此判定并发安全或不安全。

---

## 4. MO4：数值复核、特殊值、逐调完整性、串行稳定性、两后端适用 API/dtype、采样/截断

### 4.1 大数量全量或分块数值复核：当前只做“完整值逐元素比较”，超过 65,536 元素只看哈希

#### 当前比较器算法

[HEAD] `result_comparator.py`：

1. **哈希快速路径**：`compare_described` 对 `tensor`/`ndarray` 先比较 `dtype`、`shape`；如果 `sha256` 相同，直接 `equal=True`。
2. **完整值比较**：如果哈希不同且 `value` 存在，递归调用 `_compare_values`：
   - dict：key 集合必须一致，逐 key 递归；
   - list/tuple：长度必须一致，逐 index 递归；
   - float：`atol=1e-6`、`rtol=1e-5`；NaN mask；Inf 符号/值；
   - int：`exact_integer=True`；
   - 其他：直接相等比较。
3. **截断值**：如果 `value` 缺失/被截断，返回 `values truncated; hash differs`，并标记 `inconclusive=True`。
4. **差异上限**：`_result` 只保留前 **50** 条 diff。

#### 截断边界

- [RUN-PYC] 模板 `_describe` / `_describe_input` 的阈值为 **65,536 个元素**：
  - 小于等于 65,536：保存完整 `value`；
  - 大于 65,536：设置 `value_truncated=True`，不保存完整值。
- `_reconstruct_arg` 遇到 `value_truncated` 会拒绝回放：
  > `captured argument is truncated and cannot be replayed`
- `case_capture._check_arg_desc` 把 `value_truncated` 的 tensor/ndarray 标记为 ineligible：
  > `tensor argument data is truncated`
- 实际数据：
  - 主扫描 / stress full：`truncated_args=0`；
  - stress pilot / pilot2：各 3 个截断参数（各 180 个 case、126 eligible、54 ineligible）；
  - 相关 warning：`tensor argument data is truncated` 各 3 次。

#### 结论与缺口

- 当前**没有**“大张量全量或分块数值复核”：超过 65,536 元素时只比较哈希，哈希不同即 `inconclusive`，不会做分块容差比较。
- 设计文档 [fix_guidance.md R04] 建议的“对大浮点输出采用分块容差比较；哈希相同可快速判等，哈希不同不能直接判数值错误；缺少完整值时保留 INCONCLUSIVE”**尚未实现**。
- 建议补充的实验/字段：
  - `chunk_size`、`chunk_count`、`compared_elements`、`first_diff_index`；
  - 分块比较结果：每块 `equal/inconclusive/diff`；
  - 大张量（>65,536）单独一组实验，记录“全量 vs 分块 vs 仅哈希”的差异分布；
  - 对 `value_truncated` 的 case 明确标记“未完成数值复核”，不能计入 CLEAN。

### 4.2 特殊值位置检查：完整值按位置检查 NaN/Inf；`-0.0` 不单独区分

- `_compare_values` 对 list/tuple 按 index 递归，因此对完整值而言，NaN mask 和 Inf 符号是**按位置**检查的。
- `-0.0` 与 `+0.0` 会被 float 容差判为相等（`abs(-0.0 - 0.0) = 0`），当前没有单独的 `-0.0` 符号检查。
- 对截断值（>65,536）不做位置检查，只比较哈希。
- 实际捕获的参数值中特殊值出现次数（按扫描、按 captured arg 统计，不是比较位置数）：

| 扫描 | `-inf` | `inf` | `nan` | `-0.0` |
|---|---:|---:|---:|---:|
| `vuln_scan`（初次） | 13,945 | 597 | 713 | 1,201 |
| `vuln_scan_retry` | 863 | 264 | 336 | 839 |
| `vuln_scan_retry2` | 28 | 34 | 3 | 277 |
| `vuln_scan_parallel2` | 1,278 | 21 | 0 | 51 |
| `vuln_scan_stress_full` | 764 | 26 | 9 | 4,140 |

> 这些计数说明 case 中确实包含特殊值，但不能直接当作“特殊值位置检查覆盖数”。当前没有记录“哪些位置被检查、哪些位置被截断”。

建议补充字段：

- `special_value_positions`：NaN/±Inf/-0.0 的位置列表或分块统计；
- `nan_mask_checked`、`inf_sign_checked`、`neg_zero_checked`；
- `value_truncated` 与 `position_check_complete` 的联合标记。

### 4.3 逐调调用完整性验证：已实现并已量化

#### 实现机制

- 模板 `_invoke_once` 记录每次 target call 的：
  - `case_id`、`expected_calls`、`call_index`；
  - `TARGET_ENTER` / `TARGET_RETURN` / `TARGET_EXCEPTION` / `DESCRIBE_START` / `DESCRIBE_END`；
  - `outputs`、`exceptions`、`status`、`complete`；
  - `actual_calls`。
- `_summarize_trial_status` 将 trial 状态归为：
  - `COMPLETED`、`PARTIAL`、`EXCEPTION`、`INCOMPLETE`、`NO_TARGET_CALL`。
- `ConcurrentRunner` 汇总 `task_progress`：
  - `submitted`、`started`、`completed`、`partial`、`failed`、`pending`。
- `observation_complete` 为 `False` 时，O2/O3 不会给出 PASS；`baseline_covers_tasks` 要求 baseline 覆盖 test 实际使用的 input_id 且 `complete=True`。

#### 实际数据

- 742 份唯一 `combined_observation.json`：
  - `observation_complete=True`：**704**；
  - `observation_complete=False`：**38**。
- 任务进度合计：
  - submitted **2754**；
  - started **2754**；
  - completed **2685**；
  - failed **67**；
  - pending **0**；
  - partial **0**。
- task 状态：
  - `COMPLETED` **2685**；
  - `EXCEPTION` **67**；
  - `NO_TARGET_CALL` **2**。

> 这组数据可以直接作为“逐调调用完整性验证”的实测记录。但注意：742 份 observation 包含初次、retry、parallel、stress 多批扫描，不能简单当作 742 个独立 API。

### 4.4 串行稳定性记录：只有“串行 baseline 资格检查”，没有重复串行稳定性统计

#### 当前实现

[RUN-PYC] `runner.run`：

- baseline 运行时 `external_threads=1`；`parallel_kernel_execution` 场景下 `intra_op_threads=1`、`inter_op_threads=1`；
- baseline 最多尝试 **3 次**（`for _attempt in range(3)`）；
- `_baseline_bad_cases` 找出 baseline 中 `complete=False` 的 case；
- `mark_ineligible(..., reason="baseline_not_complete")` 把这些 case 标记为 ineligible，并加入 `baseline_excluded`；
- 如果还有 case 被排除就继续重试；没有坏 case 就停止；
- 最后 `baseline_covers_tasks(baseline_obs, test_obs)` 要求：
  - baseline 执行状态为 `COMPLETED`；
  - test 实际使用的每个 `input_id` 在 baseline 中 `complete=True`。

#### 实际数据

- 有 `baseline_qualification` 记录的 207 份 observation：
  - `attempts=1`：**183**；
  - `attempts=2`：**24**；
  - `attempts=3`：**0**；
  - `excluded_cases` 合计：**56**。
- `process_repetitions`：
  - 642 份 observation 没有该字段（默认 1 次，未重复）；
  - 100 份 observation 为 `process_repetitions=2`（stress 扫描）；
  - 这 100 份对应 200 个 test run：`COMPLETED` 196、`SETUP_ERROR` 4。
- **注意**：`process_repetitions` 重复的是**并发 test 进程**，不是串行 baseline。串行 baseline 仍然只跑一轮（最多 3 次用于排除坏 case），没有“多次串行重复 + 方差/稳定率”的统计。

#### 缺口与建议

- 当前没有：
  - `serial_repeats`（串行重复次数）；
  - `serial_mismatch_count` / `serial_mismatch_rate`；
  - `serial_variance` / `serial_output_hash_distribution`；
  - “串行稳定后才进入并发比较”的独立状态。
- 建议补充：
  - 对每个 plan 至少跑 3–5 次串行 baseline；
  - 记录每次的 `run_id`、输出哈希、耗时、是否一致；
  - 只有串行稳定（全部一致）的 case 才进入并发比较；
  - 报告 `serial_stable_case_count`、`serial_unstable_case_count`、`serial_stability_rate`。

### 4.5 两后端兼容的适用 API 和 dtype：本目录只有 TensorFlow CPU 后端

#### 结论先说

- `/root/concurrent_oracle` 是 **TensorFlow CPU** 实现；`environment.device=CPU`、`cuda_visible_devices=-1`、`require_cpu_only=true`。
- 实际运行记录 `tensorflow_version=2.21.0-dev0+selfbuilt`，`gpu_devices=[]`。
- 本目录没有 PyTorch runtime/adapter；与 PyTorch 版本的兼容性只在 `HarnessArtifact`、`SuitabilityReport`、`ConcurrentTestPlan`、`ExecutionObservation`、`OracleResult` 等 schema/CLI 层面对齐，**未在本目录做 PyTorch 实测**。
- 因此，文章如果写“两后端兼容”，应写成：
  > “TensorFlow CPU 后端已完成本轮实测；PyTorch 后端沿用相同的数据模型/CLI 约定，但本轮未在 `/root/concurrent_oracle` 中重新验证。”

如果“两后端”指 CPU/GPU，则更直接：**本模块只做 CPU，不包含 GPU/CUDA 后端**（设计文档 §1.3 明确第一阶段只做 CPU）。

#### 适用 API 现状

- 合同文件 `cpu_ops.json` 只列出 6 条 CPU op（`tf.math.log1p`、`tf.nn.relu`、`tf.math.reduce_sum`、`tf.linalg.matmul`、`tf.argsort` 等），不是全量 362 个 API 的合同。
- 实际扫描覆盖 362 个 API，但决策分布是：
  - `READY` 4；
  - `CONDITIONAL` 345；
  - `INSUFFICIENT_INFO` 13。
- `CONDITIONAL` 的含义是“需要合同/配置/额外信息”，不能当作“已确认适用”。
- `NOT_APPLICABLE` 19 个 API 已明确跳过。

#### dtype 现状

- 当前没有正式的 dtype allowlist；captured arg 的 `dtype` 是自由字符串，比较器按 float/int/其他分别处理。
- 实际捕获的 dtype 共 **21 种**（主扫描 20 种，stress 扫描 20 种，并集 21 种）：

| dtype | 出现次数（全部 capture，含 retry/stress，可能重复计数） |
|---|---:|
| float32 | 2822 |
| float64 | 1342 |
| int32 | 1323 |
| float16 | 1045 |
| int64 | 791 |
| bfloat16 | 577 |
| complex128 | 404 |
| complex64 | 366 |
| uint8 | 224 |
| int8 | 223 |
| int16 | 213 |
| bool | 179 |
| uint16 | 100 |
| string | 32 |
| qint8 | 24 |
| uint64 | 22 |
| uint32 | 21 |
| qint16 | 17 |
| quint8 | 16 |
| quint16 | 8 |
| qint32 | 4 |

- 支持但没有充分验证的 dtype：`string`（比较器按普通值比较）、量化类型（`qint8/quint8/qint16/quint16/qint32`）、`bfloat16/complex64/complex128`。
- 不支持/未覆盖的情况：
  - `value_truncated` 的 tensor/ndarray 直接 ineligible；
  - `unsupported_input` 参数类型 ineligible（初次扫描 13 次）；
  - 没有 dtype 级合同，不能声称某个 dtype 的并发语义已验证。

#### 可直接写入文章的表述

> 本轮只在 TensorFlow CPU（2.21.0-dev0+selfbuilt）上完成并发预言机实验；适用 API 以合同文件和 P0 场景能力为准，而不是以扫描到的 362 个 harness 数量为准。dtype 覆盖为 capture 中实际出现的 21 种，未建立逐 dtype 的并发语义合同；`string`、量化 dtype、`bfloat16`、复数类型的并发行为仍需单独验证。PyTorch 后端不在本目录，仅在数据模型/CLI 层面对齐，未做本轮实测。

### 4.6 采样、类型转换、输入截断范围

| 项目 | 当前实现 |
|---|---|
| 默认 case | `DEFAULT_INPUT_CASES`：8 组固定 64 字节 |
| 语料采样 | `corpus_cases`：按大小排序后等距抽样，最多 `cases_per_api=12` |
| 支持参数 kind | `tensor`、`ndarray`、`sparse_tensor`、`ragged_tensor`、`dtype`、`tensor_shape`、`python`、`list`、`tuple`、`dict` |
| 类型转换 | `ndarray` → `ndarray_to_tensor`；`ragged flat_values` 递归检查；`sparse` 检查 dtype/dense_shape/indices/values |
| 截断阈值 | 65,536 个元素（`_describe` / `_describe_input`） |
| 截断后行为 | `value_truncated=True`；`_reconstruct_arg` 拒绝回放；`_check_arg_desc` 标记 ineligible |
| 失败保留 | `capture_error`、`capture_exceptions`、`target_exception`、`ineligible_reason`、`arg_conversions` |

实际 capture 统计见 3.3 节。主扫描 2744 个 case 中 1759 eligible；stress full 1212 个 case 中 652 eligible。大量 ineligible 的主要原因是 `capture_status=target_exception`、`no captured target call`、`unsupported_input`。

### 4.7 建议的 MO4 补实验矩阵（待实测）

| 实验 | 目的 | 最小规模 | 记录字段 |
|---|---|---|---|
| 完整值 vs 分块比较 | 证明 >65,536 元素不只是哈希 | 10–20 个大张量 API | `chunk_size`、`chunk_count`、`first_diff_index`、`inconclusive` |
| 特殊值位置矩阵 | 验证 NaN/±Inf/-0.0 在具体位置的行为 | 10 个 dtype × 5 种特殊值 | `special_value_positions`、`nan_mask_checked`、`inf_sign_checked`、`neg_zero_checked` |
| 逐调完整性 | 检查 expected/actual call 是否一一对应 | 全部已执行 plan | `expected_calls`、`actual_calls`、`complete`、`task_progress` |
| 串行稳定性 | 重复串行 baseline，确认输出稳定 | 10–20 个 API × 3–5 次 | `serial_repeats`、`serial_mismatch_rate`、`serial_variance` |
| dtype 矩阵 | 统一两后端适用 dtype | 21 种 dtype 抽样 | dtype、API、是否通过、失败类型 |
| 截断边界 | 验证 65,536 前后行为 | 65,535 / 65,536 / 65,537 元素 | `value_truncated`、`eligible`、`inconclusive` |

---

## 5. MO6：候选复现清单、差异/执行失效分布、复现与缩减入口、实际边界、待实测比例

### 5.1 独立运行的候选复现清单：本轮为空

`outputs/vuln_scan_combined/findings.json`：

| 类别 | 数量 |
|---|---:|
| confirmed（contract-gated VIOLATION） | **0** |
| candidates（INCONCLUSIVE + concrete diff） | **0** |
| inconclusive（无 concrete diff） | **0** |
| execution_issues | **36 条 / 34 个 API** |
| warnings | **196 条 / 178 个 API** |

因此，“独立运行的候选复现清单”在本轮**没有条目**。当前最接近“可复现清单”的是 34 个 EXECUTION_ISSUE API，但它们的原因是“没有合格可回放 case”，不是发现了并发缺陷。

#### 34 个 EXECUTION_ISSUE API（按 API 去重）

```text
llm.tf.broadcast_static_shape
llm.tf.cast
llm.tf.clip_by_global_norm
llm.tf.clip_by_value
llm.tf.extract_volume_patches
llm.tf.histogram_fixed_width
llm.tf.image.sample_distorted_bounding_box
llm.tf.math.bessel_i0
llm.tf.math.bessel_i1
llm.tf.nn.conv3d_transpose
llm.tf.nn.ctc_loss
llm.tf.nn.depth_to_space
llm.tf.nn.depthwise_conv2d          (3 条 entry)
llm.tf.nn.embedding_lookup
llm.tf.nn.fractional_avg_pool
llm.tf.nn.fractional_max_pool
llm.tf.nn.max_pool3d
llm.tf.nn.sampled_softmax_loss
llm.tf.nn.space_to_depth
llm.tf.nn.sufficient_statistics
llm.tf.nn.weighted_moments
llm.tf.parallel_stack
llm.tf.quantization.fake_quant_with_min_max_vars
llm.tf.signal.hann_window
llm.tf.signal.inverse_stft
llm.tf.signal.inverse_stft_window_fn
llm.tf.sparse.add
llm.tf.sparse.mask
llm.tf.sparse.reshape
llm.tf.switch_case
llm.tf.tensor_scatter_nd_add
llm.tf.tensor_scatter_nd_sub
llm.tf.unravel_index
llm.tf.vectorized_map
```

统一原因：

> `no eligible replayable case; strict path does not fall back to raw TestOneInput bytes`

即：capture 没有得到可回放的 target call，严格模式不回退到原始 `TestOneInput` 字节，因此没有进入差分比较。它不是“并发差异候选”。

#### 建议的候选复现清单字段（未来有 candidate 时）

| 字段 | 说明 |
|---|---|
| `candidate_id` | 稳定 ID，建议 `short_hash(plan_id + oracle_id + evidence_hash)` |
| `api` / `plan_id` | 目标 API 与计划 |
| `oracle` / `contract_status` | 触发 Oracle 与合同状态 |
| `status` | `CANDIDATE` / `CONFIRMED_VIOLATION` / `INCONCLUSIVE` |
| `diff_summary` | 具体差异摘要（dtype/shape/值位置） |
| `evidence_paths` | `oracle_result`、`report`、`events`、`stdout`、`stderr` |
| `repro_command` | 独立运行命令 |
| `trigger_count` / `total_runs` | 触发次数 / 实际执行次数 |
| `serial_stable` | 串行是否稳定 |
| `reduced_config` / `reduced_input` | 缩减后的配置/输入 |
| `reproduction_rate` | `trigger_count / total_runs` |

### 5.2 差异与执行失效的分布统计

#### 最终合并状态（362 API）

| status | count | 占比 |
|---|---:|---:|
| CLEAN | 131 | 36.2% |
| WARNING | 178 | 49.2% |
| EXECUTION_ISSUE | 34 | 9.4% |
| NOT_APPLICABLE | 19 | 5.2% |
| **合计** | **362** | **100%** |

#### 首轮 → retry 的状态迁移

| 迁移 | 数量 |
|---|---:|
| INCONCLUSIVE → WARNING | 20 |
| INCONCLUSIVE → CLEAN | 11 |

#### 决策分布（362 API）

| decision | count |
|---|---:|
| CONDITIONAL | 345 |
| READY | 4 |
| INSUFFICIENT_INFO | 13 |

#### Oracle 结果

> 下表只列 PASS / INCONCLUSIVE；`NOT_APPLICABLE` 条目（未选中的 Oracle 或场景不适用）未列出，因此每行 PASS 数不等于 result 总数。

| 扫描 | O2 PASS | O3 PASS | O2 INCONCLUSIVE | O3 INCONCLUSIVE | O8 PASS | O8 NOT_APPLICABLE |
|---|---:|---:|---:|---:|---:|---:|
| 初次 `vuln_scan` | 278 | 278 | 31 | 31 | — | — |
| `vuln_scan_retry` | 176 | 176 | 13 | 13 | — | — |
| `vuln_scan_retry2` | 12 | 12 | 1 | 1 | — | — |
| `vuln_scan_retry3` | 1 | 1 | 0 | 0 | — | — |
| `vuln_scan_parallel2` | 26 | 28 | 0 | 0 | — | — |
| `vuln_scan_stress_full` | 44 | 70 | 3 | 6 | 9 | 76 |

#### 执行失效与警告

- **EXECUTION_ISSUE**：
  - 初次扫描：34 条 `pipeline_error`，全部是 `no eligible replayable case; strict path does not fall back to raw TestOneInput bytes`；
  - 合并报告：36 条 / 34 个 API（`tf.nn.depthwise_conv2d` 有 3 条）；
  - stress full：18 条 `pipeline_error` + 2 条 `execution_status`（`SETUP_ERROR`），对应 10 个 API。
- **WARNING**：
  - 合并报告：196 条 `capture_warning` / 178 个 API；
  - case 级原因：`capture_status=target_exception` **534**、`no captured target call` **238**、`unsupported argument kind: unsupported_input` **13**；
  - 初次扫描：178 条 `capture_warning` + 18 条 `run_warning`；case 级原因 464 / 236 / 13。
- **执行完整性**：
  - 742 份 observation：704 complete / 38 incomplete；
  - 任务：2754 submitted、2754 started、2685 completed、67 failed、0 pending、0 partial；
  - task 状态：2685 COMPLETED、67 EXCEPTION、2 NO_TARGET_CALL。

#### 压力扫描单独统计（48 API）

| status | count | 占比 |
|---|---:|---:|
| WARNING | 28 | 58.3% |
| EXECUTION_ISSUE | 10 | 20.8% |
| CLEAN | 7 | 14.6% |
| INCONCLUSIVE | 1 | 2.1% |
| NOT_APPLICABLE | 2 | 4.2% |

### 5.3 TensorFlow 复现及缩减入口：当前没有实现

#### 当前已有的入口

- `ConcurrentRunner.run`：可独立跑 baseline + concurrent；
- `ConcurrentOraclePipeline.run_harness`：可对单个 harness 执行 analyze → generate → capture → validate → run → evaluate；
- `Reporter.package_reproducer`：**只写 `oracle_result.json`**（[HEAD] `reporter.py:99-105`），不包含 harness、plan、cases、observation、events、运行命令；
- `OracleResult.failure_artifacts`：仅在 `VIOLATION` 时收集 `stdout`、`stderr`、`events` 路径（[HEAD] `oracle_engine.py:49-53`）；
- `OracleResult.reproduction_rate`：字段存在，但**始终为 `None`**；
- 报告固定写：
  > `- reproduction_rate: not measured (P3)`

#### 当前不存在的入口

- 没有 `reproducer.py`；
- 没有 `reducer.py`；
- CLI 只有 `analyze`、`generate`、`run`、`pipeline`、`report`，没有 `reproduce` / `reduce`；
- 没有“候选 → 独立复现 → 缩减 → 最小化”的自动化流程。

#### 建议的入口设计（待实现）

```bash
# 候选独立复现
python -m concurrent_oracle.cli reproduce \
  --result outputs/oracle_results/<plan_id>.json \
  --trials 20 \
  --out outputs/reproduce/<candidate_id>/

# 配置缩减 / 输入缩减
python -m concurrent_oracle.cli reduce \
  --candidate outputs/reproduce/<candidate_id>/candidate.json \
  --reduce-threads \
  --reduce-cases \
  --reduce-input \
  --out outputs/reduced/<candidate_id>/
```

建议的复现包内容：

```text
<out>/
  candidate.json
  oracle_result.json
  plan.json
  cases.json
  observation.json
  events.jsonl
  stdout.log
  stderr.log
  concurrent_harness.py
  run_config.json
  README.md
```

### 5.4 配置缩减与输入缩减的实际边界：只有固定预算，没有自动缩减

当前可以确认的固定边界：

| 维度 | 当前值 | 说明 |
|---|---:|---|
| `max_plans`（scan_harnesses 默认） | 1 | 每个 API 只执行第一个适用计划 |
| `max_plans_per_artifact` | 8 | `ScenarioGenerator` 上限 |
| `max_configs_per_scenario` | 4 | `ScenarioGenerator` 上限 |
| `cases_per_api` | 12 | corpus 抽样上限 |
| 默认 case 数 | 8 | `DEFAULT_INPUT_CASES`，每组 64 字节 |
| `trials`（scan.yaml） | 3 | stress 为 5，默认 10 |
| `timeout_seconds`（scan.yaml） | 30 | stress 为 60，默认 60 |
| `no_progress_timeout_seconds` | 10 | stress 为 20 |
| `process_repetitions` | 1 | stress 为 2 |
| 输入截断 | 65,536 元素 | 超过即 `value_truncated`，不可回放 |
| 线程配置 | C03/C04/C08/C09/C01/C10 | 见 3.4 节 |

**这些是预算/上限，不是“缩减后的边界”**。当前没有自动 reducer，因此不能写“配置缩减到 X 线程/输入缩减到 Y 字节仍能触发”这类结论。实际缩减边界需要后续实验按 R12 的口径测量。

### 5.5 复现失败和触发比例：当前可填值与待实测项

| 指标 | 当前值 | 状态 |
|---|---:|---|
| confirmed violation | 0 / 362（0%） | 已实测 |
| candidate（concrete diff） | 0 / 362（0%） | 已实测 |
| inconclusive（无 concrete diff） | 0 / 362（0%） | 已实测（首轮 31，retry 后 0） |
| EXECUTION_ISSUE | 34 / 362（9.4%） | 已实测 |
| WARNING | 178 / 362（49.2%） | 已实测 |
| CLEAN | 131 / 362（36.2%） | 已实测 |
| NOT_APPLICABLE | 19 / 362（5.2%） | 已实测 |
| 独立复现尝试次数 | 0 | 未执行（无 candidate） |
| 复现失败率 | 不适用 | **待实测**（有 candidate 后） |
| 触发率（trigger / total runs） | 当前范围内 0 | **待实测**（扩展到 O4–O9 后重算） |
| stress 扫描 candidate | 0 / 48（0%） | 已实测 |
| stress 扫描 EXECUTION_ISSUE | 10 / 48（20.8%） | 已实测 |

> 建议文章写法：
>
> “本轮 362 个 API 的探索性扫描未产生 contract-gated violation 或 concrete-diff candidate；34 个 API 因没有合格可回放 case 进入 EXECUTION_ISSUE，178 个 API 带 capture warning。由于候选数为 0，复现失败率与触发率在本轮没有可计算的分母，标记为待实测；0 candidate 不代表无缺陷。”

---

## 6. 完成度核对表

> **注意**：下表“当前证据/状态”描述的是 **2026-10-30 旧扫描产物**，不是 2026-11-05 的代码状态。O1/O4/O5/O6/O7/O9 的实现状态请以 [ORACLE_IMPLEMENTATION_O1_O4_O5_O6_O7_O9.md](ORACLE_IMPLEMENTATION_O1_O4_O5_O6_O7_O9.md) 为准；旧扫描没有重跑，因此下表仍保留历史状态，避免把“代码已实现”误写成“已有实验数据”。

| 编号 | 补充要求 | 当前证据 | 状态 | 建议 |
|---|---|---|---|---|
| MO2-1 | 实际启用的 LLM 适用性 prompt | 无；`llm_suitability.py` 为 P1 占位，696 份报告全 null | **未实现/未启用** | 文章改为规则法；或实现 P1 后重跑 |
| MO2-2 | 驱动/转换 prompt | 无；`transformer.py` 固定模板，`use_llm_transform=false` | **未实现/未启用** | 同上 |
| MO2-3 | 模型版本 | 无 LLM 记录 | **无数据** | 启用后记录 `model_version` |
| MO2-4 | 采样参数 | config 无 temperature/top_p/max_tokens | **无数据** | 增加 llm config 段 |
| MO2-5 | 输出提取 | 无 LLM 输出 | **无数据** | 实现 JSON 解析 + 有限重试 |
| MO2-6 | 回退记录 | `llm_error` + rules retained 骨架存在，未触发 | **骨架/未触发** | 启用后记录 `fallback_used` |
| MO2-7 | 第三章生成模型设置 | `fuzz_api_one` 为 gpt-5.4，见 2.6 | **模块外可填** | 与并发预言机分开写 |
| M03-1 | PSR 状态 ↔ 底层调用映射 | 只有 `psr_path`；362/362 有路径，metadata 空 | **部分实现** | 增加 `psr_state_id`/`psr_param_map` |
| M03-2 | 已有 fuzz 追溯接入 | stress 101 文件/1212 case 用 corpus；主扫描全 defaults | **部分实现** | 文章限定为 stress 48 API |
| M03-3 | 不同配置耗时 | 只有每 API `elapsed_seconds`；无逐 plan/config 耗时 | **部分实现** | 增加 per-plan `duration_seconds` |
| M03-4 | 自动命名和别名 | artifact/plan/run/case 命名明确；TF import 别名解析已实现 | **已实现（命名）** | 增加 PSR/参数别名传递 |
| M03-5 | TF 资源/Dataset/callback | 合同存在，后端未实现；19 个 NOT_APPLICABLE | **未实现** | 不计入完成度 |
| MO4-1 | 大数量全量/分块数值复核 | 完整值逐元素；>65,536 仅哈希 | **部分实现** | 增加分块容差比较 |
| MO4-2 | 特殊值位置检查 | NaN/Inf 按位置；`-0.0` 不单独区分；截断值无位置检查 | **部分实现** | 增加位置记录与 `-0.0` 检查 |
| MO4-3 | 逐调调用完整性 | 已实现并量化：2754/2685/67 | **已实现/已实测** | 直接引用 |
| MO4-4 | 串行稳定性记录 | baseline 最多 3 次资格检查；无重复串行稳定性统计 | **部分实现** | 增加 3–5 次串行重复 |
| MO4-5 | 两后端适用 API/dtype | 仅 TF CPU；21 种 dtype 实测；无 dtype 合同 | **部分实现** | 文章限定 TF；PyTorch 另测 |
| MO4-6 | 采样/转换/截断范围 | 8 默认/12 corpus/65,536 截断 | **已实现** | 直接引用 |
| MO6-1 | 独立候选复现清单 | 0 confirmed / 0 candidate；34 个 EXECUTION_ISSUE | **空/部分** | 引用 34 API 清单 |
| MO6-2 | 差异/执行失效分布 | 131/178/34/19；36 issue/196 warning | **已实测** | 直接引用 |
| MO6-3 | TF 复现/缩减入口 | 无 reproducer/reducer；`package_reproducer` 只写 JSON | **未实现** | 实现 CLI + 复现包 |
| MO6-4 | 配置/输入缩减边界 | 固定预算：max_plans=1、plans=8、configs=4、cases=12、截断=65536 | **仅固定边界** | 自动缩减待实测 |
| MO6-5 | 复现失败/触发比例 | 0 candidate，复现率不适用；触发率 0（当前范围） | **待实测** | 有 candidate 后按 R12 统计 |

---

## 附录 A：证据文件索引

| 编号 | 文件/目录 | 用途 |
|---|---|---|
| E1 | `outputs/vuln_scan_combined/FINAL_VULN_REPORT.md` | 最终合并状态、执行失败、警告、范围限制 |
| E2 | `outputs/vuln_scan_combined/per_api_status.csv` | 362 API 的 status/decision/plans/confirmed/candidates |
| E3 | `outputs/vuln_scan_combined/findings.json` | confirmed/candidates/inconclusive/execution_issues/warnings |
| E4 | `outputs/vuln_scan/scan_results.jsonl` | 初次扫描 362 API 的 summary/findings/elapsed |
| E5 | `outputs/vuln_scan_stress_full/scan_results.jsonl` | 48 API 压力扫描的 summary/findings/elapsed |
| E6 | `outputs/**/captured_cases/*/cases.json` | case 来源、target_calls、eligible、截断字段 |
| E7 | `outputs/**/executions/**/combined_observation.json` | task_progress、baseline_qualification、process_repetitions |
| E8 | `outputs/**/suitability/*_suitability.json` | 696 份 suitability；LLM 字段全 null |
| E9 | `outputs/**/oracle_results/*.json` | O2/O3/O8 结果、reproduction_rate |
| E10 | `outputs/**/reports/*_report.md` | 单计划报告；`reproduction_rate: not measured (P3)` |
| E11 | `configs/scan.yaml`、`configs/scan_stress.yaml` | 实际运行配置；`use_llm_transform=false` |
| E12 | `contracts/tensorflow/*.json` | CPU op/callback/tf.data/resource/threading 合同 |
| E13 | `__pycache__/*.cpython-310.pyc` | 运行版本行为（corpus、process_repetitions、O8、C08/C09） |
| E14 | git HEAD `475ef33` | 稳定接口/数据模型/默认值 |

## 附录 B：关键代码位置

### B.1 git HEAD（稳定接口）

| 主题 | 文件/行 |
|---|---|
| Pipeline 传 `llm_client=None` | `concurrent_oracle/pipeline.py:61-66` |
| LLM 适用性占位 | `concurrent_oracle/llm_suitability.py:1-41` |
| LLM 结果校验/回退 | `concurrent_oracle/api_suitability_analyzer.py:40-48, 146-165` |
| 固定模板转换 | `concurrent_oracle/transformer.py:1, 68-90` |
| 配置字段 | `concurrent_oracle/config.py:41-45` |
| PSR 路径加载 | `concurrent_oracle/harness_loader.py:81-89` |
| 别名解析 | `concurrent_oracle/harness_analyzer.py:77-120` |
| 命名规则 | `concurrent_oracle/schemas.py:354-373` |
| 数值比较器 | `concurrent_oracle/result_comparator.py:13-167` |
| baseline 资格检查 | `concurrent_oracle/runner.py:284-331` |
| 场景能力 | `concurrent_oracle/constants.py:80-142` |
| Oracle 注册 | `concurrent_oracle/oracle_engine.py:20-26, 68-75` |
| 报告/复现包 | `concurrent_oracle/reporter.py:95-105` |
| CLI 命令 | `concurrent_oracle/cli.py:149-194` |

### B.2 运行版本（pyc）

| 主题 | pyc |
|---|---|
| corpus 加载 | `__pycache__/case_capture.cpython-310.pyc`（`corpus_cases`） |
| corpus 转换 | `__pycache__/transformer.cpython-310.pyc`（`transform(..., cases=...)`） |
| process_repetitions | `__pycache__/runner.cpython-310.pyc` |
| baseline_qualification | `__pycache__/runner.cpython-310.pyc` |
| O8 | `__pycache__/oracles/threadpool.cpython-310.pyc`、`oracle_engine` |
| C08/C09/C10 | `__pycache__/constants.cpython-310.pyc` |
| scan_harnesses | `__pycache__/scan_harnesses.cpython-310.pyc` |
| combine_scans | `__pycache__/combine_scans.cpython-310.pyc` |

## 附录 C：复现/验证命令（恢复源码后）

> 当前工作区源码被删除，以下命令需要先恢复运行版本源码；`git checkout -- concurrent_oracle` 只能恢复 HEAD 版本，不含 `scan_harnesses`/corpus/O8/C08/C09。命令仅作为实现/恢复后的参考。
>
> **不要直接执行 `git checkout -- concurrent_oracle`**：它会同时覆盖当前工作区对 `fix_guidance.md` 的本地修改。更安全的方式是先备份 `fix_guidance.md`，或只恢复代码文件（见下方 `git restore` 示例）。

```bash
cd /root

# 1) 恢复 HEAD 代码（仅代码；不要覆盖 docs；见上方说明）
git restore --source=HEAD -- concurrent_oracle/pipeline.py concurrent_oracle/runner.py concurrent_oracle/transformer.py concurrent_oracle/case_capture.py concurrent_oracle/schemas.py concurrent_oracle/config.py concurrent_oracle/llm_suitability.py concurrent_oracle/api_suitability_analyzer.py concurrent_oracle/oracle_engine.py concurrent_oracle/reporter.py concurrent_oracle/harness_loader.py concurrent_oracle/harness_analyzer.py concurrent_oracle/result_comparator.py concurrent_oracle/scenario_generator.py concurrent_oracle/suitability_rules.py concurrent_oracle/constants.py concurrent_oracle/contract_registry.py concurrent_oracle/harness_validator.py concurrent_oracle/process_manager.py concurrent_oracle/observation.py concurrent_oracle/cli.py concurrent_oracle/oracles concurrent_oracle/adapters concurrent_oracle/templates

# 2) 单 harness 分析（HEAD CLI）
python -m concurrent_oracle.cli analyze \
  --harness /root/fuzz_output_experiment/llm.tf.math.log1p.py \
  --output-dir /root/concurrent_oracle/outputs/rerun_one

# 3) 单 harness 端到端（HEAD CLI；严格合同模式默认拒绝 CONDITIONAL，需加 --allow-unknown-contract）
python -m concurrent_oracle.cli pipeline \
  --harness /root/fuzz_output_experiment/llm.tf.math.log1p.py \
  --max-plans 1 \
  --allow-unknown-contract \
  --output-dir /root/concurrent_oracle/outputs/rerun_one

# 4) 批量扫描（仅运行版本有 scan_harnesses.py；需先恢复该文件）
python -m concurrent_oracle.scan_harnesses \
  --harness-dir /root/fuzz_output_experiment \
  --config configs/scan.yaml \
  --output-dir outputs/vuln_scan_rerun \
  --max-plans 1

# 5) 合并 retry
python -m concurrent_oracle.combine_scans \
  --initial outputs/vuln_scan_rerun \
  --retry outputs/vuln_scan_rerun_retry \
  --out outputs/vuln_scan_rerun_combined

# 6) 压力扫描（仅运行版本）
python -m concurrent_oracle.scan_harnesses \
  --harness-list outputs/stress_full_harnesses.txt \
  --config configs/scan_stress.yaml \
  --output-dir outputs/vuln_scan_stress_rerun \
  --max-plans 1
```

## 附录 D：可直接引用的“事实句”

1. **LLM 未启用**：`concurrent_oracle` 的 pipeline 在构造 `APISuitabilityAnalyzer` 时传入 `llm_client=None` 和 `model_config={}`；696 份 suitability 报告的 `llm_result`、`model_config`、`prompt_version`、`raw_llm_response` 全部为空。
2. **转换是模板**：`HarnessTransformer` 是固定模板转换；两个实际配置都设置 `use_llm_transform: false`。
3. **PSR 只有路径**：362/362 个初始 artifact 有 `psr_path`，但 `metadata` 全为空，case/plan 中没有 PSR 状态字段。
4. **语料只在压力扫描**：初次 362-API 扫描的 343 个 capture 文件全部是 `deterministic_defaults`；stress 48-API 扫描的 101 个 capture 文件全部是 `argfuzz_corpus`。
5. **完整性已量化**：742 份 observation 中 704 份 complete；2754 个任务中 2685 个 completed、67 个 exception、2 个 no target call。
6. **串行 baseline 资格**：183 份一次通过、24 份两次通过、56 个 case 因 `baseline_not_complete` 被排除；没有第三次尝试。
7. **截断边界**：模板 `_describe`/`_describe_input` 在 65,536 个元素处截断；截断参数不可回放，`_check_arg_desc` 标记 ineligible。
8. **数值比较**：完整值逐元素比较，NaN mask/Inf 符号按位置；超过 65,536 元素仅比较哈希，缺失值返回 `values truncated; hash differs`（inconclusive）。
9. **候选为 0**：最终合并报告 confirmed=0、candidates=0、inconclusive=0；34 个 API 进入 EXECUTION_ISSUE，统一原因是“no eligible replayable case”。
10. **复现/缩减未实现**：`Reporter.package_reproducer` 只写 `oracle_result.json`；`OracleResult.reproduction_rate` 始终为 null；没有 `reproducer.py` / `reducer.py` / CLI `reproduce`/`reduce`。
11. **Dataset/callback/resource 未实现**：`SCENARIO_CAPABILITIES` 中 `shared_resource_access`、`tfdata_*`、`python_callback_lifecycle`、`exception_recovery`、`invalid_input_cpu` 全部 `implemented=False`；19 个 API 被标记 NOT_APPLICABLE。
12. **TF CPU only**：运行记录为 `tensorflow_version=2.21.0-dev0+selfbuilt`、`gpu_devices=[]`、`device=CPU`；本目录没有 PyTorch runtime。

---

### 备注：本文没有做的事

- 没有把 `guidance.md` 中的建议字段/建议参数写成“已实现”。
- 没有把 `fix_guidance.md` 的 R01–R12 计划写成“已完成”。
- 没有把 0 candidate 写成“无缺陷”。
- 没有把 stress 扫描的 corpus 覆盖外推到主扫描 362 API。
- 没有把“固定预算”写成“缩减边界”。

如果后续需要，我可以基于这份汇总继续产出：
1. MO2 的 P1 LLM 实现骨架（prompt/parse/fallback + 记录字段）；
2. MO6 的 `reproducer.py` / `reducer.py` 与 CLI 草案；
3. MO4 的分块数值比较与串行稳定性实验脚本；
4. 一份只包含“可直接粘贴进论文”的精简版（不含工程细节）。

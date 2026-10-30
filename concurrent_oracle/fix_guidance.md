# ArgSpaceFuzz-TF concurrent_oracle 优化修改清单

审查日期：2026年10月30日  
目标仓库：<https://github.com/yyds1233/ArgSpaceFuzz-TF>  
目标模块：`concurrent_oracle/`  
审查提交：`54d75ffb860a601d67e14c32f07f79b05898198a`

代码快照：<https://github.com/yyds1233/ArgSpaceFuzz-TF/tree/54d75ffb860a601d67e14c32f07f79b05898198a/concurrent_oracle>

本文根据已完成的源码审查和小型逻辑验证整理，供项目开发 agent 按项修改。文中“已确认”指能够从代码与不依赖TensorFlow的验证中确认的行为；TensorFlow真实算子执行、并发触发概率和性能收益仍需在项目容器中验证。本文不表示已经修改项目源码。

## 1. 修改目标与总体顺序

保留现有“加载—分析—计划生成—驱动转换—隔离执行—预言机—报告”的模块结构。先修复会产生误报、假通过和配置失效的问题，再扩展共享资源、Dataset和callback等测试场景。

推荐顺序：

1. 修复O3任务串扰误报、标量序列化和比较器错误。
2. 固定目标API的真实输入，拆分独立基准与测试进程。
3. 打通trials、运行配置、契约和计划预算。
4. 落实独立输入、共享只读、共享资源三类执行模型。
5. 接入进展监控、独立运行目录、故障采集和复现记录。
6. 扩展TensorFlow专属Dataset及callback检查。

### 优先级总表

| 编号  | 优先级 | 修改项 | 主要影响 | 修改完成标志 |
| --- | --- | --- | --- | --- |
| R01 | P0  | O3串扰判定 | 正常相同输出被误报；绕过契约 | 先检查自身基准，疑似串扰经过契约门控 |
| R02 | P0  | 标量序列化与描述失败处理 | 标量无法采集；采集失败可能被当成通过 | 标量正常序列化，describe_error不可判为相等 |
| R03 | P0  | 固定真实调用输入 | 相同字节不保证相同张量 | 独立进程中目标调用前参数摘要一致 |
| R04 | P0  | 基准进程与比较关系 | 预热掩盖问题；内部并行缺少有效对照 | baseline与test独立启动，并区分比较目的 |
| R05 | P0  | trials、配置生效和计划截断 | 实际调用强度不足；重要场景未执行 | 配置影响实际执行，调用数和计划覆盖可核对 |
| R06 | P0  | 场景/预言机契约绑定 | UNKNOWN被抬升为SUPPORTED | 计划分别绑定契约，未知状态不被其他场景覆盖 |
| R07 | P1  | 场景执行后端及能力清单 | 场景名称与实际行为不一致 | 能核对实际共享关系和已实现检查 |
| R08 | P1  | 别名、图执行与回调观测 | 漏识别目标，观测对象与真实执行不一致 | 别名可识别，不支持的观测模式明确报告 |
| R09 | P0  | 递归与dtype相关比较 | 嵌套输出误报或无法判定 | 嵌套张量统一比较，不把哈希差异直接当数值违规 |
| R10 | P1  | Dataset消费与生命周期 | 配置未作用于数据流水线 | 有真实消费、元素身份及恢复/停止观测 |
| R11 | P0/P1 | 运行隔离、进展与故障产物 | 可能读旧结果；超时无法定位 | 唯一run_id、原子写入、阶段事件及故障记录 |
| R12 | P0/P1 | 可移植测试与回归集 | 新环境无法复现工程验证 | 单元测试不依赖/root路径，集成测试单独运行 |

P0应在正式批量实验前完成。P1用于增加场景覆盖和复核能力。R11中的旧结果隔离、R12中的关键回归测试应与P0修复同步完成。

## 2. 已确认的问题与验证结果

| 验证  | 当前结果 | 应有行为 |
| --- | --- | --- |
| 两个输入合法地产生相同输出 | O2为PASS，O3为VIOLATION；UNKNOWN契约也会走串扰违规分支 | 与各自基准一致时不报告串扰 |
| `_jsonable(np.array(1.0))` | `TypeError: 'float' object is not iterable` | 返回可序列化标量 |
| 比较两个相同describe_error | `equal=True, inconclusive=False` | 观测失败，返回INCONCLUSIVE |
| 嵌套张量数值为1.0与1.0000001，哈希不同 | 在默认容差内仍因sha256差异判为不一致 | 张量节点递归使用数值比较 |
| 模板传入trials=7，两个case、两个外部线程 | 两次基准调用加两次工作任务，共四次调用 | trials作用于实际执行循环，调用次数可核对 |
| 一个场景SUPPORTED，另一场景UNKNOWN | 两类计划均复制为SUPPORTED | 计划按各自场景及预言机绑定契约 |
| 外部线程配置仅为[1,2] | 仍生成C04的4线程计划 | 显式配置过滤或生成实际运行矩阵 |
| 独立调用、共享只读、内部并行三个场景依次生成 | 默认四计划额度被前两类用完 | 至少先覆盖每种适用场景，再扩展配置 |
| 新检出仓库，默认契约目录缺少JSON | 简单目标得到CONDITIONAL，pipeline直接跳过 | 提供基础契约fixture，明确严格/探索策略 |

上述模板调用次数验证使用替代的目标调用记录函数，证明执行循环没有使用trials；不代表已进行真实TensorFlow并发测试。

## 3. 逐项修改要求

### R01 修复O3任务串扰误报

**涉及文件**

- `oracles/independence.py`
- `result_comparator.py::find_cross_talk`
- `tests/test_oracle_engine.py`
- `tests/test_result_comparator.py`

**当前问题**

`IndependenceOracle`先查找其他输入的基准是否匹配当前输出，再比较自身基准。只要匹配其他基准，就直接返回VIOLATION。正常API常有非单射关系，例如不同负数经ReLU均输出零，不同输入经reduce_sum得到相同和，因此这个条件不能单独证明串扰。

`find_cross_talk`还会搜索全部基准，不仅是本轮实际选择的其他任务。串扰分支没有检查契约状态。

**修改方案**

1. 先将任务输出与自己的有效基准比较。
2. 自身基准匹配且观测完整时，该任务的输出关联检查通过。
3. 自身不匹配且比较结果明确时，再检查其他本轮任务的基准。
4. 排除与自身基准等价、无法区分的基准组。
5. 匹配其他基准时标记为疑似结果交换/污染证据，最终经过对应契约和完整性检查决定状态。
6. 比较结果为INCONCLUSIVE时，不进入匹配成功分支。

**建议判定结构**

```python
own = compare_to_own_baseline(task)
if own.inconclusive:
    return INCONCLUSIVE
if own.equal:
    return PASS

others = find_distinguishable_selected_baselines(task)
if matches_other_baseline(task, others):
    return contract_gated_candidate("suspected_cross_talk")
return contract_gated_candidate("output_mismatch")
```

这段是修改建议，不是现有接口。

**验收标准**

- 两个任务均输出零，且各自基准也为零：不报串扰。
- 两个任务的可区分输出被交换：能够检出关联异常。
- UNKNOWN契约下的疑似串扰：保留证据，不直接升级为VIOLATION。
- 缺失输出、describe_error和无法判定比较不参与匹配。

### R02 修复标量序列化及描述失败处理

**涉及文件**

- `templates/concurrent_harness.py::_jsonable`
- `templates/concurrent_harness.py::_safe_describe`
- `result_comparator.py::compare_described`

**当前问题**

零维NumPy数组的`tolist()`返回普通标量，当前代码对它继续迭代，导致标量张量结果转成describe_error。比较器随后可能将两个相同的describe_error视为正常相等。

**修改方案**

```python
if isinstance(value, np.ndarray):
    return _jsonable(value.tolist())
```

同时统一描述状态：正常结果、描述失败、输出类型不支持。出现描述失败时保留对象类型、异常类型与原因，并将相关比较标记为INCONCLUSIVE。

不允许描述失败增加“可比较任务数”，不允许两个描述错误相同就产生PASS。

**验收标准**

- 标量、零维数组、空张量、多维张量、布尔值、复数和字符串均有对应测试。
- 标量归约接口在TF集成测试中能够采集有效输出。
- 注入输出描述失败后，O2/O3不会产生假通过或串扰误报。

### R03 从固定字节升级为固定目标调用实例

**涉及文件**

- `case_capture.py`
- `transformer.py`
- `templates/concurrent_harness.py::_invoke`
- `schemas.py`

**当前问题**

当前CaseCapture保存的是固定字节，执行阶段仍重新运行原始TestOneInput。原始驱动若使用TensorFlow、NumPy或Python随机源，相同字节也可能构造出不同输入。串并行输出差异可能来自输入变化。

**修改方案**

将执行组织为三个边界：

```python
case = build_case(seed)
result = invoke_target(case)
observation = describe_result(result)
```

优先通过目标调用包装器捕获调用前的args/kwargs，并保存真实数据及参数构造描述。至少区分以下类型：

| 类型  | 建议保存内容 |
| --- | --- |
| 普通Tensor | dtype、shape、实际数值、内容摘要 |
| SparseTensor | indices、values、dense_shape |
| RaggedTensor | flat_values、全部nested_row_splits |
| Variable | 初始值、dtype、shape及资源构造描述 |
| Dataset、迭代器、生成器 | 专用构造配方、消费约束与来源，不能通用深拷贝 |

记录seed、来源harness哈希、捕获位置、实际参数摘要和实例来源。捕获到调用边界后，再通过基准验证可比较性；不能把“捕获成功”自动当作“输入合法”。

第一版优先冻结普通数据或使用显式种子的stateless构造。不要在多个工作线程中反复修改全局随机种子。

**验收标准**

- 同一case在两个独立新进程中，目标调用前的参数摘要一致。
- 原始驱动包含随机张量构造时，仍能够重放同一实例。
- 捕获实例和专用合成实例分别标记。
- 不支持的资源类型产生明确转换失败原因，不自动退化为任意字符串描述。

### R04 拆分基准进程并明确比较关系

**涉及文件**

- `templates/concurrent_harness.py::run_case`
- `harness_validator.py::run_serial_baseline`
- `runner.py`
- `pipeline.py`

**当前问题**

正式执行在同一进程先运行串行基准，再执行多线程任务，提前改变初始化、缓存和全局资源状态。内部并行场景的基准也沿用当前内部线程配置，不能评价从单内部线程到多内部线程的变化。

验证器没有设置模板已有的capture_only，因此所谓串行基准流程还会继续进入一个工作线程执行。验证器生成的独立基准也没有成为正式比较时统一使用的参考结果。

**修改方案**

- 显式提供`mode=baseline`和`mode=concurrent`，分别启动新子进程。
- 基准结果作为独立产物，由父进程传入比较阶段。
- 为两类目的分别建立参考：外部线程干扰比较保持内部线程配置相同；内部并行比较使用内部线程1与T的对照。
- 共享更新必须对齐总操作数，避免将N×R与R的最终状态直接比较。
- 首次目标调用与预热后调用分别组织和记录；新进程不自动等于目标内核绝对冷启动。
- 基准只为实际有效、适用的case提供比较资格。记录输入拒绝、未到达目标、基准不稳定和有效输出等类别。
- 对需要数值比较的API增加串行重复检查，建立稳定参考。

**验收标准**

- baseline与test的PID不同。
- baseline模式不创建并发工作池，也不运行额外工作任务。
- 内部并行场景的参考组与测试组实际线程配置可以核对。
- 某个case基准失败，不因另一个case成功就被认定为可比较。

### R05 让trials、运行配置与计划预算真正生效

**涉及文件**

- `config.py`
- `scenario_generator.py`
- `runner.py`
- `templates/concurrent_harness.py`
- `cli.py`

**当前问题**

plan.trials没有进入实际执行循环；运行矩阵主要由固定常量生成，未按YAML线程列表过滤；deterministic字段未作用于TF设置。计划截断按照候选顺序进行，普通算子的内部并行场景可能被前两类场景耗尽额度。CLI默认只运行第一个计划。

**修改方案**

1. 分别定义每线程调用次数、独立进程重复次数和每API配置预算。
2. 将trials显式传递至模板并使用；每轮保存trial_id和目标调用记录。
3. 调用数核对使用计划中的预期值，而非只检查收到N个任务结果。
4. 从配置生成运行组合，或将固定C00—C07与显式配置做一致性校验和过滤。
5. 保存requested_runtime与actual_runtime，记录实际CPU可见性及TF线程设置。
6. deterministic等设置必须实际应用；未支持的配置项直接报告，不能只写进JSON。
7. 先为各适用场景分配一个配置，再扩展线程规模；截断时保存未选计划及原因。
8. max_plans属于执行预算，默认值和使用说明需明确实际覆盖范围。

**验收标准**

- 修改trials能够改变实际完成的调用数。
- 配置只允许外部线程1、2时，不生成或执行4线程计划。
- 候选包含独立调用、共享只读和内部并行时，最小场景覆盖策略能够保留三类。
- 重复task_id、缺失trial_id、缺失调用记录均可被完整性检查发现。

### R06 按场景和预言机绑定契约

**涉及文件**

- `contract_registry.py`
- `api_suitability_analyzer.py`
- `scenario_generator.py`
- `schemas.py`
- `pipeline.py`

**当前问题**

aggregate_status只要发现一个SUPPORTED，就返回API级SUPPORTED；计划随后直接复制该聚合值。未知场景因此获得不应有的判定权限。公开提交缺少默认契约JSON，而pipeline只运行READY，导致新检出环境容易全部跳过。

**修改方案**

- API级状态只用于汇总展示。
- 每个计划按API、场景、预言机重新查询并保存契约。
- 契约记录包含来源、支持条件、框架版本及验证记录。
- UNKNOWN不能因其他场景受支持而被升级；UNSUPPORTED不作为受支持语义的违规依据。
- 提供严格模式与探索模式：严格模式仅运行具有依据的计划；探索模式允许未知场景采集证据，差异保留为待复核。
- 将少量可公开、具有依据的契约和对应fixture纳入仓库，保证新检出后能够执行基础闭环。
- 加载契约失败、目录缺失和格式错误要出现在报告中，不能仅表现为零候选。

**验收标准**

- 一个场景SUPPORTED、另一个UNKNOWN时，生成计划保留各自状态。
- 同一场景的不同预言机可以具有不同支持条件。
- 未知契约下的差异不会从O3等分支绕过门控。
- 新检出仓库可使用随仓库提供的fixture跑通基础流程。

### R07 实现场景后端与能力清单

**涉及文件**

- `transformer.py`
- `templates/concurrent_harness.py`
- `scenario_generator.py`
- `oracle_engine.py`
- `constants.py`

**当前问题**

state_model存在于计划中，但模板仍主要执行原始TestOneInput，没有据此通用地构造共享输入或共享资源。当前真正注册的预言机只有O2、O3；其余编号显式显示未实现，但默认配置启用O1—O9，容易混淆计划能力与完成能力。

**修改方案**

先实现三个执行后端：

| 后端  | 实际构造要求 | 需要的观测 |
| --- | --- | --- |
| 独立调用 | 每任务独立输入及资源实例 | 输入摘要、输出、异常、任务身份 |
| 共享只读 | 各任务绑定同一输入对象 | 共享身份、执行前后输入摘要 |
| 共享资源 | 显式指定共享Variable等对象 | 初始/最终状态、操作数、契约关系 |

不要将普通深拷贝或重新运行TestOneInput等同于共享模型已经落实。不同进程之间不比较Python对象地址；同一进程内可记录资源身份，结合计划关系核对实际共享。

增加能力清单，分别描述场景构造、观测采集和预言机是否可用。生成计划前验证三者是否组成可执行组合。默认配置优先启用已经完成的组合；暂未实现项继续明确保留状态。

后续建议先补O4输入保持、O5异常隔离与恢复、O8进展性，再扩展O6/O7。原生故障产物应能够独立归档，不必等待所有预言机实现后才采集。

**验收标准**

- 独立场景实际使用不同对象，共享场景实际绑定同一对象。
- O4能够发现注入的输入修改。
- O5能够执行非法任务、合法任务和后续恢复任务，并分别记录。
- 报告区分已完成检查、未实现检查和不适用检查。

### R08 完善别名、图执行与回调观测

**涉及文件**

- `harness_analyzer.py`
- `harness_loader.py`
- `templates/concurrent_harness.py::_patch_targets`
- `transformer.py`

**当前问题**

导入别名没有还原，`from tensorflow.math import log1p as op`形式无法识别目标。模板先导入原始harness再包装tf属性，也不能覆盖原始模块已经保存的函数引用。

线程局部_PATCH_STATE只在设置过记录的线程上有效；内部callback线程可能无法关联记录。tf.function的Python包装触发也不能直接等同于每次底层图执行。

**修改方案**

- 建立Import、ImportFrom及常见模块级绑定的符号映射。
- 在导入原始harness前安装必要包装，核对已有函数别名引用。
- 记录目标进入、返回、异常、调用位置和call_index。
- 区分Eager、图执行、callback三种观测模式。
- 图执行以可重放函数/图调用的输入输出为观测边界，不把trace期间的Python调用计作每轮内核执行。
- callback使用显式任务关联与事件记录，不能假设调用线程的TLS自然传播。
- 静态只匹配叶名称的结果保留歧义信息，避免误把不同命名空间的同名API视为完全等价。

**验收标准**

- import别名、from别名和常见模块级函数绑定有测试。
- 未命中包装时显示明确的NO_TARGET_CALL或不支持原因。
- tf.function和callback集成测试能够说明采集边界，不虚增目标调用次数。

### R09 统一递归和dtype相关比较规则

**涉及文件**

- `result_comparator.py`
- `adapters/tensorflow.py`
- `templates/concurrent_harness.py::_describe`

**当前问题**

顶层Tensor使用专用逻辑，但list、tuple、dict中的张量描述直接作为普通字典比较，sha256成为必须相等的字段，导致嵌套浮点输出无法正确使用容差。模板、adapter和比较器中还存在多套数值逻辑，后续容易不一致。

**修改方案**

- 递归遇到张量描述时继续使用统一的张量比较入口。
- 哈希用于精确相等的快速判断；浮点哈希不同后继续数值比较。
- 为大张量保存可重放结果文件，按块比较完整数据，避免超过4096元素就长期无法判定。
- 整数、布尔、索引与shape使用精确规则；浮点/复数使用声明的容差与NaN/Inf规则。
- SparseTensor分别比较dtype、dense_shape、indices与values。若要规范化索引顺序，需先明确合法语义，不能随意改变表示。
- RaggedTensor保存全部nested_row_splits，而不仅是最外层row_splits。
- 字符串/object数组按内容及长度编码计算摘要，不依赖object数组中的指针字节。
- describe_error、未知对象类型与截断数据保留明确的无法判定状态。
- 比较结果显式传播equal、inconclusive和原因，不能只检查equal。

**验收标准**

- 顶层和嵌套张量在相同容差下给出一致结论。
- 微小合法浮点差异通过，超出容差可检出。
- 大整数张量、字符串、复数、SparseTensor和多层RaggedTensor有对应测试。
- adapter与oracle调用同一比较核心或通过共同测试集约束一致性。

### R10 落实Dataset真实消费与生命周期检查

**涉及文件**

- `scenario_generator.py`
- `transformer.py`
- `templates/`下的Dataset专用模板（建议新增）
- O6/O7对应预言机实现（建议新增）

**当前问题**

当前已识别并行map、interleave和提前终止等场景，但tf_data_parallel_calls主要存在于配置/运行记录中，没有通用地作用于Dataset构造。仅构造Dataset对象，也不足以评价实际数据消费行为。

**修改方案**

使用专用构造和消费后端，将num_parallel_calls、deterministic及相关选项真正施加到流水线。至少记录element_id、已消费数量、缺失/重复元素、内容、允许顺序和最后进展时间。

| 场景  | 判定重点 |
| --- | --- |
| 确定性顺序 | 输出序列和内容符合参考关系 |
| 允许乱序 | 元素多重集合及内容正确，不以排序变化直接判错 |
| 提前停止 | 按消费上限停止，保留取消与恢复证据 |
| 异常传播 | 预期异常可见，合法任务及后续恢复符合要求 |
| callback生命周期 | 记录进入、退出和结束后活动，结合契约判定 |

TensorFlow允许deterministic=False的并行map乱序；包含有状态操作时，输出顺序确定不等于状态访问顺序确定。op determinism还可能改变性能及部分Dataset并行行为。参考组与触发组应分别声明确定性策略，不能为消除误报而无条件关闭实际希望测试的并发行为。

**验收标准**

- 变更num_parallel_calls能够改变实际构造参数，并有运行记录。
- 允许乱序的合法结果不报告数据损坏。
- 注入缺失、重复或错误内容时能够检出。
- 提前终止和异常后确实执行恢复任务，而不只记录异常数量。

### R11 补齐独立运行、进展监控和故障采集

**涉及文件**

- `watchdog.py`
- `process_manager.py`
- `runner.py`
- `harness_validator.py`
- `observation.py`
- `config.py`

**当前问题**

Watchdog未接入主执行流程，观测主要在结束时写出。超时后缺少阶段证据。同一计划/配置复用固定目录，旧observation.json没有在执行前清理或校验本轮身份。Sanitizer配置与硬编码预加载路径未形成一致控制。

**修改方案**

1. 每次独立运行使用唯一run_id；baseline、test和复现记录分别保存。
2. 结果包含run_id、计划/输入/配置哈希、源码版本及实际运行环境。
3. 使用临时文件加原子替换写结果，读取时校验本轮身份。
4. 保存阶段事件：FRAMEWORK_READY、CASE_READY、TARGET_ENTER、TARGET_RETURN/TARGET_EXCEPTION、TASK_FINISHED。
5. 持续时间使用monotonic时钟；心跳与阶段进展需要由父进程实际读取。
6. 超时前尽可能导出线程堆栈，再结束任务。无进展只是候选证据，结合正常耗时和资源压力复核。
7. 核对退出码、信号和最终观测的一致性，不能让旧文件或失败退出满足验证通过条件。
8. Sanitizer启用方式由配置与显式环境决定，验证器和runner使用一致环境管理。
9. 语义差异、Python异常、原生崩溃、Sanitizer报告、超时分别归档；资源耗尽需单独复核。
10. 若原始任务可能创建子进程，补充进程组清理，避免只结束直接子进程。

**验收标准**

- 上一次运行成功、下一次在写结果前失败，不会读到上次结果并通过。
- 卡在导入、屏障、目标调用与结果转换时，可以通过事件识别最后阶段。
- 重复运行不会覆盖历史复现记录。
- sanitizer.enabled关闭/开启时，实际加载及报告状态与配置一致。

### R12 建立可移植的分层测试与回归集

**涉及文件**

- `tests/`
- `tests/fixtures/`（建议新增）
- `contracts/tensorflow/`（基础样例需纳入仓库）

**当前验证记录**

执行命令：

```bash
python -m unittest discover -s concurrent_oracle/tests -v
```

本次审查环境运行37项：20项通过、6项失败、9项错误、2项跳过。失败和错误主要来自`/root/fuzz_output_experiment/`中的harness及`/root/concurrent_oracle/contracts/tensorflow`缺失。本环境没有TensorFlow，未执行真实TF端到端用例；这些结果不代表项目容器中的算子执行结论。

**修改方案**

- 单元测试：比较、契约、计划、观测状态机，不依赖TF或/root绝对路径。
- fixture测试：使用仓库内最小harness和临时目录验证加载、转换与调用关联。
- TF集成测试：明确要求TensorFlow环境，覆盖标量、嵌套输出、随机输入冻结、共享关系和不同线程配置。
- 插桩专项测试：需要ASan等构建时单独启用，保存框架版本与环境。

不要只测试“正常不同输出”和“明显错误输出”。第一批回归用例必须包含非单射输出、相同描述失败、UNKNOWN契约、配置截断及旧观测污染。

**验收标准**

- 新检出仓库后，不依赖开发者/root目录即可运行全部纯逻辑测试。
- 未满足TF/插桩条件的专项测试明确skip，不伪装成功。
- 每个P0修复至少有一个能够复现旧行为的回归用例。
- 真实TF运行报告实际调用数和配置，不仅检查最终状态字段。

## 4. 文件改造分工建议

以下是职责划分建议，不要求一次性新增全部文件。

| 文件或目录 | 建议职责 |
| --- | --- |
| `schemas.py` | 计划身份、case身份、契约绑定、run/task/trial/call观测结构 |
| `case_capture.py` | 捕获真实参数、保存内容及来源、重放一致性检查 |
| `contract_registry.py` | 按场景/预言机查询契约及来源 |
| `scenario_generator.py` | 能力检查、配置生成、按场景分配计划预算 |
| `transformer.py` | 根据场景选择模板或构造后端 |
| `templates/` | baseline/test明确模式，独立调用、共享资源、Dataset等专用执行 |
| `runner.py` | 新进程运行、唯一run_id、环境快照、阶段与故障产物 |
| `result_comparator.py` | 统一结构递归、dtype规则和无法判定传播 |
| `oracles/` | 检查性质、适用条件、契约门控及证据输出 |
| `watchdog.py` | 消费进展事件、区分进程总时限与无进展时限 |
| `tests/fixtures/` | 小型harness、契约样例及可重复的正负对照 |

现有LLM配置也应顺带梳理：当前pipeline没有接入实际LLM client，transformer主要使用固定模板。`use_llm_transform`、`max_repair_attempts`等字段若暂不实现，应明确标为未支持；不要在报告中把配置声明视作已经调用模型或完成修复循环。

## 5. 第一阶段验收清单

- [ ] 相同合法输出不会被O3误报为串扰。
- [ ] 标量正常采集，describe_error不能产生PASS。
- [ ] 嵌套张量遵守与顶层一致的容差和结构比较规则。
- [ ] 实际目标输入可在独立进程中重放并核对。
- [ ] baseline和test使用独立进程及明确的比较关系。
- [ ] trials、线程设置与确定性策略作用于实际执行。
- [ ] 不同场景/预言机分别绑定契约。
- [ ] 计划预算不会静默丢弃重要场景。
- [ ] 独立、共享只读和共享资源场景的对象关系可以核对。
- [ ] 未实现能力、观测缺失和目标未调用均明确报告。
- [ ] 每次运行使用唯一run_id，不读旧结果、不覆盖复现历史。
- [ ] 纯逻辑测试不依赖/root绝对路径。

完成后再开展批量实验，统计API数、有效case数、计划数、配置运行数、可比较任务数和独立缺陷数。不要用异常目录数量替代独立缺陷数量，也不要用计划生成数量替代实际完成的测试规模。

## 6. 项目agent实施要求

1. 开始修改前确认当前提交与本清单审查提交之间的差异，已修复项直接复核，不重复改写。
2. 保留原始harness，只修改并发模块及必要的测试、配置与契约样例。
3. 按依赖推进：R01/R02/R09比较正确性，R03/R04输入与基准，R05/R06计划与契约，随后推进场景后端和专项能力。
4. 每项提交说明修改文件、解决的行为、回归测试和仍不支持的范围。
5. 改造计划或观测格式时显式声明版本；旧产物兼容或拒绝方式要清楚，避免静默误读。
6. 先完成小型正负对照，再扩大API规模，不通过降低检查强度来消除误报。
7. 交付时提供配置生效记录、测试结果、剩余任务清单和代表性复现命令。未运行的TF测试保留待测说明。

## 7. 参考资料

- [审查代码快照](https://github.com/yyds1233/ArgSpaceFuzz-TF/tree/54d75ffb860a601d67e14c32f07f79b05898198a/concurrent_oracle)
- [执行模板](https://github.com/yyds1233/ArgSpaceFuzz-TF/blob/54d75ffb860a601d67e14c32f07f79b05898198a/concurrent_oracle/templates/concurrent_harness.py)
- [比较器](https://github.com/yyds1233/ArgSpaceFuzz-TF/blob/54d75ffb860a601d67e14c32f07f79b05898198a/concurrent_oracle/result_comparator.py)
- [O3独立调用预言机](https://github.com/yyds1233/ArgSpaceFuzz-TF/blob/54d75ffb860a601d67e14c32f07f79b05898198a/concurrent_oracle/oracles/independence.py)
- [场景生成器](https://github.com/yyds1233/ArgSpaceFuzz-TF/blob/54d75ffb860a601d67e14c32f07f79b05898198a/concurrent_oracle/scenario_generator.py)
- [TensorFlow op determinism文档](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism)
- [TensorFlow Dataset文档](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

官方文档用于解释确定性与Dataset顺序等语义；正式契约仍应绑定项目实际测试的TensorFlow版本。

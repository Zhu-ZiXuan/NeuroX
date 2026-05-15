# NeuroX 代码审查报告

## 审查范围

本次审查覆盖 [neurox](neurox) 与 [example](example) 两个目录，重点关注目录组织、模块化与封装、docstring 与注释一致性、冗余与死代码、类型注解，以及其他不符合最佳实践的问题。

附加验证结果：make lint 通过；make check 当前失败，共报出 17 个错误，分布在 6 个文件中。这说明代码风格整体较统一，但类型契约、调试脚本和部分抽象边界仍然存在明显问题。

## 总体结论

项目的硬件抽象分层大体清楚，主路径也能串起来，但当前存在几类比较集中的问题：

1. 公共 API 的契约有静默失败风险，尤其是量化检查点加载与 profiler 指标语义。
2. operator、macro、profiling 三层之间存在抽象泄漏，导致协议不完整、类型系统失效、示例脚本不得不依赖内部细节。
3. example 目录混合了可复用示例与一次性调试脚本，同时存在较强环境耦合。
4. 类型注解表面上覆盖率不低，但关键位置的注解与真实行为不一致，已经能被 make check 复现。

## 高优先级问题

1. 量化模型加载使用了静默放宽的 strict=False，容易把检查点不匹配问题拖成运行时行为偏差。受影响位置：[neurox/api/replace.py](neurox/api/replace.py#L147-L170)。当前 build_evaluator 的文档要求输入是与 eval operator 对应的量化检查点，但 load_state_dict 使用 strict=False，会在缺少 weight_int、rescale 参数或层名漂移时继续运行。对于仿真器项目，这类错误不应该被静默吞掉，否则得到的是“能跑但不可信”的结果。

2. 示例校准流程在 train 模式下直接跑验证集，容易污染 BatchNorm 等训练态状态。受影响位置：[example/common/procedures.py](example/common/procedures.py#L64-L66)、[example/common/procedures.py](example/common/procedures.py#L81-L82)、[example/mobilenet/prepare.py](example/mobilenet/prepare.py#L20-L23)。run_prepare 先用验证集创建 val_loader，再在 _calibrate 中执行 model.train()。对于包含 BatchNorm 的模型，例如这里使用的 MobileNet V3 Small，这会在“校准”阶段改写 running mean 和 running var，而且数据源还是验证集，语义上不干净。

3. macro 抽象层没有把真实需要的物理态接口建模完整，XbarMacro 只能依赖 type: ignore 和具体实现细节。受影响位置：[neurox/macro/base.py](neurox/macro/base.py#L14-L29)、[neurox/macro/xbar_macro.py](neurox/macro/xbar_macro.py#L334-L413)。NeuroxMacroQuantMatMul 只描述了 fabricate 和 matmul，但 XbarMacro 在运行时又要求 physical_state 一定有 rram_g_prog__mS 和 _replace。现在这两个要求完全没有体现在协议里，只能靠 type: ignore[attr-defined] 兜底。这说明协议边界没有覆盖真实依赖，后续新增 macro 实现时很容易出现“通过静态检查但运行崩溃”的情况。

4. 库代码在导入时直接改写全局 Torch Dynamo 配置，副作用范围过大。受影响位置：[neurox/common/noise.py](neurox/common/noise.py#L20-L22)。noise 模块导入后会无条件执行 torch._dynamo.config.cache_size_limit = 128。这是全进程级副作用，不仅影响 NeuroX，也会影响调用方项目里其他 torch.compile 路径；这类策略更适合放到显式初始化逻辑或配置入口，而不是底层工具模块的 import-time side effect。

5. QAT 抽象中的 sync_qparams 目前是空实现，属于未完成或已经失效的接口设计。受影响位置：[neurox/operator/base.py](neurox/operator/base.py#L60)、[neurox/operator/linear.py](neurox/operator/linear.py#L171-L172)、[neurox/operator/conv2d.py](neurox/operator/conv2d.py#L384-L385)。基类把 sync_qparams 定义成抽象方法，但两个子类都只写了 pass，而且主流程里也没有真正依赖它。这样会误导后续维护者以为量化参数同步存在统一入口，实际上这个入口是空的。

## 中优先级问题

6. profiler 的数据模型存在重复实现，而且文档与实际统计口径已经开始漂移。受影响位置：[neurox/api/profiler.py](neurox/api/profiler.py#L44-L91)、[neurox/api/profiler.py](neurox/api/profiler.py#L99-L242)、[neurox/api/profiler.py](neurox/api/profiler.py#L274-L304)。ProfilerReport 和 NeuroxProfiler 暴露了两套几乎重复的聚合属性，但底层容器不同：前者用 list，后者的 area 用按模块名去重的 dict。与此同时，analyze_static 的 docstring 声称会收集 leakage power 和 latency，但实现只汇总了 area 与 leaky_energy，latency 仍停留在默认值 0.0。这里既有组织重复，也已经出现语义漂移。

7. dataclass 序列化工具的类型守卫不正确，既触发了 make check，也可能在运行时错误接受 dataclass 类型对象。受影响位置：[neurox/common/load_dump.py](neurox/common/load_dump.py#L32-L44)。is_dataclass 对 dataclass 类和实例都会返回 True，因此 dataclass_to_dict 当前可能把 dataclass 类本身放行，然后在 asdict(obj) 处报错。这个问题不是纯粹的类型标注瑕疵，而是防线写错了。

8. ADC.drive 的文档与实现不一致，当前接口语义不清楚。受影响位置：[neurox/analog/adc.py](neurox/analog/adc.py#L49-L57)、[neurox/analog/adc.py](neurox/analog/adc.py#L169-L177)。抽象接口与 GeneralADC.drive 的 docstring 都写成“返回信号和动态能耗”，但函数签名与实际实现只返回一个 Tensor。这会误导调用方，也让“ADC 参考驱动是否计入能耗”这个问题悬空。

9. Linear 与 Conv2d 的量化 operator 存在成片重复代码，说明可复用的量化骨架尚未被提炼出来。受影响位置：[neurox/operator/linear.py](neurox/operator/linear.py#L58-L65)、[neurox/operator/linear.py](neurox/operator/linear.py#L174-L234)、[neurox/operator/conv2d.py](neurox/operator/conv2d.py#L201-L208)、[neurox/operator/conv2d.py](neurox/operator/conv2d.py#L387-L453)。重复内容包括输入量化 buffer 注册、from_torch、_compute_qparams、_fold_bias、extract_quantized_params 等关键逻辑。当前重复还算可读，但随着量化参数或导出格式演进，两个分支很容易出现行为漂移。

10. macro 协议对外暴露的能力比实际使用能力更弱，导致 example 与类型系统都退化到“猜具体实现”。受影响位置：[neurox/macro/base.py](neurox/macro/base.py#L14-L29)、[neurox/macro/fake.py](neurox/macro/fake.py#L31-L47)、[neurox/macro/xbar_macro.py](neurox/macro/xbar_macro.py#L106-L145)、[example/alexnet/profile_tensors.py](example/alexnet/profile_tensors.py#L41-L43)。例如 FakeMacro 和 XbarMacro 都有 output_rescale_factor、latency__ns、leakage_power__uW 等属性，但协议里没有，结果 example 中一旦想做 profiling 或 introspection，就会出现类型不兼容甚至只能硬写具体类。

11. example 目录的组织混合了可复用示例和一次性 profiling 脚本，职责边界不够清晰。受影响位置：[example/alexnet/prepare.py](example/alexnet/prepare.py#L10-L27)、[example/alexnet/evaluate.py](example/alexnet/evaluate.py#L10-L25)、[example/mobilenet/prepare.py](example/mobilenet/prepare.py#L10-L27)、[example/mobilenet/evaluate.py](example/mobilenet/evaluate.py#L10-L25)、[example/alexnet/profile_memory.py](example/alexnet/profile_memory.py#L1-L66)、[example/alexnet/profile_tensors.py](example/alexnet/profile_tensors.py#L1-L145)。prepare 和 evaluate 是面向用户的流程示例，但同级目录里又放了强依赖内部实现、硬编码路径和 GPU 设备号的 profiling 脚本。建议至少把调试型脚本独立到 profiling 或 debug 子目录，避免 example 目录同时承担“文档示例”和“临时研究脚本”两种职责。

12. 共享示例入口和 profiling 脚本都存在较强环境耦合，降低可移植性。受影响位置：[example/common/cli.py](example/common/cli.py#L15)、[example/common/cli.py](example/common/cli.py#L33)、[example/alexnet/evaluate_1t1r.py](example/alexnet/evaluate_1t1r.py#L31-L32)、[example/alexnet/evaluate_1t1r.py](example/alexnet/evaluate_1t1r.py#L70)、[example/alexnet/profile_memory.py](example/alexnet/profile_memory.py#L17-L18)、[example/alexnet/profile_memory.py](example/alexnet/profile_memory.py#L26)、[example/alexnet/profile_tensors.py](example/alexnet/profile_tensors.py#L18-L19)、[example/alexnet/profile_tensors.py](example/alexnet/profile_tensors.py#L31)。共享 CLI 默认就是 cuda:1，而 profiling 脚本又硬编码了 cuda:3 和 /data/ImageNet。对一个示例目录来说，这种默认值过于具体，会让大多数环境第一次运行就失败。

13. 数字模块存在明显的模板式重复，基础能力没有抽成统一基类或 mixin。受影响位置：[neurox/digital/accumulator.py](neurox/digital/accumulator.py#L9-L56)、[neurox/digital/adder.py](neurox/digital/adder.py#L9-L56)、[neurox/digital/shift_adder.py](neurox/digital/shift_adder.py#L9-L69)、[neurox/digital/subtractor.py](neurox/digital/subtractor.py#L9-L61)、[neurox/digital/requantizer.py](neurox/digital/requantizer.py#L9-L47)。这些模块都有几乎相同的 config 字段、面积/漏电/时延属性转发和动态能耗计算框架。现在规模还可控，但一旦指标定义变化，就要在 5 个文件里重复改动。

## 低优先级问题

14. profiling 脚本大量依赖 monkey patch 和内部私有方法，维护成本偏高，也直接导致了当前的一批 mypy 错误。受影响位置：[example/alexnet/profile_tensors.py](example/alexnet/profile_tensors.py#L39-L43)、[example/alexnet/profile_tensors.py](example/alexnet/profile_tensors.py#L83)、[example/alexnet/profile_tensors.py](example/alexnet/profile_tensors.py#L138)、[example/alexnet/profile_memory.py](example/alexnet/profile_memory.py#L33-L41)。这些脚本通过直接替换 _eval_tiles、vec_mat_mul 等方法来插桩，虽然对研究型调试很方便，但它们已经不再是普通“示例”，而是对内部实现有强假设的实验脚本。

15. 注释与输出文案有局部漂移，降低了调试信息可信度。受影响位置：[example/alexnet/profile_tensors.py](example/alexnet/profile_tensors.py#L42)、[example/alexnet/profile_tensors.py](example/alexnet/profile_tensors.py#L140)、[example/alexnet/profile_memory.py](example/alexnet/profile_memory.py#L54)。profile_tensors 实际选择的是 features.3，但末尾注释写成“只关心 classifier.1”；profile_memory 在 forward 前打印 len(peak_per_layer)，此时字典还是空的，因此输出会是 0 层左右，和真实含义不符。

16. 代码库里存在确认未被使用的辅助函数，说明死代码清理还不彻底。受影响位置：[neurox/analog/adc.py](neurox/analog/adc.py#L13)、[neurox/operator/qat_util.py](neurox/operator/qat_util.py#L9)。_zero_gaussian 在仓库里没有任何调用；derive_multiplier_and_shift 也只有定义、没有引用，而实际使用的是 derive_multiplier_and_shift_tensor。对于核心仿真器代码，这类死代码会增加阅读噪音。

17. 类型注解在若干关键位置与真实行为不一致，已经直接反映为 make check 失败。受影响位置：[neurox/operator/linear.py](neurox/operator/linear.py#L24)、[neurox/operator/conv2d.py](neurox/operator/conv2d.py#L147)、[neurox/operator/conv2d.py](neurox/operator/conv2d.py#L275)、[neurox/device/selector.py](neurox/device/selector.py#L47-L54)、[neurox/analog/adc.py](neurox/analog/adc.py#L119)、[example/alexnet/profile_tensors.py](example/alexnet/profile_tensors.py#L25-L138)、[example/alexnet/profile_memory.py](example/alexnet/profile_memory.py#L33-L41)。最典型的是 bias_int 被标成可选，但实现里总是注册成 Tensor 并直接 view；Selector 依赖对 _buffers 的 cast 访问；example profiling 脚本则缺少关键函数注解并包含方法重绑定。当前类型标注更像“给编辑器提示”，还没有达到可作为设计契约的强度。

## 建议的修复优先级

1. 先收紧公共 API 契约：去掉或显式校验 strict=False，修正 calibration 流程，补全 macro 协议与 profiler 指标语义。
2. 再处理抽象与复用：清理 sync_qparams 空接口，抽取 operator 与 digital 模块中的共性骨架。
3. 最后整理 example：拆分调试脚本与正式示例，移除硬编码设备和数据路径，并补齐 profiling 脚本的类型注解。

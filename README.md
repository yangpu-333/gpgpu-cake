<a id="report-top"></a>

# 九格7B训练与微调性能优化研究报告

截至2026年10月27日。研究对象为官方九格7B，采用Megatron、TP2、BF16、S2048及CPU AdamW offload，覆盖继续预训练（CPT）与全参数指令微调（SFT）。

## 已验证成果：双A100

定位并修复FP32主参数在CPU复制、转换后丢失pinned属性的问题，改善主机到GPU的搬运与重叠，保持模型、数据和AdamW更新公式不变。

| 任务 | step耗时：原生→修复 | 输入吞吐：原生→修复 | 配对吞吐提升 |
|---|---:|---:|---:|
| CPT | 6.581→3.592秒 | 311.2→570.2 tokens/s | **84.4%** |
| SFT | 6.563→3.413秒 | 312.0→600.1 tokens/s | **92.9%** |

SFT配对时间线中，HtoD主机调用范围由约2.90秒/步降至4.33毫秒/步，CPU AdamW与HtoD重叠由0增至1.58秒/步，搬运量相同。三步全状态、原生保存恢复及真实下一步比较均通过，受检状态最大误差为0。

正式测量为三种子×三臂，每例120步、剔除前20步，关闭profiler，使用GPU同步计时及两rank慢端。提升为逐种子配对比的几何平均，与表内合并吞吐口径不同；CPT/SFT修复组种子均值CV分别为13.6%/9.24%。SFT答案监督吞吐为263.3→506.4 targets/s。

## 国产GPU研究：双BI-V150

完成CoreX 4.4真实7B迁移及三臂各35步采集，获得12个rank/window观测。现有证据支持三个研究判断：

- **CPU offload与同步是端到端优先方向。** GPU计算区间约1.70秒/步；本地moment候选optimizer范围仍约3.45秒，其中AdamW数学之外约2.11秒。12窗口未记录到CPU AdamW与HtoD重叠。
- **NUMA预算必须覆盖完整工作集。** 识别两rank约28.57GiB延后分配的FP32梯度及新增审计文件缓存，建立节点预算、实际页位置与容量验证方法。
- **重复前反向不能替代连续训练验证。** 输入与RNG一致时，捕获到embedding梯度微差及后续传播；严格模式固定权重16次重复通过，连续更新与旧参考的兼容检查仍未通过。

上述时间来自profiler窗口，仅用于归因；范围存在重叠，不可相加或换算正式加速比。BI-V150原执行配置已通过完整数值和原生恢复；新增NUMA/严格路径尚未完成验收，**目前没有新增BI-V150正式端到端加速结论**。

**新增算子基准。** 双BI-V150已顺序完成7B形状`[1,14,S,S]`的scale/mask/softmax前后向：S2048/S4096×BF16/FP32输入共8例、8份GPU trace。全输出与梯度有限，12行CPU float64抽样对照通过。BF16输入、FP32中间计算的子链中位耗时分别为11.54–11.60毫秒、44.40–44.66毫秒，每次13个GPU kernel，可用于下一步融合研究。输入为合成数据，不含QK/PV矩阵乘、整模型更新或TP通信，不代表训练性能通过。

## 下一阶段目标

| 优先级 | 研究任务 | 交付与验收 |
|---|---|---|
| 1 | 定位跨独立进程的embedding及连续更新差异；控制新增审计文件缓存 | 可复现的首个分歧证据；缓存写回前后内容与驻留验证，保留原数值标准 |
| 2 | 统一现有trace的算子、shape、搬运及同步归因；必要时用ixSYS补充短时间线 | CPU offload关键路径与GPU热点表，明确统计分母和未归因项 |
| 3 | 端到端优先验证offload回传/同步编排；GPU首选7B真实shape的mask/scale/softmax前后向融合 | 单一候选的语义、边界正确性及微基准；GEMM先建立真实shape基准再选题 |
| 4 | 完成三臂全状态、SFT语义、原生保存恢复，再做正式对照 | 无profiler的step延迟、有效输入tokens/s、答案targets/s，沿用原三种子协议 |

同组提供的7B/8B/4B原始trace、分析器及ixSYS资料用于交叉分析，既有工作归原作者。已独立核验172项文件哈希、复算四份SFT trace并通过分析器9项测试；7B/4B的BF16对齐未通过，8B仅通过短前向检查，不能直接继承其训练正确性或性能结论。截图中的8B GEMM形状不作为7B目标。

[算子基准复核](#appendix-operator) · [详细计划与验收标准](#appendix-plan) · [CPT正式结果](#appendix-cpt) · [SFT正式结果](#appendix-sft)

[下载离线单文件HTML](docs/jiuge7b/report.html) · [其他项目历史概览](docs/project-overview-20261027.md)


---

## 完整内容与复核证据

以下内容均在本文件内。表格用于阅读，折叠区保留完整关键复核JSON；也可从GitHub下载对应JSON。


<a id="appendix-plan"></a>

## 详细研究计划与验收标准

| 顺序 | 要做什么 | 交付与准入 |
|---|---|---|
| P0 数值与内存 | 固定初始权重、实际输入和RNG，沿embedding梯度→首步更新→次步前向定位跨独立进程差异；为新增自有审计pack做有界写回/退缓存 | 可复现的分歧证据；内容SHA、inode和驻留检查。保留旧参考、原容差与内存保护 |
| P1 同口径分析 | 复用runtime关联、exclusive归因和区间并集方法分析现有trace；必要时用ixSYS采短CUDA/NVTX窗口 | 算子/shape热点、搬运及同步依赖表，明确分母与未归因项 |
| P2 单一候选 | 端到端优先研究offload回传/批次/同步；GPU方向验证7B真实shape的mask/scale/softmax前后向融合 | 保留mask、cast、缩放舍入、重计算和梯度语义；先通过边界正确性与隔离微基准 |
| P3 模型验收 | 三臂完整数值/SFT、原生恢复通过后，补必要的新trace与无profiler正式对照 | 分别报告step延迟、有效输入tokens/s及答案targets/s |

真实7B配置：官方9G7B_MHA，7,668,831,744参数、28层、hidden=3584、FFN=18944、28个MHA头；TP2、BF16、S2048、micro/global batch=1、CPU AdamW offload。数据为48,818条中文Alpaca；CPT监督全文，SFT只监督答案及结束标记。

attention目标为每rank `[1,14,S,S]`，覆盖S2048/S4096。GEMM先对FC1 `[S,3584]×[3584,18944]`、FC2 `[S,9472]×[9472,3584]`及对应反向建基准，再决定是否优化。

验收标准：

1. 三臂各3次真实S2048 BF16更新，每rank199组、3585字段；原FullAudit逐块有限性/CPU float64比较，`atol=3e-4, rtol=1e-3`。覆盖source、answer mask、逐token loss、RNG、scheduler和TP；新增raw独立核验。
2. 原生iteration2保存，native与optimized恢复并真实执行step3；995项恢复、1195项下一步字段，以及HDO调用、398个moment恢复和原12项负例均须通过。
3. 三种子×三臂轮换，每例120步、剔除20步，GPU同步计时取两rank慢端，13条loss精确一致。native/control种子CV≥10%须完整九例确认；两baseline所有配对更快且各几何平均收益>1%，才接受收益。
4. GPU实验先核对资源与他人占用，保留NUMA/RSS/host保护。费用优先抵扣券；正式实验保留初始3小时，诊断与微基准不替代正式结果。


<a id="appendix-cpt"></a>

## CPT正式结果：双A100

| seed | arm | 秒/步 | 有效输入tokens/s | 保留步数 |
|---|---|---:|---:|---:|
| 63100 | native | 6.596589 | 310.463 | 100 |
| 63100 | control | 6.617616 | 309.477 | 100 |
| 63100 | optimized | 3.031859 | 675.493 | 100 |
| 63101 | native | 6.528131 | 313.719 | 100 |
| 63101 | control | 6.522060 | 314.011 | 100 |
| 63101 | optimized | 3.922645 | 522.097 | 100 |
| 63102 | native | 6.617288 | 309.492 | 100 |
| 63102 | control | 6.730243 | 304.298 | 100 |
| 63102 | optimized | 3.821173 | 535.961 | 100 |

配对吞吐提升为84.4%。合并吞吐与逐种子几何平均是不同统计量；修复组种子均值CV为13.6%。该结果不外推到BI-V150。

[独立JSON文件](docs/jiuge7b/evidence/cpt-formal.json)

<details>
<summary>展开完整复核数据</summary>

```json
{
  "schema": "jiuge7b-pinning-local-formal-review-v1",
  "all_9_cases_and_both_actual_ranks_verified": true,
  "all_adamw_states_reached_120": true,
  "no_skipped_steps": true,
  "case_gpu_occupancy_gates_passed": true,
  "source_analyzer_sha256": "cc956fd4f68f0f9535f706f7a54046d9cd6c95c071661f31b138c5aebf4005d6",
  "cases": [
    {
      "case": "pinning-formal-primary-63100-native",
      "seed": 63100,
      "arm": "native",
      "retained_interval_seconds_per_step": [
        6.487945604324341,
        6.435611987113953,
        6.558213067054749,
        6.747781872749329,
        6.696613502502442,
        6.45647280216217,
        6.530529737472534,
        6.616345524787903,
        6.632964015007019,
        6.80341260433197
      ],
      "mean_seconds_per_step": 6.596589071750641,
      "effective_tokens_per_second": 310.4634800991916,
      "retained_steps": 100,
      "rank_aggregation": "max seconds per aligned 10-step window",
      "interval_cv": 0.01894344072345723
    },
    {
      "case": "pinning-formal-primary-63100-control",
      "seed": 63100,
      "arm": "control",
      "retained_interval_seconds_per_step": [
        7.140278625488281,
        6.387320423126221,
        6.359623289108276,
        6.327520251274109,
        6.354228115081787,
        6.649939632415771,
        6.91784439086914,
        6.6538749694824215,
        6.786618852615357,
        6.598909616470337
      ],
      "mean_seconds_per_step": 6.61761581659317,
      "effective_tokens_per_second": 309.4770166114502,
      "retained_steps": 100,
      "rank_aggregation": "max seconds per aligned 10-step window",
      "interval_cv": 0.041159591056001074
    },
    {
      "case": "pinning-formal-primary-63100-optimized",
      "seed": 63100,
      "arm": "optimized",
      "retained_interval_seconds_per_step": [
        3.2948030471801757,
        3.29657039642334,
        3.2896162033081056,
        3.287817692756653,
        3.140045142173767,
        2.7996667623519897,
        2.7977221727371218,
        2.8005772113800047,
        2.8021171569824217,
        2.80965518951416
      ],
      "mean_seconds_per_step": 3.031859097480774,
      "effective_tokens_per_second": 675.4931328113895,
      "retained_steps": 100,
      "rank_aggregation": "max seconds per aligned 10-step window",
      "interval_cv": 0.08133221565323437
    },
    {
      "case": "pinning-formal-primary-63101-native",
      "seed": 63101,
      "arm": "native",
      "retained_interval_seconds_per_step": [
        6.832960796356201,
        6.746096897125244,
        6.622026324272156,
        6.5955328941345215,
        6.377528691291809,
        6.349135255813598,
        6.461628460884095,
        6.3850359439849855,
        6.334228992462158,
        6.5771328687667845
      ],
      "mean_seconds_per_step": 6.528130712509156,
      "effective_tokens_per_second": 313.71920848270355,
      "retained_steps": 100,
      "rank_aggregation": "max seconds per aligned 10-step window",
      "interval_cv": 0.026699040126395998
    },
    {
      "case": "pinning-formal-primary-63101-control",
      "seed": 63101,
      "arm": "control",
      "retained_interval_seconds_per_step": [
        6.47738242149353,
        6.551250553131103,
        6.704895710945129,
        6.387695455551148,
        6.495565056800842,
        6.503013610839844,
        6.474786376953125,
        6.643029999732971,
        6.6171023607254025,
        6.36587438583374
      ],
      "mean_seconds_per_step": 6.522059593200684,
      "effective_tokens_per_second": 314.01123690054317,
      "retained_steps": 100,
      "rank_aggregation": "max seconds per aligned 10-step window",
      "interval_cv": 0.016615497908307207
    },
    {
      "case": "pinning-formal-primary-63101-optimized",
      "seed": 63101,
      "arm": "optimized",
      "retained_interval_seconds_per_step": [
        4.343968105316162,
        3.9865413427352907,
        3.826153540611267,
        3.859147047996521,
        3.8847460031509398,
        3.8868401527404783,
        3.868663954734802,
        3.8840732097625734,
        3.856518864631653,
        3.8297977447509766
      ],
      "mean_seconds_per_step": 3.9226449966430663,
      "effective_tokens_per_second": 522.0966979557528,
      "retained_steps": 100,
      "rank_aggregation": "max seconds per aligned 10-step window",
      "interval_cv": 0.03940681272340352
    },
    {
      "case": "pinning-formal-primary-63102-native",
      "seed": 63102,
      "arm": "native",
      "retained_interval_seconds_per_step": [
        7.596609139442444,
        7.082291221618652,
        6.4757061958312985,
        6.411815214157104,
        6.357697534561157,
        6.307601451873779,
        6.307801008224487,
        6.362488508224487,
        6.380314230918884,
        6.890558552742005
      ],
      "mean_seconds_per_step": 6.61728830575943,
      "effective_tokens_per_second": 309.49233362214255,
      "retained_steps": 100,
      "rank_aggregation": "max seconds per aligned 10-step window",
      "interval_cv": 0.06553139077342222
    },
    {
      "case": "pinning-formal-primary-63102-control",
      "seed": 63102,
      "arm": "control",
      "retained_interval_seconds_per_step": [
        7.160420489311218,
        7.436274123191834,
        6.795439767837524,
        6.470556354522705,
        6.499416637420654,
        6.513255500793457,
        6.7990007400512695,
        6.854710125923157,
        6.438434672355652,
        6.334926128387451
      ],
      "mean_seconds_per_step": 6.730243453979492,
      "effective_tokens_per_second": 304.29805608132176,
      "retained_steps": 100,
      "rank_aggregation": "max seconds per aligned 10-step window",
      "interval_cv": 0.05235035772157111
    },
    {
      "case": "pinning-formal-primary-63102-optimized",
      "seed": 63102,
      "arm": "optimized",
      "retained_interval_seconds_per_step": [
        3.8703490018844606,
        3.8051167249679567,
        3.81312096118927,
        3.816561150550842,
        3.8212250232696534,
        3.826675009727478,
        3.8166300296783446,
        3.8151211977005004,
        3.8138670206069945,
        3.813059949874878
      ],
      "mean_seconds_per_step": 3.821172606945038,
      "effective_tokens_per_second": 535.9611330505535,
      "retained_steps": 100,
      "rank_aggregation": "max seconds per aligned 10-step window",
      "interval_cv": 0.00475315139141113
    }
  ],
  "seed_means": {
    "63100": {
      "native": 6.596589071750641,
      "control": 6.61761581659317,
      "optimized": 3.031859097480774
    },
    "63101": {
      "native": 6.528130712509156,
      "control": 6.522059593200684,
      "optimized": 3.9226449966430663
    },
    "63102": {
      "native": 6.61728830575943,
      "control": 6.730243453979492,
      "optimized": 3.821172606945038
    }
  },
  "arm_aggregate": {
    "native": {
      "mean_seconds_per_step": 6.580669363339742,
      "effective_tokens_per_second": 311.21454170136633
    },
    "control": {
      "mean_seconds_per_step": 6.623306287924448,
      "effective_tokens_per_second": 309.21112673498055
    },
    "optimized": {
      "mean_seconds_per_step": 3.591892233689626,
      "effective_tokens_per_second": 570.1730081963719
    }
  },
  "seed_mean_cv": {
    "native": 0.0070907756949103815,
    "control": 0.01573361060640571,
    "optimized": 0.13576394534239222
  },
  "vs_native": {
    "seed_throughput_ratios": [
      2.1757571376690508,
      1.66421654727762,
      1.731742840857911
    ],
    "geomean_throughput_ratio": 1.8440295855043458,
    "all_seeds_faster": true
  },
  "vs_control": {
    "seed_throughput_ratios": [
      2.182692402193712,
      1.6626688366605065,
      1.761303177393027
    ],
    "geomean_throughput_ratio": 1.8558554105992549,
    "all_seeds_faster": true
  },
  "gain_vs_control": 0.8558554105992549,
  "round_promotion_passed": true,
  "repeat_required": false,
  "formal_performance_passed": true
}
```

</details>


<a id="appendix-sft"></a>

## SFT正式结果：双A100

| arm | 秒/步 | 输入tokens/s | 答案targets/s |
|---|---:|---:|---:|
| native | 6.563320 | 312.037 | 263.331 |
| control | 6.474713 | 316.307 | 266.934 |
| optimized | 3.412722 | 600.107 | 506.435 |

配对吞吐提升为92.9%，三种子均更快，修复组CV为9.24%。完整数值和原生恢复最大误差为0。复核JSON的round字段只记录当轮统计，最终验收以顶层`formal_performance_passed=true`为准。

[独立JSON文件](docs/jiuge7b/evidence/sft-formal.json)

<details>
<summary>展开完整复核数据</summary>

```json
{
  "schema": "jiuge7b-sft-pinning-independent-completion-review-v1",
  "utc": "2026-10-20T12:01:24.219827+00:00",
  "all_validation_stages_completed": true,
  "formal_performance_passed": true,
  "rounds": [
    {
      "round": "primary",
      "arms": {
        "native": {
          "mean_seconds_per_step": 6.563319898446401,
          "aggregate_input_token_slots_per_second": 312.0372055131399,
          "aggregate_supervised_answer_targets_per_second": 263.3306558381291
        },
        "control": {
          "mean_seconds_per_step": 6.474713263511657,
          "aggregate_input_token_slots_per_second": 316.30744353445493,
          "aggregate_supervised_answer_targets_per_second": 266.93434334356164
        },
        "optimized": {
          "mean_seconds_per_step": 3.412721984386444,
          "aggregate_input_token_slots_per_second": 600.1074829329233,
          "aggregate_supervised_answer_targets_per_second": 506.43543225630197
        }
      },
      "formal_logged_loss_checks": [
        {
          "arm": "native",
          "seed": 63100,
          "intervals": 13,
          "regular_intervals": 12,
          "logged_iterations": [
            1,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120
          ],
          "extra_first_step_checked": true,
          "logged_loss_max_abs": 0.0,
          "passed": true,
          "scope": "posthoc aggregate loss check; no extra synchronization or GPU capture"
        },
        {
          "arm": "native",
          "seed": 63101,
          "intervals": 13,
          "regular_intervals": 12,
          "logged_iterations": [
            1,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120
          ],
          "extra_first_step_checked": true,
          "logged_loss_max_abs": 0.0,
          "passed": true,
          "scope": "posthoc aggregate loss check; no extra synchronization or GPU capture"
        },
        {
          "arm": "native",
          "seed": 63102,
          "intervals": 13,
          "regular_intervals": 12,
          "logged_iterations": [
            1,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120
          ],
          "extra_first_step_checked": true,
          "logged_loss_max_abs": 0.0,
          "passed": true,
          "scope": "posthoc aggregate loss check; no extra synchronization or GPU capture"
        },
        {
          "arm": "control",
          "seed": 63100,
          "intervals": 13,
          "regular_intervals": 12,
          "logged_iterations": [
            1,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120
          ],
          "extra_first_step_checked": true,
          "logged_loss_max_abs": 0.0,
          "passed": true,
          "scope": "posthoc aggregate loss check; no extra synchronization or GPU capture"
        },
        {
          "arm": "control",
          "seed": 63101,
          "intervals": 13,
          "regular_intervals": 12,
          "logged_iterations": [
            1,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120
          ],
          "extra_first_step_checked": true,
          "logged_loss_max_abs": 0.0,
          "passed": true,
          "scope": "posthoc aggregate loss check; no extra synchronization or GPU capture"
        },
        {
          "arm": "control",
          "seed": 63102,
          "intervals": 13,
          "regular_intervals": 12,
          "logged_iterations": [
            1,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120
          ],
          "extra_first_step_checked": true,
          "logged_loss_max_abs": 0.0,
          "passed": true,
          "scope": "posthoc aggregate loss check; no extra synchronization or GPU capture"
        },
        {
          "arm": "optimized",
          "seed": 63100,
          "intervals": 13,
          "regular_intervals": 12,
          "logged_iterations": [
            1,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120
          ],
          "extra_first_step_checked": true,
          "logged_loss_max_abs": 0.0,
          "passed": true,
          "scope": "posthoc aggregate loss check; no extra synchronization or GPU capture"
        },
        {
          "arm": "optimized",
          "seed": 63101,
          "intervals": 13,
          "regular_intervals": 12,
          "logged_iterations": [
            1,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120
          ],
          "extra_first_step_checked": true,
          "logged_loss_max_abs": 0.0,
          "passed": true,
          "scope": "posthoc aggregate loss check; no extra synchronization or GPU capture"
        },
        {
          "arm": "optimized",
          "seed": 63102,
          "intervals": 13,
          "regular_intervals": 12,
          "logged_iterations": [
            1,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120
          ],
          "extra_first_step_checked": true,
          "logged_loss_max_abs": 0.0,
          "passed": true,
          "scope": "posthoc aggregate loss check; no extra synchronization or GPU capture"
        }
      ],
      "schema": "jiuge7b-pinning-formal-round-v1",
      "seed_mean_cv": {
        "native": 0.016615676088611437,
        "control": 0.004121553899155234,
        "optimized": 0.09235236420146993
      },
      "vs_native": {
        "seed_throughput_ratios": [
          1.7362090083618429,
          2.1322519630722727,
          1.9375653599486875
        ],
        "geomean_throughput_ratio": 1.9285561884581284,
        "all_seeds_faster": true
      },
      "vs_control": {
        "seed_throughput_ratios": [
          1.735872221113173,
          2.106775028787087,
          1.8834958887353659
        ],
        "geomean_throughput_ratio": 1.9026848667698706,
        "all_seeds_faster": true
      },
      "gain_vs_control": 0.9026848667698706,
      "round_promotion_passed": true,
      "repeat_required": false,
      "all_expected_cases_verified": true,
      "formal_performance_passed": false,
      "note": "Round result only. Final promotion also requires any predeclared independent repeat."
    }
  ],
  "repeat_performed": false,
  "primary_full_numeric_and_resume_verified": true,
  "semantic_source_reconstruction_verified": true,
  "maximum_numeric_error": 0.0,
  "ACT_unique_original_chunks_verified": 18601,
  "verified_local_key_files": 726,
  "verified_local_key_bytes": 34398873,
  "local_large_raw_copy": false,
  "formal_throughput_unit": "input token slots/s;2048perstep;notTPsum",
  "old_CPT_84_4_percent_reused_as_SFT_result": false,
  "exact_original_physical_GPU_pair_preserved": true,
  "complete_SFT_parent_reused_without_CPT_numeric_references": true,
  "resource_handoff_only_no_mathematical_or_timing_standard_change": true,
  "postprocessing_recovery": "Megatron logs iteration1 in addition to 10..120; all13 checked",
  "failed_original_review_preserved": true,
  "existing_extracted_keys_reverified_without_overwrite": true
}
```

</details>


<a id="appendix-bi"></a>

## BI-V150原执行配置的正确性与恢复

旧执行配置三臂各3次真实更新，每rank3585字段最大差0；原生保存、两条真实恢复路径与下一步比较通过。该结论只对其原执行路径有效，不能直接套用到新增NUMA布局和严格执行路径。

[独立JSON文件](docs/jiuge7b/evidence/bi-correctness.json)

<details>
<summary>展开完整复核数据</summary>

```json
{
  "passed": true,
  "utc": "2026-10-26T14:05:38.646955+00:00",
  "rows": [
    {
      "arm": "native",
      "rank": 0,
      "complete_fields": 3585,
      "max_abs": 0.0,
      "manifest_SHA256_matches_remote_raw_gate": true
    },
    {
      "arm": "native",
      "rank": 1,
      "complete_fields": 3585,
      "max_abs": 0.0,
      "manifest_SHA256_matches_remote_raw_gate": true
    },
    {
      "arm": "control",
      "rank": 0,
      "complete_fields": 3585,
      "max_abs": 0.0,
      "manifest_SHA256_matches_remote_raw_gate": true
    },
    {
      "arm": "control",
      "rank": 1,
      "complete_fields": 3585,
      "max_abs": 0.0,
      "manifest_SHA256_matches_remote_raw_gate": true
    },
    {
      "arm": "optimized",
      "rank": 0,
      "complete_fields": 3585,
      "max_abs": 0.0,
      "manifest_SHA256_matches_remote_raw_gate": true
    },
    {
      "arm": "optimized",
      "rank": 1,
      "complete_fields": 3585,
      "max_abs": 0.0,
      "manifest_SHA256_matches_remote_raw_gate": true
    },
    {
      "arm": "resume-restored",
      "rank": 0,
      "complete_fields": 995,
      "max_abs": 0.0,
      "manifest_SHA256_matches_remote_raw_gate": true
    },
    {
      "arm": "resume-restored",
      "rank": 1,
      "complete_fields": 995,
      "max_abs": 0.0,
      "manifest_SHA256_matches_remote_raw_gate": true
    },
    {
      "arm": "resume-next-step",
      "rank": 0,
      "complete_fields": 1195,
      "max_abs": 0.0,
      "manifest_SHA256_matches_remote_raw_gate": true
    },
    {
      "arm": "resume-next-step",
      "rank": 1,
      "complete_fields": 1195,
      "max_abs": 0.0,
      "manifest_SHA256_matches_remote_raw_gate": true
    }
  ],
  "source_reconstruction": [
    {
      "arm": "native",
      "source_reconstructed_batches": 3,
      "both_ranks": true
    },
    {
      "arm": "control",
      "source_reconstructed_batches": 3,
      "both_ranks": true
    },
    {
      "arm": "optimized",
      "source_reconstructed_batches": 3,
      "both_ranks": true
    }
  ],
  "resume_policy_recomputed_locally": {
    "passed": true,
    "GPU_used": false,
    "ranks": [
      {
        "rank": 0,
        "native_initialization_plus_real_update": 2,
        "actual_training_updates": 1,
        "real_optimizer_iteration": 3,
        "initialization_moment_events": 398,
        "resume_copy_events": 398,
        "final_moment_objects": 398,
        "final_sampled_moment_pages_local": true,
        "resume_max_abs": 0.0,
        "passed": true
      },
      {
        "rank": 1,
        "native_initialization_plus_real_update": 2,
        "actual_training_updates": 1,
        "real_optimizer_iteration": 3,
        "initialization_moment_events": 398,
        "resume_copy_events": 398,
        "final_moment_objects": 398,
        "final_sampled_moment_pages_local": true,
        "resume_max_abs": 0.0,
        "passed": true
      }
    ],
    "negative_gate_regressions_rejected": 12,
    "native_CoreX_no_dist_API_used_without_redirect": true,
    "real_training_updates_per_resume": 1,
    "optimizer_iteration": 3,
    "formal_performance_passed": false
  },
  "full_raw_SHA_verified_on_DataThinking": true,
  "raw_chunk_count": 18603,
  "local_review_scope": "independent key/coverage/policy/source/SHA recomputation; tensor originals remain Data Thinking",
  "full_raw_local_copy_verified": false,
  "raw_tensors_downloaded": false,
  "full_model_correctness_and_native_resume_passed": true,
  "formal_performance_passed": false,
  "key_review_source_parent_SHA256": "d2116ab8cc6fef112871eeec5ef0919109f9ed27941f7d5e0bd59f67be79963d",
  "key_review_derived_SHA256": "c2a761cd42281dba5b24e70ec8bcb422744c1ee7c8fdabfe244d9c64a34efbdb",
  "same_window_raw_reuse_locally_recomputed": true,
  "raw_reuse_regression_negative_cases": 12,
  "original_resume_archive_AST_unchanged": true,
  "cross_Python_AST_serialization_proof": {
    "passed": true,
    "same_original_verifier_SHA256": "079624580909b92008a382ec4ca69c153d4e575cb150faffc3d13d53e43cece8",
    "Python313_compact_AST_SHA256": "e0df4e1b5b3f819d3fac5f64625d84db053240b0c89cb3f4c35873b0e961bf06",
    "Python312_expanded_AST_SHA256": "cc6493201290e222dfedd443dd0f1a365cbf1b911eeb18f529b827912eea6d26",
    "only_empty_AST_field_serialization_differs": true
  },
  "corrupt_AST_proof_negative_cases_rejected": 2,
  "versioned_local_review_SHA256": "21a9ba9bc5bf2ac70b366c0814da73a4ecfd463eb568083d931c392cfcbc21ba",
  "host_cache_actual_7B_reclamation": [
    {
      "rank": 0,
      "record_SHA256": "4561a99cd1fc0acde922335bc3be9082e273bf2ac93bb487b4d84cd5051735ca",
      "records": [
        {
          "label": "before-HDO-call-0",
          "reserved_before": 56643584011,
          "reserved_after": 28321792000,
          "RSS_before": 58613854208,
          "RSS_after": 30199865344
        },
        {
          "label": "resume-restored:after",
          "reserved_before": 113287168009,
          "reserved_after": 56643584000,
          "RSS_before": 145886392320,
          "RSS_after": 89218527232
        },
        {
          "label": "resume-next-step:before",
          "reserved_before": 56647835660,
          "reserved_after": 56647835648,
          "RSS_before": 89754091520,
          "RSS_after": 89691160576
        },
        {
          "label": "before-HDO-call-1",
          "reserved_before": 56647835648,
          "reserved_after": 56647835648,
          "RSS_before": 89691160576,
          "RSS_after": 89691160576
        },
        {
          "label": "resume-next-step:after",
          "reserved_before": 84969627648,
          "reserved_after": 56647835648,
          "RSS_before": 118072897536,
          "RSS_after": 89693855744
        }
      ]
    },
    {
      "rank": 1,
      "record_SHA256": "83aa18bbc567b04143c5943844837391889839bc48a68c0e05faaae8a6fc37fd",
      "records": [
        {
          "label": "before-HDO-call-0",
          "reserved_before": 56643584011,
          "reserved_after": 28321792000,
          "RSS_before": 58612621312,
          "RSS_after": 30198616064
        },
        {
          "label": "resume-restored:after",
          "reserved_before": 113287168009,
          "reserved_after": 56643584000,
          "RSS_before": 145885093888,
          "RSS_after": 89217196032
        },
        {
          "label": "resume-next-step:before",
          "reserved_before": 56643584012,
          "reserved_after": 56643584000,
          "RSS_before": 89722871808,
          "RSS_after": 89658060800
        },
        {
          "label": "before-HDO-call-1",
          "reserved_before": 56643584000,
          "reserved_after": 56643584000,
          "RSS_before": 89658060800,
          "RSS_after": 89658060800
        },
        {
          "label": "resume-next-step:after",
          "reserved_before": 84965376000,
          "reserved_after": 56643584000,
          "RSS_before": 118042738688,
          "RSS_after": 89660956672
        }
      ]
    }
  ],
  "host_cache_small_probe_and_original_guard_regressions_passed": true,
  "bundle_SHA256": "fb77e04ff66f66964b3b4de2971ec5e5ae8d8687323bb6452e90d46c3b30a97c",
  "key_zip_bytes": 5186022,
  "key_file_count": 134,
  "key_uncompressed_bytes": 31620664,
  "max_points": 0,
  "campaign_start_epoch": 1793005196,
  "deadline_epoch": 1793026796,
  "existing_key_bundle_reused_without_GPU_or_raw_verifier_rerun": true
}
```

</details>


<a id="appendix-trace"></a>

## BI-V150十二窗口瓶颈分析

| 臂 | 窗口耗时 | CPU AdamW | optimizer整体 | AdamW之外 | GPU计算并集 |
|---|---:|---:|---:|---:|---:|
| pinned基线 | 6.880 | 2.087 | 4.082 | 1.996 | 1.701 |
| 绑核对照 | 5.810 | 1.539 | 3.575 | 2.036 | 1.702 |
| 本地moment候选 | 5.626 | 1.337 | 3.446 | 2.110 | 1.702 |

单位为秒/rank/步。各臂35步，采第21–25和31–35步、两rank，共12个rank/window。它们不是独立种子；范围可重叠，不可相加为step耗时，不用于正式加速判断。12窗口CPU AdamW/HtoD记录重叠均为0。候选未记录GPU活动空档代理30.6%–33.8%，不等于硬件利用率；rank间collective时长差不能全归因为链路慢。

[独立JSON文件](docs/jiuge7b/evidence/bi-trace.json)

<details>
<summary>展开完整复核数据</summary>

```json
{
  "platform": "Data Thinking BI-V150 TP2 CoreX4.4",
  "objective": "answer_only_SFT",
  "source_analysis_SHA256": "32cd5463bec583bdfb3146981a788b2499637f674908b977a02d7d5b7cf4b1a7",
  "trace_windows": [
    {
      "window": "window-01",
      "rank": 0,
      "steps": [
        21,
        22,
        23,
        24,
        25
      ],
      "trace_sha256": "21458ece0945ca06d5366b46a9de1ed32e149b982d3271414eeb2867c274e466",
      "decompressed_sha256": "39508b48063fe5e72bb9454ad05bb366d829c1d75c0c0d9514d980e58ae1fb41",
      "case": "corex-S2048-trace-native-sl02"
    },
    {
      "window": "window-01",
      "rank": 1,
      "steps": [
        21,
        22,
        23,
        24,
        25
      ],
      "trace_sha256": "3d48fa04f3d88945ad13a3536243c5205161b6bd5071f75fb68488c83798bd90",
      "decompressed_sha256": "f2ae7bf88841b9be09b41995830f7e408a1221482e28e5fe623442e92791967c",
      "case": "corex-S2048-trace-native-sl02"
    },
    {
      "window": "window-02",
      "rank": 0,
      "steps": [
        31,
        32,
        33,
        34,
        35
      ],
      "trace_sha256": "20a36bd5980514b7bd001260201a17ceeb81ccc4bfeac0a11d2fc7ff23e1782b",
      "decompressed_sha256": "bca367c2ca7dc9f865aa567e9d40c850dabdddccd018ea6d2689999e1e52e327",
      "case": "corex-S2048-trace-native-sl02"
    },
    {
      "window": "window-02",
      "rank": 1,
      "steps": [
        31,
        32,
        33,
        34,
        35
      ],
      "trace_sha256": "09e9348ea567ea25d16926f6a20cc630537aebf355ec3648eb59fc4790f93f79",
      "decompressed_sha256": "eef4c1507fbed83d24580f55a4942358989f83711347063a3fc9393a523d188c",
      "case": "corex-S2048-trace-native-sl02"
    },
    {
      "window": "window-01",
      "rank": 0,
      "steps": [
        21,
        22,
        23,
        24,
        25
      ],
      "trace_sha256": "46d5fd892540296e46bbd587281bf69834f2ff7e10c4648cd223a7e16e33fc6f",
      "decompressed_sha256": "bd245ba050345760dc7380774701320986219295587b2ebc7ae0303460c39d46",
      "case": "corex-S2048-trace-control-sl02"
    },
    {
      "window": "window-01",
      "rank": 1,
      "steps": [
        21,
        22,
        23,
        24,
        25
      ],
      "trace_sha256": "b6d5b6e8731390b5ab432d2b806e4d1d20fcb7cd817e81db746c7e553b84433e",
      "decompressed_sha256": "98874fc301e974b953faae9393163f6da69d55ff8f5e7f8558dc4ec83e2f71cb",
      "case": "corex-S2048-trace-control-sl02"
    },
    {
      "window": "window-02",
      "rank": 0,
      "steps": [
        31,
        32,
        33,
        34,
        35
      ],
      "trace_sha256": "1ec7f4281c74ad1612d7cacf0abf9786df8106faf32396b96f733db858e2339a",
      "decompressed_sha256": "ce26a4eb268f973b4bac0764faf64adfb28656daa7447a972d380d2925581ec7",
      "case": "corex-S2048-trace-control-sl02"
    },
    {
      "window": "window-02",
      "rank": 1,
      "steps": [
        31,
        32,
        33,
        34,
        35
      ],
      "trace_sha256": "eb1339fa5a3dcc9b17ee569e2a037603613fffe8f8fbcbfac56c02c05578c10f",
      "decompressed_sha256": "4624c3dd356f2cd49b89f72bde95ecdc61f65f2a70a4098abe52eb4328646e80",
      "case": "corex-S2048-trace-control-sl02"
    },
    {
      "window": "window-01",
      "rank": 0,
      "steps": [
        21,
        22,
        23,
        24,
        25
      ],
      "trace_sha256": "f6ef6567f2ec9a2bb9c935dd3052c717ff265a1d14f75b58a0366df41cbae2f1",
      "decompressed_sha256": "a25879b5acd380f67437689c5b5064b3403a1782c8174b79e4e62c8ed4ad7a49",
      "case": "corex-S2048-trace-optimized-fu01"
    },
    {
      "window": "window-01",
      "rank": 1,
      "steps": [
        21,
        22,
        23,
        24,
        25
      ],
      "trace_sha256": "c455d9441f708f3f2f7c0b70966e7d694e69b0e4e34bdc97b622142dbc96bd3d",
      "decompressed_sha256": "10ce9cfcae9367f4746d33894c072b23b917c74ea5ad2712c3219c2e3c4c1707",
      "case": "corex-S2048-trace-optimized-fu01"
    },
    {
      "window": "window-02",
      "rank": 0,
      "steps": [
        31,
        32,
        33,
        34,
        35
      ],
      "trace_sha256": "d5977f3d0b667923596e25ea6391164ea66011e77b49d266b5416592b215f481",
      "decompressed_sha256": "265888ec38e3660215ca478e621e424c133cf3c9009496d3e5e1b28de5bee46e",
      "case": "corex-S2048-trace-optimized-fu01"
    },
    {
      "window": "window-02",
      "rank": 1,
      "steps": [
        31,
        32,
        33,
        34,
        35
      ],
      "trace_sha256": "84abe88accf66ab2a88bd7704a63508348efa341ea83dfba340e8bb7113fe6f5",
      "decompressed_sha256": "dbbb7ea5fdd7f21f5ed04b5d14544beb582a3b941349a1c135d0cb56f091fe37",
      "case": "corex-S2048-trace-optimized-fu01"
    }
  ],
  "summaries": [
    {
      "case": "corex-S2048-trace-native-sl02",
      "rank_window_observations": 4,
      "window_step_seconds_mean": 6.879624096057129,
      "CPUAdamW_seconds_per_rank_step": 2.0868928639404296,
      "optimizer_seconds_per_rank_step": 4.082399140686035,
      "optimizer_outside_AdamW_seconds_per_rank_step": 1.9955062767456055,
      "GPU_compute_seconds_per_rank_step": 1.7012239693237303,
      "HtoD_seconds_per_rank_step": 0.6795936127563477,
      "GPU_idle_proxy_percent_range": [
        32.73941597742518,
        47.02506349952989
      ],
      "all_CPUAdamW_HtoD_recorded_overlap_zero": true,
      "all_compute_collective_recorded_overlap_zero": true,
      "collective_seconds_per_rank_step": {
        "0": 0.20695556286621095,
        "1": 1.0768614037841797
      }
    },
    {
      "case": "corex-S2048-trace-control-sl02",
      "rank_window_observations": 4,
      "window_step_seconds_mean": 5.809531743054199,
      "CPUAdamW_seconds_per_rank_step": 1.5389487947021485,
      "optimizer_seconds_per_rank_step": 3.5750690804077148,
      "optimizer_outside_AdamW_seconds_per_rank_step": 2.0361202857055662,
      "GPU_compute_seconds_per_rank_step": 1.7021800504272462,
      "HtoD_seconds_per_rank_step": 0.7238684170043945,
      "GPU_idle_proxy_percent_range": [
        32.610777075736195,
        36.08110556458591
      ],
      "all_CPUAdamW_HtoD_recorded_overlap_zero": true,
      "all_compute_collective_recorded_overlap_zero": true,
      "collective_seconds_per_rank_step": {
        "0": 0.22462256286621093,
        "1": 0.31284826892089845
      }
    },
    {
      "case": "corex-S2048-trace-optimized-fu01",
      "rank_window_observations": 4,
      "window_step_seconds_mean": 5.625814999658203,
      "CPUAdamW_seconds_per_rank_step": 1.3369099299682619,
      "optimizer_seconds_per_rank_step": 3.446460483190918,
      "optimizer_outside_AdamW_seconds_per_rank_step": 2.109550553222656,
      "GPU_compute_seconds_per_rank_step": 1.7018729405639648,
      "HtoD_seconds_per_rank_step": 0.7846714262939454,
      "GPU_idle_proxy_percent_range": [
        30.586543210176544,
        33.786194779801384
      ],
      "all_CPUAdamW_HtoD_recorded_overlap_zero": true,
      "all_compute_collective_recorded_overlap_zero": true,
      "collective_seconds_per_rank_step": {
        "0": 0.18815748044433594,
        "1": 0.24668251774902344
      }
    }
  ],
  "cross_window_attribution_only": true,
  "rank_windows_are_not_independent_seeds": true,
  "range_categories_are_not_additive_wall_time": true,
  "idle_is_unrecorded_activity_proxy_not_hardware_utilization": true,
  "formal_performance_accepted": false,
  "strict_repair_new_trace_collected": false,
  "created_epoch": 1793092583.8759742
}
```

</details>


<a id="appendix-operator"></a>

## 新增八例算子基准与GPU trace

| GPU | S | 输入 | 无profiler中位毫秒 | kernel数 | 抽样前向最大差 | 抽样反向最大差 |
|---|---:|---|---:|---:|---:|---:|
| 0 | 2048 | bfloat16 | 11.540 | 13 | 1.81e-10 | 0 |
| 0 | 2048 | float32 | 9.846 | 9 | 2.47e-10 | 5.82e-11 |
| 0 | 4096 | bfloat16 | 44.398 | 13 | 8.98e-11 | 0 |
| 0 | 4096 | float32 | 36.784 | 9 | 1.25e-10 | 2.91e-11 |
| 1 | 2048 | bfloat16 | 11.596 | 13 | 1.81e-10 | 0 |
| 1 | 2048 | float32 | 9.596 | 9 | 2.47e-10 | 5.82e-11 |
| 1 | 4096 | bfloat16 | 44.655 | 13 | 8.98e-11 | 0 |
| 1 | 4096 | float32 | 36.575 | 9 | 1.25e-10 | 2.91e-11 |

子链为BF16/FP32输入→FP32缩放→有限值-10000因果mask→FP32 softmax及反向；合成输入，shape为`[1,14,S,S]`。每例2次预热、7次无profiler同步计时，另采1次GPU trace。全输出/梯度有限，每例12行CPU float64抽样比较通过原容差。BF16输入路径记录13次kernel启动，FP32为9次，支持进一步研究cast和中间矩阵搬运；不能把两条精度路径的差异称为优化收益。未包含QK/PV矩阵乘、整模型更新或TP通信。

[独立JSON文件](docs/jiuge7b/evidence/operator-validation.json)

<details>
<summary>展开完整复核数据</summary>

```json
{
  "passed_resource_and_isolated_operator_validation": true,
  "full_model_training_accepted": false,
  "TP2_communication_validated": false,
  "optimization_speedup_accepted": false,
  "scope": "synthetic-input actual7B attention shapes; sampled CPU float64 rows, not complete tensor/full model gate",
  "closed_cases": 8,
  "closed_GPU_traces": 8,
  "raw_traces_remain_DT": true,
  "double_end_key_SHA256": "ef8f6166ec326d6b2fa8423c6f2170df56ca95723fc5d7ae4bea07346c29858c",
  "key_evidence": "results/datathinking-ssh-20261025-01/resource-minimal-attention-closed-keys-20261027-02.json",
  "source_end_SHA_equal": true,
  "all_own_probe_processes_ended": true,
  "post_GPU_idle": true,
  "summary": [
    {
      "device": 0,
      "S": 2048,
      "dtype": "torch.bfloat16",
      "median_ms": 11.540129780769348,
      "kernels": 13,
      "forward_error": 1.807562036246907e-10,
      "backward_error": 0.0,
      "max_GPU_reserved_GiB": 1.060546875
    },
    {
      "device": 0,
      "S": 2048,
      "dtype": "torch.float32",
      "median_ms": 9.846318513154984,
      "kernels": 9,
      "forward_error": 2.466347000761038e-10,
      "backward_error": 5.820766091346741e-11,
      "max_GPU_reserved_GiB": 1.279296875
    },
    {
      "device": 0,
      "S": 4096,
      "dtype": "torch.bfloat16",
      "median_ms": 44.397871010005474,
      "kernels": 13,
      "forward_error": 8.982778724412471e-11,
      "backward_error": 0.0,
      "max_GPU_reserved_GiB": 4.189453125
    },
    {
      "device": 0,
      "S": 4096,
      "dtype": "torch.float32",
      "median_ms": 36.78358066827059,
      "kernels": 9,
      "forward_error": 1.2461759846694315e-10,
      "backward_error": 2.9103830456733704e-11,
      "max_GPU_reserved_GiB": 5.064453125
    },
    {
      "device": 1,
      "S": 2048,
      "dtype": "torch.bfloat16",
      "median_ms": 11.5959532558918,
      "kernels": 13,
      "forward_error": 1.807562036246907e-10,
      "backward_error": 0.0,
      "max_GPU_reserved_GiB": 1.060546875
    },
    {
      "device": 1,
      "S": 2048,
      "dtype": "torch.float32",
      "median_ms": 9.596055373549461,
      "kernels": 9,
      "forward_error": 2.466347000761038e-10,
      "backward_error": 5.820766091346741e-11,
      "max_GPU_reserved_GiB": 1.279296875
    },
    {
      "device": 1,
      "S": 4096,
      "dtype": "torch.bfloat16",
      "median_ms": 44.655460864305496,
      "kernels": 13,
      "forward_error": 8.982778724412471e-11,
      "backward_error": 0.0,
      "max_GPU_reserved_GiB": 4.189453125
    },
    {
      "device": 1,
      "S": 4096,
      "dtype": "torch.float32",
      "median_ms": 36.57521586865187,
      "kernels": 9,
      "forward_error": 1.2461759846694315e-10,
      "backward_error": 2.9103830456733704e-11,
      "max_GPU_reserved_GiB": 5.064453125
    }
  ],
  "minimal_validation_elapsed_seconds": 20.970463037490845,
  "private_source_guard_byte_identical_to_prior_control": true
}
```

</details>


<a id="appendix-failures"></a>

## 失败原因与未完成项

| 问题 | 已确定的事实 | 目前结论 |
|---|---|---|
| NUMA工作集 | 两rank约28.57GiB延后分配的FP32梯度未完整纳入早期预算；新增审计pack也形成大量文件缓存 | 公共预分配与容量探针有证据；新增完整路径仍须通过原保护 |
| 连续更新差异 | 实际输入/RNG相同，已捕获embedding梯度微差；严格固定权重重复通过，真实连续更新对旧参考仍失败 | 不能指认具体kernel或声称长训练修好，保留原数值标准 |
| 审计布局与入口 | 出现过参考符号链接越界、整块缓冲峰值及模块路径错误 | 失败保留，CPU修复版本化；不把编排修复当训练优化 |
| 检查器断档 | Windows关机导致本机观察器中断；远端持久监督与故障终态探针随后通过 | 与NUMA拒绝、数值失败分别记录 |
| 资源准入 | 当前新增GPU配额为0，现有GPU本地节点余量不足整模型80GiB准入；ACT暂不可达 | 先在现有卡做独立算子研究；不动不明归属环境 |

最近严格native尝试在原NUMA16GiB保护下停止。首步受检字段在原容差内，第二步部分梯度和逐token loss超原容差；三臂完整数值、原生恢复、长loss与正式性能均未接受。预算结束和安全停止是执行边界，不替代数值失败的结论。

[独立JSON文件](docs/jiuge7b/evidence/bi-numeric-failure.json)

<details>
<summary>展开完整复核数据</summary>

```json
{
  "local_failed_keys_review_passed": true,
  "both_ends_ZIP_and_key_SHA_verified": true,
  "key_receipt": {
    "ZIP_SHA256": "782251e48fb54d240e3bf73adfd74c7c2e06628ece77a90b4162ea421293f947",
    "ZIP_bytes": 2213323,
    "expanded_bytes": 13455807,
    "key_count": 32,
    "failed_partial_case_only": true,
    "no_raw_tensor_payload_downloaded": true
  },
  "original_NUMA16_guard_correctly_stopped": true,
  "stop": {
    "RSS_bytes": 181257265152,
    "host_available": 882955321344,
    "node1_MemFree_bytes": 16996687872,
    "epoch": 1793091408.9477134
  },
  "comparisons": [
    {
      "rank": 0,
      "closed_fields": 1831,
      "per_step_phase": {
        "step-1/gradient": {
          "fields": 199,
          "failed": 0,
          "SHA_different": 1,
          "max_abs": 9.597133612260222e-06
        },
        "step-1/updated_parameter": {
          "fields": 199,
          "failed": 0,
          "SHA_different": 1,
          "max_abs": 1.52587890625e-05
        },
        "step-1/master_param": {
          "fields": 199,
          "failed": 0,
          "SHA_different": 1,
          "max_abs": 1.9613653421401978e-05
        },
        "step-1/exp_avg": {
          "fields": 199,
          "failed": 0,
          "SHA_different": 1,
          "max_abs": 9.597133612260222e-07
        },
        "step-1/exp_avg_sq": {
          "fields": 199,
          "failed": 0,
          "SHA_different": 1,
          "max_abs": 1.6924273094076625e-10
        },
        "step-1/step": {
          "fields": 199,
          "failed": 0,
          "SHA_different": 0,
          "max_abs": 0.0
        },
        "step-1/lm loss": {
          "fields": 1,
          "failed": 0,
          "SHA_different": 0,
          "max_abs": 0.0
        },
        "step-2/gradient": {
          "fields": 199,
          "failed": 24,
          "SHA_different": 199,
          "max_abs": 0.0017784093506634235
        },
        "step-2/updated_parameter": {
          "fields": 88,
          "failed": 0,
          "SHA_different": 88,
          "max_abs": 3.0517578125e-05
        },
        "step-2/master_param": {
          "fields": 88,
          "failed": 0,
          "SHA_different": 88,
          "max_abs": 1.7212238162755966e-05
        },
        "step-2/exp_avg": {
          "fields": 87,
          "failed": 0,
          "SHA_different": 87,
          "max_abs": 0.00017784094234229997
        },
        "step-2/exp_avg_sq": {
          "fields": 87,
          "failed": 0,
          "SHA_different": 87,
          "max_abs": 4.92474646307528e-06
        },
        "step-2/step": {
          "fields": 87,
          "failed": 0,
          "SHA_different": 0,
          "max_abs": 0.0
        }
      },
      "failed_fields": [
        {
          "field": "step-2/0/gradient",
          "max_abs": 0.00031440809834748507
        },
        {
          "field": "step-2/3/gradient",
          "max_abs": 0.0017784093506634235
        },
        {
          "field": "step-2/6/gradient",
          "max_abs": 0.0003581121563911438
        },
        {
          "field": "step-2/7/gradient",
          "max_abs": 0.0007656477391719818
        },
        {
          "field": "step-2/15/gradient",
          "max_abs": 0.0005323805380612612
        },
        {
          "field": "step-2/39/gradient",
          "max_abs": 0.0004611470067175105
        },
        {
          "field": "step-2/43/gradient",
          "max_abs": 0.0005700588226318359
        },
        {
          "field": "step-2/47/gradient",
          "max_abs": 0.0003679213114082813
        },
        {
          "field": "step-2/51/gradient",
          "max_abs": 0.001479336991906166
        },
        {
          "field": "step-2/55/gradient",
          "max_abs": 0.0003939089656341821
        },
        {
          "field": "step-2/59/gradient",
          "max_abs": 0.0003328805323690176
        },
        {
          "field": "step-2/67/gradient",
          "max_abs": 0.0005953796207904816
        },
        {
          "field": "step-2/74/gradient",
          "max_abs": 0.0003711602184921503
        },
        {
          "field": "step-2/79/gradient",
          "max_abs": 0.0005403314717113972
        },
        {
          "field": "step-2/86/gradient",
          "max_abs": 0.000524252318427898
        },
        {
          "field": "step-2/87/gradient",
          "max_abs": 0.0007179213571362197
        },
        {
          "field": "step-2/90/gradient",
          "max_abs": 0.00042681791819632053
        },
        {
          "field": "step-2/91/gradient",
          "max_abs": 0.0006246936973184347
        },
        {
          "field": "step-2/94/gradient",
          "max_abs": 0.0006553097628057003
        },
        {
          "field": "step-2/98/gradient",
          "max_abs": 0.000535621540620923
        },
        {
          "field": "step-2/102/gradient",
          "max_abs": 0.0007324641337618232
        },
        {
          "field": "step-2/106/gradient",
          "max_abs": 0.0008764492231421173
        },
        {
          "field": "step-2/107/gradient",
          "max_abs": 0.0004175463691353798
        },
        {
          "field": "step-2/183/gradient",
          "max_abs": 0.0006536913570016623
        }
      ],
      "complete_three_step_gate_passed": false
    },
    {
      "rank": 1,
      "closed_fields": 1819,
      "per_step_phase": {
        "step-1/gradient": {
          "fields": 199,
          "failed": 0,
          "SHA_different": 1,
          "max_abs": 1.1996398825431243e-06
        },
        "step-1/updated_parameter": {
          "fields": 199,
          "failed": 0,
          "SHA_different": 1,
          "max_abs": 3.0517578125e-05
        },
        "step-1/master_param": {
          "fields": 199,
          "failed": 0,
          "SHA_different": 1,
          "max_abs": 1.6383826732635498e-05
        },
        "step-1/exp_avg": {
          "fields": 199,
          "failed": 0,
          "SHA_different": 1,
          "max_abs": 1.1996417015325278e-07
        },
        "step-1/exp_avg_sq": {
          "fields": 199,
          "failed": 0,
          "SHA_different": 1,
          "max_abs": 7.57360840708543e-12
        },
        "step-1/step": {
          "fields": 199,
          "failed": 0,
          "SHA_different": 0,
          "max_abs": 0.0
        },
        "step-1/lm loss": {
          "fields": 1,
          "failed": 0,
          "SHA_different": 0,
          "max_abs": 0.0
        },
        "step-2/gradient": {
          "fields": 199,
          "failed": 18,
          "SHA_different": 199,
          "max_abs": 0.0013362369500100613
        },
        "step-2/updated_parameter": {
          "fields": 85,
          "failed": 0,
          "SHA_different": 85,
          "max_abs": 3.0517578125e-05
        },
        "step-2/master_param": {
          "fields": 85,
          "failed": 0,
          "SHA_different": 85,
          "max_abs": 1.703668385744095e-05
        },
        "step-2/exp_avg": {
          "fields": 85,
          "failed": 0,
          "SHA_different": 85,
          "max_abs": 7.956300396472216e-05
        },
        "step-2/exp_avg_sq": {
          "fields": 85,
          "failed": 0,
          "SHA_different": 85,
          "max_abs": 5.046128990215948e-07
        },
        "step-2/step": {
          "fields": 85,
          "failed": 0,
          "SHA_different": 0,
          "max_abs": 0.0
        }
      },
      "failed_fields": [
        {
          "field": "step-2/35/gradient",
          "max_abs": 0.0003128617536276579
        },
        {
          "field": "step-2/43/gradient",
          "max_abs": 0.00035820365883409977
        },
        {
          "field": "step-2/47/gradient",
          "max_abs": 0.0003736679209396243
        },
        {
          "field": "step-2/51/gradient",
          "max_abs": 0.00045449123717844486
        },
        {
          "field": "step-2/55/gradient",
          "max_abs": 0.0003031713713426143
        },
        {
          "field": "step-2/63/gradient",
          "max_abs": 0.0005680056056007743
        },
        {
          "field": "step-2/67/gradient",
          "max_abs": 0.0003699129447340965
        },
        {
          "field": "step-2/74/gradient",
          "max_abs": 0.0007956298068165779
        },
        {
          "field": "step-2/83/gradient",
          "max_abs": 0.00030204118229448795
        },
        {
          "field": "step-2/86/gradient",
          "max_abs": 0.0006420603021979332
        },
        {
          "field": "step-2/90/gradient",
          "max_abs": 0.00038238835986703634
        },
        {
          "field": "step-2/94/gradient",
          "max_abs": 0.0005819958169013262
        },
        {
          "field": "step-2/95/gradient",
          "max_abs": 0.00034148678241763264
        },
        {
          "field": "step-2/98/gradient",
          "max_abs": 0.0005733107682317495
        },
        {
          "field": "step-2/102/gradient",
          "max_abs": 0.0013362369500100613
        },
        {
          "field": "step-2/106/gradient",
          "max_abs": 0.0009331400506198406
        },
        {
          "field": "step-2/107/gradient",
          "max_abs": 0.0003070463426411152
        },
        {
          "field": "step-2/183/gradient",
          "max_abs": 0.0006536913570016623
        }
      ],
      "complete_three_step_gate_passed": false
    }
  ],
  "native_full_case_exit0": false,
  "control_optimized_not_started": true,
  "native_against_accepted_prior_reference_passed": false,
  "three_arm_numeric_SFT_accepted": false,
  "native_resume_accepted": false,
  "long_loss_accepted": false,
  "formal_performance_accepted": false,
  "max_points": 0,
  "deadline_epoch": 1793109373,
  "epoch": 1793091974.8066723
}
```

</details>


<a id="appendix-peer"></a>

## 外部参考与研究归因

| 参考内容 | 可用价值 | 边界 |
|---|---|---|
| 同组7B/8B trace | 原始日志、实际shape与算子分析，支持交叉分析 | S4096短采样，与我方S2048正式训练配置不同 |
| 4B分析器 | runtime关联、exclusive归因、并集/重叠计算、未归因保留 | 热点百分比不能直接外推到7B |
| ixSYS资料 | CPU API、GPU kernel、memcpy及同步依赖分析 | 提供PDF较旧；现有环境实查V4.4.0，以实际help与短探针为准 |

同组7B BF16短前向相对RMS约12.98%，未达其3%门槛；4B最终BF16对齐也未通过。8B仅达到相对RMS<3%、top-1≥90%的短前向标准，约1.88%/90.62%，不等于完整训练与恢复正确性。其checkpoint缺少optimizer/RNG，不能替代精确断点恢复。

我方独立核验172项文件哈希、重算四份SFT rank trace，主要热点排序一致；不同选步边界导致累计kernel时长差不超过约0.011%。分析器9项测试通过。7B S4096的GEMM/BMM约占设备kernel累计时间37.20%，copy/cast约24.59%；该分母不是端到端墙钟。

既有采集和实现归原作者；本报告新增贡献为独立核验与适用范围判断。截图的hidden=4096、FFN=16384与8B相符，不作为7B优化目标。

[独立JSON文件](docs/jiuge7b/evidence/peer-review.json)

<details>
<summary>展开完整复核数据</summary>

```json
{
  "scope": "offline independent audit of supplied colleague evidence; no model training or GPU work",
  "source_authorship": "third-party supplied work; not claimed as our experiment",
  "four_SFT_rank_traces_recomputed": [
    {
      "model": "7B",
      "rank": 0,
      "raw_trace_SHA256": "f6d850723c264c72c89ce7c3515b1b238456dc09fe35ad5c5b7e29da24a45935",
      "analyzer_SHA256": "47fc7e21a243d0a02df3c7011fa42825b3fbb01380fd914eaab9e555db18863f",
      "selected_step_names": [
        "ProfilerStep#4",
        "ProfilerStep#5",
        "ProfilerStep#6",
        "ProfilerStep#7"
      ],
      "published_kernel_sum_us": 17608057.323999964,
      "recomputed_kernel_sum_us": 17609869.417,
      "selected_kernel_count": 25468,
      "raw_kernel_count": 25468,
      "selection_status_counts": {
        "correlated": 29664
      },
      "attribution_coverage_percent": 89.77252366072992,
      "top_ops": [
        {
          "op": "aten::mm",
          "cpu_event_category": "cpu_op",
          "kernel_count": 1580,
          "device_self_us": 4931079.515000006,
          "selected_kernel_duration_share_pct": 28.001794892582797
        },
        {
          "op": "aten::copy_",
          "cpu_event_category": "cpu_op",
          "kernel_count": 4196,
          "device_self_us": 2480272.3809999856,
          "selected_kernel_duration_share_pct": 14.08455862032464
        },
        {
          "op": "aten::mul",
          "cpu_event_category": "cpu_op",
          "kernel_count": 6344,
          "device_self_us": 1615954.9659999756,
          "selected_kernel_duration_share_pct": 9.176416518114467
        },
        {
          "op": "aten::masked_fill_",
          "cpu_event_category": "cpu_op",
          "kernel_count": 356,
          "device_self_us": 1250923.1749999986,
          "selected_kernel_duration_share_pct": 7.103534645137106
        },
        {
          "op": "aten::bmm",
          "cpu_event_category": "cpu_op",
          "kernel_count": 672,
          "device_self_us": 835429.4930000001,
          "selected_kernel_duration_share_pct": 4.744098171412353
        },
        {
          "op": "aten::_softmax",
          "cpu_event_category": "cpu_op",
          "kernel_count": 224,
          "device_self_us": 725832.0079999999,
          "selected_kernel_duration_share_pct": 4.1217341867356785
        },
        {
          "op": "aten::div",
          "cpu_event_category": "cpu_op",
          "kernel_count": 568,
          "device_self_us": 604240.5269999999,
          "selected_kernel_duration_share_pct": 3.43126069076177
        }
      ],
      "devices": [
        {
          "device": "0",
          "window_start_us": 1560521239508.596,
          "window_end_us": 1560557083048.4111,
          "host_step_envelope_us": 35843539.81518555,
          "wall_window_with_device_completion_us": 35843539.81518555,
          "device_completion_tail_us": 0.0,
          "kernel_duration_sum_us": 17609869.417,
          "activity_duration_sum_us": 25512270.833,
          "activity_union_us": 25512232.75366211,
          "activity_window_coverage_pct": 71.17665522213169,
          "recognized_compute_union_us": 15430116.25415039,
          "communication_union_us": 1801045.2219238281,
          "recognized_compute_communication_overlap_us": 0.0,
          "communication_not_overlapped_with_recognized_compute_us": 1801045.2219238281,
          "unknown_kernel_union_us": 378707.67236328125
        }
      ]
    },
    {
      "model": "7B",
      "rank": 1,
      "raw_trace_SHA256": "e194af76a4c0ff4d5c5476080c6eb7491b93691716ff8f9d9e20b596fa8b9c0b",
      "analyzer_SHA256": "47fc7e21a243d0a02df3c7011fa42825b3fbb01380fd914eaab9e555db18863f",
      "selected_step_names": [
        "ProfilerStep#4",
        "ProfilerStep#5",
        "ProfilerStep#6",
        "ProfilerStep#7"
      ],
      "published_kernel_sum_us": 17213806.33399996,
      "recomputed_kernel_sum_us": 17213863.843,
      "selected_kernel_count": 25236,
      "raw_kernel_count": 25236,
      "selection_status_counts": {
        "correlated": 29196
      },
      "attribution_coverage_percent": 91.8122987618891,
      "top_ops": [
        {
          "op": "aten::mm",
          "cpu_event_category": "cpu_op",
          "kernel_count": 1580,
          "device_self_us": 4931321.673999998,
          "selected_kernel_duration_share_pct": 28.647383986398356
        },
        {
          "op": "aten::copy_",
          "cpu_event_category": "cpu_op",
          "kernel_count": 4196,
          "device_self_us": 2479230.9189999867,
          "selected_kernel_duration_share_pct": 14.402524277012704
        },
        {
          "op": "aten::mul",
          "cpu_event_category": "cpu_op",
          "kernel_count": 6344,
          "device_self_us": 1616546.726999978,
          "selected_kernel_duration_share_pct": 9.390958019325483
        },
        {
          "op": "aten::masked_fill_",
          "cpu_event_category": "cpu_op",
          "kernel_count": 356,
          "device_self_us": 1250908.3299999994,
          "selected_kernel_duration_share_pct": 7.266865483594957
        },
        {
          "op": "aten::bmm",
          "cpu_event_category": "cpu_op",
          "kernel_count": 672,
          "device_self_us": 836198.6469999993,
          "selected_kernel_duration_share_pct": 4.857704549231919
        },
        {
          "op": "aten::_softmax",
          "cpu_event_category": "cpu_op",
          "kernel_count": 224,
          "device_self_us": 725884.4210000001,
          "selected_kernel_duration_share_pct": 4.216859315377822
        },
        {
          "op": "aten::div",
          "cpu_event_category": "cpu_op",
          "kernel_count": 568,
          "device_self_us": 604162.3180000003,
          "selected_kernel_duration_share_pct": 3.5097426325100294
        }
      ],
      "devices": [
        {
          "device": "1",
          "window_start_us": 1560521239610.146,
          "window_end_us": 1560557083222.593,
          "host_step_envelope_us": 35843612.447021484,
          "wall_window_with_device_completion_us": 35843612.447021484,
          "device_completion_tail_us": 0.0,
          "kernel_duration_sum_us": 17213863.843,
          "activity_duration_sum_us": 25117532.789,
          "activity_union_us": 25117492.545166016,
          "activity_window_coverage_pct": 70.07522632460896,
          "recognized_compute_union_us": 15426357.74243164,
          "communication_union_us": 1409419.7348632812,
          "recognized_compute_communication_overlap_us": 0.0,
          "communication_not_overlapped_with_recognized_compute_us": 1409419.7348632812,
          "unknown_kernel_union_us": 378086.10498046875
        }
      ]
    },
    {
      "model": "8B",
      "rank": 0,
      "raw_trace_SHA256": "39fe3f5e7886171d2eb01ade45e8fb3da68a948e0d6f03712411a97588fbd6a5",
      "analyzer_SHA256": "47fc7e21a243d0a02df3c7011fa42825b3fbb01380fd914eaab9e555db18863f",
      "selected_step_names": [
        "ProfilerStep#4",
        "ProfilerStep#5",
        "ProfilerStep#6",
        "ProfilerStep#7"
      ],
      "published_kernel_sum_us": 21401672.148999963,
      "recomputed_kernel_sum_us": 21403994.006,
      "selected_kernel_count": 30412,
      "raw_kernel_count": 30412,
      "selection_status_counts": {
        "correlated": 34656
      },
      "attribution_coverage_percent": 87.29652586223939,
      "top_ops": [
        {
          "op": "aten::mm",
          "cpu_event_category": "cpu_op",
          "kernel_count": 2060,
          "device_self_us": 5606448.7289999975,
          "selected_kernel_duration_share_pct": 26.193469907664845
        },
        {
          "op": "aten::copy_",
          "cpu_event_category": "cpu_op",
          "kernel_count": 5684,
          "device_self_us": 3319326.08200001,
          "selected_kernel_duration_share_pct": 15.507975198785475
        },
        {
          "op": "aten::mul",
          "cpu_event_category": "cpu_op",
          "kernel_count": 7888,
          "device_self_us": 1976212.8599999542,
          "selected_kernel_duration_share_pct": 9.232916340034384
        },
        {
          "op": "aten::masked_fill_",
          "cpu_event_category": "cpu_op",
          "kernel_count": 404,
          "device_self_us": 1635523.9109999973,
          "selected_kernel_duration_share_pct": 7.641208975023655
        },
        {
          "op": "aten::bmm",
          "cpu_event_category": "cpu_op",
          "kernel_count": 768,
          "device_self_us": 1088172.0789999997,
          "selected_kernel_duration_share_pct": 5.083967406713726
        },
        {
          "op": "aten::_softmax",
          "cpu_event_category": "cpu_op",
          "kernel_count": 256,
          "device_self_us": 947381.3989999995,
          "selected_kernel_duration_share_pct": 4.4261897977285365
        },
        {
          "op": "aten::div",
          "cpu_event_category": "cpu_op",
          "kernel_count": 656,
          "device_self_us": 785784.4930000006,
          "selected_kernel_duration_share_pct": 3.6712049759485463
        }
      ],
      "devices": [
        {
          "device": "0",
          "window_start_us": 1563806409578.872,
          "window_end_us": 1563846662517.842,
          "host_step_envelope_us": 40252938.9699707,
          "wall_window_with_device_completion_us": 40252938.9699707,
          "device_completion_tail_us": 0.0,
          "kernel_duration_sum_us": 21403994.006,
          "activity_duration_sum_us": 30016027.212,
          "activity_union_us": 30015985.205810547,
          "activity_window_coverage_pct": 74.56843145839096,
          "recognized_compute_union_us": 18419535.583007812,
          "communication_union_us": 2719050.8359375,
          "recognized_compute_communication_overlap_us": 0.0,
          "communication_not_overlapped_with_recognized_compute_us": 2719050.8359375,
          "unknown_kernel_union_us": 265407.2666015625
        }
      ]
    },
    {
      "model": "8B",
      "rank": 1,
      "raw_trace_SHA256": "8565f78459ad2e15fc1eae153c4264a75d2506169dae86c84fae29cc42ce830e",
      "analyzer_SHA256": "47fc7e21a243d0a02df3c7011fa42825b3fbb01380fd914eaab9e555db18863f",
      "selected_step_names": [
        "ProfilerStep#4",
        "ProfilerStep#5",
        "ProfilerStep#6",
        "ProfilerStep#7"
      ],
      "published_kernel_sum_us": 20236372.03399996,
      "recomputed_kernel_sum_us": 20236431.311,
      "selected_kernel_count": 30148,
      "raw_kernel_count": 30148,
      "selection_status_counts": {
        "correlated": 34124
      },
      "attribution_coverage_percent": 92.31183603427972,
      "top_ops": [
        {
          "op": "aten::mm",
          "cpu_event_category": "cpu_op",
          "kernel_count": 2060,
          "device_self_us": 5606624.800000004,
          "selected_kernel_duration_share_pct": 27.705600428433186
        },
        {
          "op": "aten::copy_",
          "cpu_event_category": "cpu_op",
          "kernel_count": 5684,
          "device_self_us": 3317624.7650000043,
          "selected_kernel_duration_share_pct": 16.39431732805887
        },
        {
          "op": "aten::mul",
          "cpu_event_category": "cpu_op",
          "kernel_count": 7888,
          "device_self_us": 1977868.0119999566,
          "selected_kernel_duration_share_pct": 9.773798460822677
        },
        {
          "op": "aten::masked_fill_",
          "cpu_event_category": "cpu_op",
          "kernel_count": 404,
          "device_self_us": 1635472.1099999992,
          "selected_kernel_duration_share_pct": 8.08182077593394
        },
        {
          "op": "aten::bmm",
          "cpu_event_category": "cpu_op",
          "kernel_count": 768,
          "device_self_us": 1088246.7479999992,
          "selected_kernel_duration_share_pct": 5.377661363683509
        },
        {
          "op": "aten::_softmax",
          "cpu_event_category": "cpu_op",
          "kernel_count": 256,
          "device_self_us": 947343.7899999998,
          "selected_kernel_duration_share_pct": 4.681377736226882
        },
        {
          "op": "aten::div",
          "cpu_event_category": "cpu_op",
          "kernel_count": 656,
          "device_self_us": 786013.4159999993,
          "selected_kernel_duration_share_pct": 3.8841503421245163
        }
      ],
      "devices": [
        {
          "device": "1",
          "window_start_us": 1563806409803.324,
          "window_end_us": 1563846662869.167,
          "host_step_envelope_us": 40253065.84301758,
          "wall_window_with_device_completion_us": 40253065.84301758,
          "device_completion_tail_us": 0.0,
          "kernel_duration_sum_us": 20236431.311,
          "activity_duration_sum_us": 28727039.21,
          "activity_union_us": 28726996.739257812,
          "activity_window_coverage_pct": 71.36598452224699,
          "recognized_compute_union_us": 18415659.967773438,
          "communication_union_us": 1555810.0095214844,
          "recognized_compute_communication_overlap_us": 0.0,
          "communication_not_overlapped_with_recognized_compute_us": 1555810.0095214844,
          "unknown_kernel_union_us": 264961.00537109375
        }
      ]
    }
  ],
  "artifact_manifest_checks": [
    {
      "model": "7B",
      "checked": 81,
      "manifest_entries": 81,
      "failures": []
    },
    {
      "model": "8B",
      "checked": 91,
      "manifest_entries": 91,
      "failures": []
    }
  ],
  "provided_4B_analyzer_unit_tests_observed": {
    "tests": 9,
    "passed": 9,
    "source_SHA256": "3cf77d3c078e00416ded40e092630692e51892f809de9c47aa18bcd79b8cf7a7"
  },
  "formal_speedup_accepted": false,
  "epoch": 1793108137.140385,
  "max_kernel_sum_difference_percent": 0.010848951352366742,
  "scope_difference": "Recomputation selects runtime launches enclosed by ProfilerStep; supplied 7B/8B analyzer selects device starts in explicit train_step envelope. Small boundary differences are retained, not called exact reproduction.",
  "source_manifest_items_verified": 172,
  "installed_ixsys_proof": "results/peer-ixsys-capability-20261027.json",
  "screenshotted_shapes_match_8B_not_7B": true,
  "next_GPU_target": {
    "task": "7B attention mask/scale/softmax semantic-preserving fusion",
    "shapes": [
      "[1,14,2048,2048]",
      "[1,14,4096,4096]"
    ],
    "status": "proposal only; no kernel or training launched",
    "preserve": [
      "mask semantics",
      "cast boundaries",
      "scale rounding",
      "backward",
      "recomputation",
      "original tolerances"
    ]
  },
  "no_old_reference_gate_removed": true
}
```

</details>


<a id="appendix-provenance"></a>

## 来源与复核范围

模型权重、完整数值张量、检查点及全量原trace继续保留在实验原存储；本文件内嵌的是关键复核结果与SHA，不宣称这些大型原件已上传GitHub。

| 证据 | 原复核文件SHA256 | 字节数 |
|---|---|---:|
| cpt-formal | `b27f39d0e9ce7c4bcb268c5f4697fd85f11beb5b2566d9d41b0cbf8735b5c973` | 8284 |
| sft-formal | `0f2de8dd4c270422c7244044c97d96507a9ba24a67df9ab15760a7411b146495` | 8506 |
| bi-correctness | `3c19a695ede3a6bf8596a948915dee7c64484fd5633211fac6dd5864f60763de` | 7669 |
| bi-trace | `04080c4ebfdd5fc85dc716fa4ac119c1abc0ed1ccc16de808016ded2a45cf652` | 7760 |
| bi-numeric-failure | `212e20a5baa51d6bd6a7f1c6d21fa098a8fd41ad4a12c57ef8eeab35f4f6cbbe` | 10197 |
| operator-validation | `5be93d3c8979a35b39dc69b6384bb89791e94fc2d81f62945aef41d451e17b0e` | 3065 |
| peer-review | `b1386e6084efe04e57081d3cb353687b00ef601dbb3af9b7e420fa1171da1587` | 15049 |

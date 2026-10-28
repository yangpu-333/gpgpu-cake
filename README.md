<a id="report-top"></a>

# 九格7B训练与微调性能优化研究报告

截至2026年10月28日。研究对象为官方九格7B，采用Megatron、TP2、BF16、S2048及CPU AdamW offload，覆盖继续预训练（CPT）与全参数指令微调（SFT）。

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

**独立算子结果：FP32输出口径。** 在现有两张BI-V150顺序完成24个目标用例、76个边界用例及30项负例控制；输出与输入梯度的全部元素对照原生及独立CPU双精度参考，保留原容差并增加相对L2约束。

| S | 输入 | 原生→融合毫秒 | 配对加速比 |
|---|---|---:|---:|
| 2048 | bfloat16 | 11.440→2.752 | **4.16×** |
| 2048 | float32 | 9.399→3.241 | **2.90×** |
| 4096 | bfloat16 | 44.226→9.466 | **4.66×** |
| 4096 | float32 | 36.398→11.485 | **3.16×** |

采用两卡、三种子、15轮ABBA/BAAB配对；24组共360轮均更快，种子间最大CV为2.53%。时间为180次调用合并中位数，加速比为配对几何平均。无profiler计时包含分配、前后向与同步。候选保留严格确定性填充，实际为2次融合计算加2次填充，总共4次GPU启动；原生BF16/FP32分别为13/9次。两卡16份独立trace已核验。算子使用合成输入，不含QK/PV、dropout、重计算及TP通信，不能换算整模型收益。


**真实attention接入：BF16输出边界。** 直接移植全融合版未通过原容差：QK分数相同，FP32概率微差经BF16转换及PV/反向传播后，输出和Q/K/V梯度超限。修正版保留原生softmax规约，融合前后的cast/scale/mask；两卡24组真实MCore attention组件检查通过，覆盖第1/14/28层、层缩放开关及重计算，输出与Q/K/V梯度逐值相同，最大差0。这里使用合成QKV，尚未加载本轮完整7B权重。前述4.16–4.66×不能继承到该保真候选；完整更新、原生恢复与端到端测速仍待验收。


**整模型内存布局已完成容量实测。** 两rank各199组、38.35亿元素的master/梯度和398个moment采样均在Node0，一次真实CPU AdamW更新完成，自有RSS峰值164GiB。依据实测与历史整模型峰值，启动预算改为Node0空闲291GiB、Node1空闲28GiB，运行保护和数值容差保留。按最新授权已直接启动真实7B，当前native初始化和全状态审计进行中，运行16GiB余量等保护保留；三臂连续更新、原生恢复和端到端收益仍未验收。

## 下一阶段目标

| 优先级 | 研究任务 | 交付与验收 |
|---|---|---|
| 1 | 定位跨独立进程的embedding及连续更新差异；控制新增审计文件缓存 | 可复现的首个分歧证据；缓存写回前后内容与驻留验证，保留原数值标准 |
| 2 | 统一现有trace的算子、shape、搬运及同步归因；必要时用ixSYS补充短时间线 | CPU offload关键路径与GPU热点表，明确统计分母和未归因项 |
| 3 | 保真融合已通过MCore组件检查；继续加载完整7B，验证连续更新与断点恢复 | 原三臂全状态与原生恢复门禁通过后，才做端到端计时 |
| 4 | 完成三臂全状态、SFT语义、原生保存恢复，再做正式对照 | 无profiler的step延迟、有效输入tokens/s、答案targets/s，沿用原三种子协议 |

同组提供的7B/8B/4B原始trace、分析器及ixSYS资料用于交叉分析，既有工作归原作者。已独立核验172项文件哈希、复算四份SFT trace并通过分析器9项测试；7B/4B的BF16对齐未通过，8B仅通过短前向检查，不能直接继承其训练正确性或性能结论。截图中的8B GEMM形状不作为7B目标。

[真实attention接入](#appendix-attention-integration) · [融合实测与代码](#appendix-fusion) · [算子基准复核](#appendix-operator) · [详细计划与验收标准](#appendix-plan) · [CPT正式结果](#appendix-cpt) · [SFT正式结果](#appendix-sft)

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
| P2 单一候选 | 已完成scale/mask/softmax隔离融合；下一步接入真实attention，核对重计算、反向和mask调用点；端到端继续研究offload回传与同步 | 先做接入前后逐层输出/梯度比较，再进入模型完整验收 |
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


<a id="appendix-fusion"></a>

## BI-V150融合前后向：正确性与配对测速

| S | 输入dtype | 原生→融合毫秒 | 配对加速比 | 总GPU启动：原生→融合 |
|---|---|---:|---:|---:|
| 2048 | bfloat16 | 11.440→2.752 | 4.156× | 13→4 |
| 2048 | float32 | 9.399→3.241 | 2.903× | 9→4 |
| 4096 | bfloat16 | 44.226→9.466 | 4.663× | 13→4 |
| 4096 | float32 | 36.398→11.485 | 3.164× | 9→4 |

两卡16份独立trace全部闭合，并独立核对原始及压缩SHA。

融合前向在一行内完成FP32缩放、有限值-10000遮罩、max/exp/sum归一化；融合反向完成softmax导数、遮罩梯度归零、缩放及输入dtype转换。未生成整块缩放与遮罩中间矩阵，保留原生FP32舍入边界。

两卡×三种子×S2048/S4096×BF16/FP32，共24个目标用例；另有76个边界用例、30项负例控制。输出与输入梯度全部元素对照原生实现及独立CPU float64参考，原容差`atol=3e-4, rtol=1e-3`不变，并增加前向/FP32相对L2≤1e-5、BF16梯度≤1e-3约束。全遮罩行保留有限填充值对应的均匀softmax语义，输入梯度为0。

计时使用同卡同输入的15轮交替ABBA/BAAB，每臂每组30次同步调用、5对预热；包含dispatch、输出分配、前后向及GPU同步，输入/mask预先建立。表内时间是两卡三种子的180次调用合并中位数，加速比为逐轮配对比的几何平均，二者口径不同。全部360轮更快，种子中位数最大CV为2.53%，未触发10%完整确认阈值。Profiler在全部计时结束后独立运行。

严格确定性及未初始化内存填充均保留。候选实际为2次融合数学kernel加2次保护性填充，总共4次启动。最初“总计2次”的trace断言失败已保留；后续版本化trace观察器，严格核对4个事件的类型与顺序。补trace首次重试被32GiB保守启动线拒绝；用户明确要求实际尝试后，仅该启动线改为16GiB，两卡补采成功。数学实现、原数值/实时内存保护及已完成计时均未变。

S4096的独立trace中，融合BF16前向、反向和保护性填充的两卡平均设备时间分别约2.426、4.308、2.107毫秒；FP32约3.263、4.810、2.804毫秒。反向是候选的主要数学耗时，填充约占kernel累计时间24%–26%。这些是设备事件归因，未混入上述无profiler配对加速比；现有证据没有测得硬件带宽饱和度。

该结果限于合成输入的独立attention子链，未包含QK/PV矩阵乘、dropout、重计算、TP通信及整模型更新。不能将算子加速比称为7B训练加速。下一步为真实attention接入与逐层前反向对齐，再过三臂全状态、SFT及原生恢复门禁。

[融合实现](experiments/jiuge_training/softmax_fusion_20261028_v1/fused_scale_mask_softmax.py) · [完整CPU参考](experiments/jiuge_training/softmax_fusion_20261028_v1/full_reference.py) · [冻结实验契约](experiments/jiuge_training/softmax_fusion_20261028_v1/contract.json)

算法参考：[Triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)；保护性填充：[PyTorch 2.7确定性文档](https://docs.pytorch.org/docs/2.7/deterministic.html)。本结果来自实际CoreX 4.4构建。

[独立JSON文件](docs/jiuge7b/evidence/softmax-fusion.json)

<details>
<summary>展开完整复核数据</summary>

```json
{
 "date": "2026-10-28",
 "operator_correctness_accepted": true,
 "operator_paired_speedup_accepted": true,
 "full_model_training_accepted": false,
 "TP2_communication_validated": false,
 "scope": "synthetic-input isolated scale/mask/softmax forward and backward; no QK/PV, optimizer or real model integration",
 "GPU": "Iluvatar BI-V150 32GiB; CoreX4.4/Torch2.7.1/Triton3.1 installed vendor build",
 "atol": 0.0003,
 "rtol": 0.001,
 "tolerances_not_relaxed": true,
 "full_target_cases": 24,
 "boundary_cases": 76,
 "negative_controls_rejected": 30,
 "checked_target_elements_per_forward_or_backward": 3523215360,
 "max_errors": {
  "forward_vs_native": 2.9802322387695312e-08,
  "backward_vs_native": 3.0517578125e-05,
  "native_forward_vs_CPU64": 6.186074019698395e-08,
  "candidate_forward_vs_CPU64": 6.186074019698395e-08,
  "native_backward_vs_CPU64": 3.0517578125e-05,
  "candidate_backward_vs_CPU64": 1.52587890625e-05
 },
 "confirmation_required": false,
 "all_paired_groups_meet_gain_gate": true,
 "source_end_SHA_verified": true,
 "key_evidence_SHA256": "111b6f447430c4fcd1134566c36c381b3f148b0b87217bb10ea227aad6da8aef",
 "all_own_workers_ended": true,
 "post_selected_GPUs_idle": true,
 "independently_recomputed_max_seed_CV": {
  "primary": 0.025259404871000018
 },
 "schema": "jiuge7b-scale-mask-softmax-public-review-v1",
 "summary": [
  {
   "S": 2048,
   "dtype": "torch.bfloat16",
   "native_pooled_median_ms": 11.43967965617776,
   "candidate_pooled_median_ms": 2.7515841647982597,
   "paired_geomean_ratio": 4.155712281097924,
   "paired_latency_reduction_percent": 75.93673641583763,
   "native_total_GPU_launches": 13,
   "candidate_total_GPU_launches": 4,
   "candidate_fused_math_launches": 2,
   "candidate_protected_fill_launches": 2,
   "cells": 6,
   "all90_rounds_faster": true
  },
  {
   "S": 2048,
   "dtype": "torch.float32",
   "native_pooled_median_ms": 9.399170987308025,
   "candidate_pooled_median_ms": 3.240808844566345,
   "paired_geomean_ratio": 2.9029920780925598,
   "paired_latency_reduction_percent": 65.55278233287292,
   "native_total_GPU_launches": 9,
   "candidate_total_GPU_launches": 4,
   "candidate_fused_math_launches": 2,
   "candidate_protected_fill_launches": 2,
   "cells": 6,
   "all90_rounds_faster": true
  },
  {
   "S": 4096,
   "dtype": "torch.bfloat16",
   "native_pooled_median_ms": 44.22613186761737,
   "candidate_pooled_median_ms": 9.46616055443883,
   "paired_geomean_ratio": 4.663130116034761,
   "paired_latency_reduction_percent": 78.55517699235168,
   "native_total_GPU_launches": 13,
   "candidate_total_GPU_launches": 4,
   "candidate_fused_math_launches": 2,
   "candidate_protected_fill_launches": 2,
   "cells": 6,
   "all90_rounds_faster": true
  },
  {
   "S": 4096,
   "dtype": "torch.float32",
   "native_pooled_median_ms": 36.39789577573538,
   "candidate_pooled_median_ms": 11.484850663691759,
   "paired_geomean_ratio": 3.163602394507043,
   "paired_latency_reduction_percent": 68.39046519447898,
   "native_total_GPU_launches": 9,
   "candidate_total_GPU_launches": 4,
   "candidate_fused_math_launches": 2,
   "candidate_protected_fill_launches": 2,
   "cells": 6,
   "all90_rounds_faster": true
  }
 ],
 "paired_timing_statistical_gate_passed": true,
 "paired_groups": 24,
 "ABBA_BAAB_rounds": 360,
 "timing_calls_per_arm": 720,
 "unprofiled_primary_measurement_workers": 6,
 "full_trace_target": 16,
 "closed_trace_files": 16,
 "all_expected_traces_passed": true,
 "source_SHA256": {
  "corex_process_guard_data.py": "a8f86db9e75eb353121c346011428400c4d4d1208e35a5d6c12ce48009dfd058",
  "full_reference.py": "7043fab97c80736e68c4dd343956383964bdb4fef3a053b4bf4901ada8c3873e",
  "fused_scale_mask_softmax.py": "093e3a90fc652b7652938ced212107ada5861bceaccbe8d8bdcc008b29e6cd58",
  "fusion_worker.py": "4d702947378763a70cd32c5f64e58ea20ab6785cf38b3b3cc7ea281c22d33b40",
  "supervise_fusion.py": "78fb06879af2025c40fc75dad5bd0f9a6263a2d1e6dd4951f7e6214c8f3b090d"
 },
 "numerical_reference": "All elements streamed and compared on remote CPU float64; local review verifies coverage/source/SHA and recomputes all paired statistics. Raw numerical values are not downloaded.",
 "strict_deterministic_algorithms": true,
 "fill_uninitialized_memory": true,
 "trace_accounting": {
  "initial_two_total_launch_assertion_failed": true,
  "actual_candidate_total": 4,
  "math_launches": 2,
  "protected_fill_launches": 2,
  "math_and_numeric_and_live_guards_unchanged": true,
  "trace_only_initial_gate_revised_by_explicit_user_authorization": true,
  "original_failure_preserved": true,
  "observer_versioned_with_strict_event_order": true,
  "original_correctness_and_timing_not_rerun": true
 },
 "traces": [
  {
   "device": 0,
   "S": 2048,
   "dtype": "torch.bfloat16",
   "arm": "native",
   "kernel_count": 13,
   "raw_SHA256": "4f14af331ffd9554e9b5155dc5bf93b495280ba13b7154bac8df8cf9f9272c9b",
   "gzip_SHA256": "66c62c800be603e79abd138d93d386640472b5a8c51571c7c06de965be848fc2",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 348.961
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&, bool)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&, bool)::{lambda(int)#1})",
     "duration_us": 648.084
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 821.123
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 341.592
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 936.646
    },
    {
     "name": "void (anonymous namespace)::softmax_warp_forward<float, float, float, 11, false, false>(float*, float const*, int, int, int, bool const*, int, bool)",
     "duration_us": 827.638
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)",
     "duration_us": 1204.926
    },
    {
     "name": "void at::native::(anonymous namespace)::cunn_SoftMaxBackwardSmem<1, float, float, float, at::native::(anonymous namespace)::SoftMaxBackwardEpilogue>(float*, float const*, float const*, long)",
     "duration_us": 1172.953
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 346.115
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 934.115
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 817.888
    },
    {
     "name": "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)",
     "duration_us": 175.93
    },
    {
     "name": "void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)",
     "duration_us": 819.6
    }
   ]
  },
  {
   "device": 0,
   "S": 2048,
   "dtype": "torch.bfloat16",
   "arm": "candidate",
   "kernel_count": 4,
   "raw_SHA256": "0019318436149b3daf5fa33f368d87134ce6c0fe271b509a18766fd250e371b1",
   "gzip_SHA256": "4510dab919166ae54a9d54f3a2793f4a3183ca884203b9521f48566924505473",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 344.1
    },
    {
     "name": "_forward",
     "duration_us": 602.719
    },
    {
     "name": "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)",
     "duration_us": 176.426
    },
    {
     "name": "_backward",
     "duration_us": 1079.292
    }
   ]
  },
  {
   "device": 0,
   "S": 2048,
   "dtype": "torch.float32",
   "arm": "native",
   "kernel_count": 9,
   "raw_SHA256": "4a6900f350aeab260af37d4ecbc19746ef67b0da182525a05efe9abb3acb275c",
   "gzip_SHA256": "0a1dc6915f79a2869169e182dee023877c4f4499bf5f36c13f7c51d9a635d445",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 830.765
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 346.026
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 935.615
    },
    {
     "name": "void (anonymous namespace)::softmax_warp_forward<float, float, float, 11, false, false>(float*, float const*, int, int, int, bool const*, int, bool)",
     "duration_us": 828.43
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)",
     "duration_us": 1202.726
    },
    {
     "name": "void at::native::(anonymous namespace)::cunn_SoftMaxBackwardSmem<1, float, float, float, at::native::(anonymous namespace)::SoftMaxBackwardEpilogue>(float*, float const*, float const*, long)",
     "duration_us": 1171.176
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 346.307
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 933.388
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 820.673
    }
   ]
  },
  {
   "device": 0,
   "S": 2048,
   "dtype": "torch.float32",
   "arm": "candidate",
   "kernel_count": 4,
   "raw_SHA256": "aa0dd4947a752b734af28b1fc9fcce5ac21b4ae739690c3c1a73a09c6f46439c",
   "gzip_SHA256": "11fe5a060b5c05bf3e7e229dadca3eec190ffaccd284a4151caa2724cca45a19",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 345.619
    },
    {
     "name": "_forward",
     "duration_us": 817.046
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 345.965
    },
    {
     "name": "_backward",
     "duration_us": 1203.488
    }
   ]
  },
  {
   "device": 0,
   "S": 4096,
   "dtype": "torch.bfloat16",
   "arm": "native",
   "kernel_count": 13,
   "raw_SHA256": "a22d39f71432f421defcf9a9411ba1cc5f2182a312a372dd2177befa6415aa68",
   "gzip_SHA256": "24b0a754b2a0b3ecc66b1e5685c03942c48a7a0203e962b691e28f3c2e1a145a",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1385.4
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&, bool)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&, bool)::{lambda(int)#1})",
     "duration_us": 2532.834
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 3296.273
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1401.192
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 3718.65
    },
    {
     "name": "void at::native::(anonymous namespace)::cunn_SoftMaxForwardReg<float, float, float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue, long, 4>(float*, float const*, long)",
     "duration_us": 3236.484
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)",
     "duration_us": 4788.469
    },
    {
     "name": "void at::native::(anonymous namespace)::cunn_SoftMaxBackwardSmem<1, float, float, float, at::native::(anonymous namespace)::SoftMaxBackwardEpilogue>(float*, float const*, float const*, long)",
     "duration_us": 4748.08
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1381.288
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 3722.038
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 3282.965
    },
    {
     "name": "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)",
     "duration_us": 720.942
    },
    {
     "name": "void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)",
     "duration_us": 3203.334
    }
   ]
  },
  {
   "device": 0,
   "S": 4096,
   "dtype": "torch.bfloat16",
   "arm": "candidate",
   "kernel_count": 4,
   "raw_SHA256": "f859c22c7fc1eba173635e04da88e8869b8647837ef0d6ef3636899a8b2649b5",
   "gzip_SHA256": "64512f73f885c244206e6c05edc62038f70b9a44b22fac9f9adfed8e6d7cdbf9",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1401.019
    },
    {
     "name": "_forward",
     "duration_us": 2426.776
    },
    {
     "name": "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)",
     "duration_us": 706.157
    },
    {
     "name": "_backward",
     "duration_us": 4303.903
    }
   ]
  },
  {
   "device": 0,
   "S": 4096,
   "dtype": "torch.float32",
   "arm": "native",
   "kernel_count": 9,
   "raw_SHA256": "ad14b0cf6bc6582ee1d550a0ee310526ec18ad6a85bf33dfc71fe7bf1b0114ae",
   "gzip_SHA256": "dccd1cdb69bd6dd1333fef920463b0c6b57b28581f46cc5ba50ca0de748bef11",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 3252.815
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1404.561
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 3715.376
    },
    {
     "name": "void at::native::(anonymous namespace)::cunn_SoftMaxForwardReg<float, float, float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue, long, 4>(float*, float const*, long)",
     "duration_us": 3239.376
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)",
     "duration_us": 4797.277
    },
    {
     "name": "void at::native::(anonymous namespace)::cunn_SoftMaxBackwardSmem<1, float, float, float, at::native::(anonymous namespace)::SoftMaxBackwardEpilogue>(float*, float const*, float const*, long)",
     "duration_us": 4759.865
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1398.838
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 3715.411
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 3271.65
    }
   ]
  },
  {
   "device": 0,
   "S": 4096,
   "dtype": "torch.float32",
   "arm": "candidate",
   "kernel_count": 4,
   "raw_SHA256": "89c713e78e566cf9bd17aac0171589039c5edbe2bc38279b0c78f0ccc20e847a",
   "gzip_SHA256": "573a5eae814d3e1ca7544c6ab042787a452d9ff30c117ed40112416615648919",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1402.219
    },
    {
     "name": "_forward",
     "duration_us": 3255.173
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1403.853
    },
    {
     "name": "_backward",
     "duration_us": 4813.692
    }
   ]
  },
  {
   "device": 1,
   "S": 2048,
   "dtype": "torch.bfloat16",
   "arm": "native",
   "kernel_count": 13,
   "raw_SHA256": "2ec3bd2527ec42afe256a0b919888585a6b6088217cb92b6079d6e2a37c034b2",
   "gzip_SHA256": "e664a25e867674a198187567685b4811eecd02296cdd52f43718cbfc6b096565",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 348.084
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&, bool)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&, bool)::{lambda(int)#1})",
     "duration_us": 647.165
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 821.25
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 346.265
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 933.576
    },
    {
     "name": "void (anonymous namespace)::softmax_warp_forward<float, float, float, 11, false, false>(float*, float const*, int, int, int, bool const*, int, bool)",
     "duration_us": 831.038
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)",
     "duration_us": 1208.334
    },
    {
     "name": "void at::native::(anonymous namespace)::cunn_SoftMaxBackwardSmem<1, float, float, float, at::native::(anonymous namespace)::SoftMaxBackwardEpilogue>(float*, float const*, float const*, long)",
     "duration_us": 1176.576
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 343.003
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 931.542
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 816.75
    },
    {
     "name": "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)",
     "duration_us": 173.1
    },
    {
     "name": "void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)",
     "duration_us": 815.957
    }
   ]
  },
  {
   "device": 1,
   "S": 2048,
   "dtype": "torch.bfloat16",
   "arm": "candidate",
   "kernel_count": 4,
   "raw_SHA256": "155c245a365fa64712f95f603227db0cd2b06235da9bafb8e234afd9f3ea3ec2",
   "gzip_SHA256": "a47c640be1c6d30f92de189ab55ce9764f7f697ef776f403beadd10126a76339",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 345.815
    },
    {
     "name": "_forward",
     "duration_us": 604.519
    },
    {
     "name": "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)",
     "duration_us": 175.523
    },
    {
     "name": "_backward",
     "duration_us": 1079.207
    }
   ]
  },
  {
   "device": 1,
   "S": 2048,
   "dtype": "torch.float32",
   "arm": "native",
   "kernel_count": 9,
   "raw_SHA256": "e1760a2c52aa95cbef12a280c7522956a7ff8c339a7e55f6a025f5c0269d7100",
   "gzip_SHA256": "b02aca53070af7f84ee8887f9ad29b3998d6754aa20a5e6a21944d9bf9d78d06",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 822.323
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 340.888
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 934.992
    },
    {
     "name": "void (anonymous namespace)::softmax_warp_forward<float, float, float, 11, false, false>(float*, float const*, int, int, int, bool const*, int, bool)",
     "duration_us": 825.988
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)",
     "duration_us": 1202.569
    },
    {
     "name": "void at::native::(anonymous namespace)::cunn_SoftMaxBackwardSmem<1, float, float, float, at::native::(anonymous namespace)::SoftMaxBackwardEpilogue>(float*, float const*, float const*, long)",
     "duration_us": 1173.276
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 346.284
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 934.157
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 820.519
    }
   ]
  },
  {
   "device": 1,
   "S": 2048,
   "dtype": "torch.float32",
   "arm": "candidate",
   "kernel_count": 4,
   "raw_SHA256": "add54d14cea393f5df672d96f2aa9f0d86ae87eb03460b50046c80fc81e6b83b",
   "gzip_SHA256": "30bcb162f2fc7ab295e2fdddce9d17aef71ed23fe8f15346b172538cddf6905b",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 337.476
    },
    {
     "name": "_forward",
     "duration_us": 812.976
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 346.753
    },
    {
     "name": "_backward",
     "duration_us": 1206.261
    }
   ]
  },
  {
   "device": 1,
   "S": 4096,
   "dtype": "torch.bfloat16",
   "arm": "native",
   "kernel_count": 13,
   "raw_SHA256": "485d8caec4ea6943f9394427f1d13125e3728a086ef0152e98aebd8332939190",
   "gzip_SHA256": "49b5ee4eeba3adce3e844d55137b0e4a17903a2432ae97ca898c3e0e8b2395c0",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1388.311
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&, bool)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&, bool)::{lambda(int)#1})",
     "duration_us": 2531.573
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 3273.492
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1400.123
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 3714.407
    },
    {
     "name": "void at::native::(anonymous namespace)::cunn_SoftMaxForwardReg<float, float, float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue, long, 4>(float*, float const*, long)",
     "duration_us": 3245.55
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)",
     "duration_us": 4792.157
    },
    {
     "name": "void at::native::(anonymous namespace)::cunn_SoftMaxBackwardSmem<1, float, float, float, at::native::(anonymous namespace)::SoftMaxBackwardEpilogue>(float*, float const*, float const*, long)",
     "duration_us": 4742.592
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1399.565
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 3720.496
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 3287.7
    },
    {
     "name": "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)",
     "duration_us": 714.857
    },
    {
     "name": "void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)",
     "duration_us": 3218.011
    }
   ]
  },
  {
   "device": 1,
   "S": 4096,
   "dtype": "torch.bfloat16",
   "arm": "candidate",
   "kernel_count": 4,
   "raw_SHA256": "ec3ada23383c8e6772c5333d6ce109b0c5796ca4e681d059a2c11c64fbfcb529",
   "gzip_SHA256": "9690b042282b0317947a506853f42d1c8e31398b28b5e8222189a915e4afbb82",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1400.146
    },
    {
     "name": "_forward",
     "duration_us": 2425.069
    },
    {
     "name": "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)",
     "duration_us": 705.942
    },
    {
     "name": "_backward",
     "duration_us": 4311.234
    }
   ]
  },
  {
   "device": 1,
   "S": 4096,
   "dtype": "torch.float32",
   "arm": "native",
   "kernel_count": 9,
   "raw_SHA256": "df3e8129133cb76b25c89f8b3bcc986c24621a747bdad33d91b6d3a3af3818e4",
   "gzip_SHA256": "230985607251b0fe970f7a3d666267bde4915e499f6d6d17570e162e7b5e898b",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 3270.403
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1386.688
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 3718.973
    },
    {
     "name": "void at::native::(anonymous namespace)::cunn_SoftMaxForwardReg<float, float, float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue, long, 4>(float*, float const*, long)",
     "duration_us": 3244.5
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)",
     "duration_us": 4783.584
    },
    {
     "name": "void at::native::(anonymous namespace)::cunn_SoftMaxBackwardSmem<1, float, float, float, at::native::(anonymous namespace)::SoftMaxBackwardEpilogue>(float*, float const*, float const*, long)",
     "duration_us": 4754.573
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1399.284
    },
    {
     "name": "void at::native::elementwise_kernel<512, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&, bool)::{lambda(int)#5})",
     "duration_us": 3716.85
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)",
     "duration_us": 3280.007
    }
   ]
  },
  {
   "device": 1,
   "S": 4096,
   "dtype": "torch.float32",
   "arm": "candidate",
   "kernel_count": 4,
   "raw_SHA256": "641053bb6dbaba62d252e0ad285df335d0a557bc8c7839349ec28ac702b1366f",
   "gzip_SHA256": "25ecaea97b38ac156c6ff3bd0fffffefa0b44d0179e9423e5c12f0ad0007c8e6",
   "kernels": [
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1398.0
    },
    {
     "name": "_forward",
     "duration_us": 3270.342
    },
    {
     "name": "void at::native::modern::elementwise_kernel<at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)",
     "duration_us": 1403.676
    },
    {
     "name": "_backward",
     "duration_us": 4805.4
    }
   ]
  }
 ],
 "trace_only_resource_gate_revision": {
  "scope": "trace-only unchanged isolated operator; no correctness/timing rerun or full-model run",
  "previous_conservative_initial_Node1_GiB": 32,
  "user_authorized_actual_attempt": true,
  "initial_and_runtime_Node1_minimum_GiB": 16,
  "runtime_both_NUMA_minimum_GiB": 16,
  "own_RSS_cap_GiB": 8,
  "RLIMIT_DATA_per_rank_GiB": 144,
  "host_available_min_GiB": 8,
  "poll_seconds": 0.5,
  "full_model_Node1_80GiB_gate_unchanged": true,
  "numeric_tolerances_unchanged": true,
  "no_other_task_memory_or_process_touched": true
 },
 "trace_device_time_attribution": [
  {
   "S": 2048,
   "dtype": "torch.bfloat16",
   "fused_forward_ms": 0.603619,
   "fused_backward_ms": 1.0792495,
   "protected_fills_ms": 0.5209320000000001,
   "fill_fraction_of_kernel_duration_sum": 0.23637892812893,
   "scope": "Mean of two separately profiled devices; GPU kernel duration sum, not wall-clock paired latency; not used for speedup"
  },
  {
   "S": 2048,
   "dtype": "torch.float32",
   "fused_forward_ms": 0.8150109999999999,
   "fused_backward_ms": 1.2048744999999998,
   "protected_fills_ms": 0.6879065000000001,
   "fill_fraction_of_kernel_duration_sum": 0.2540470242913784,
   "scope": "Mean of two separately profiled devices; GPU kernel duration sum, not wall-clock paired latency; not used for speedup"
  },
  {
   "S": 4096,
   "dtype": "torch.bfloat16",
   "fused_forward_ms": 2.4259224999999995,
   "fused_backward_ms": 4.3075685,
   "protected_fills_ms": 2.1066319999999994,
   "fill_fraction_of_kernel_duration_sum": 0.23830347156934353,
   "scope": "Mean of two separately profiled devices; GPU kernel duration sum, not wall-clock paired latency; not used for speedup"
  },
  {
   "S": 4096,
   "dtype": "torch.float32",
   "fused_forward_ms": 3.2627574999999998,
   "fused_backward_ms": 4.809546,
   "protected_fills_ms": 2.803874,
   "fill_fraction_of_kernel_duration_sum": 0.25779958078102344,
   "scope": "Mean of two separately profiled devices; GPU kernel duration sum, not wall-clock paired latency; not used for speedup"
  }
 ],
 "correctness_case_columns": [
  "device",
  "kind",
  "shape",
  "dtype",
  "seed",
  "mask_mode",
  "scale",
  "pattern",
  "noncontiguous_upstream",
  "metrics",
  "raw_checked_tensor_SHA256",
  "max_probability_row_sum_error",
  "masked_input_gradient_nonzero",
  "passed"
 ],
 "metric_names": [
  "forward_vs_native",
  "backward_vs_native",
  "native_forward_vs_CPU64",
  "candidate_forward_vs_CPU64",
  "native_backward_vs_CPU64",
  "candidate_backward_vs_CPU64"
 ],
 "metric_value_columns": [
  "elements",
  "nonfinite",
  "bad_original_tolerance",
  "max_abs",
  "relative_L2",
  "additional_relative_L2_limit",
  "passed"
 ],
 "correctness_details": [
  [
   0,
   "target",
   [
    1,
    14,
    2048,
    2048
   ],
   "torch.bfloat16",
   63100,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     58720256,
     false,
     0,
     2.9802322387695312e-08,
     4.672940860860489e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     1.9073486328125e-06,
     7.414175452977063e-06,
     0.001,
     true
    ],
    [
     58720256,
     false,
     0,
     5.4706867369525014e-08,
     5.328086906700259e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     5.4706867369525014e-08,
     5.3041651406504105e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     1.52587890625e-05,
     2.0788746203636328e-05,
     0.001,
     true
    ],
    [
     58720256,
     false,
     0,
     1.52587890625e-05,
     2.101007929912348e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "d2eae38d4762531df6d369e8175fec9ed4ee340a38966281b03e46af5958b930",
    "upstream": "efc7cd1fc02f55761fc3903a8e482513b95991d46bade8f91fa90570b325955a",
    "native_forward": "85a5b825fdbaae4cdd4fd097dc5e6e4600f42129f8a785ffb58137c72be3ef4c",
    "native_backward": "02c62f2a64bd3e750a4ec72f61b82d8a5a0ade6fed581bb69f5d414c1109cbaa",
    "candidate_forward": "0f8ad05b3ee327d7640e835d79d7a09a93053b41fc1429318a9e756fb989ba65",
    "candidate_backward": "ac5ecd61424fefeadbb412299ccb5c6b8f57ffdedd55f3d9889eee9ba87e65b5"
   },
   1.4045508578419685e-07,
   0,
   true
  ],
  [
   0,
   "target",
   [
    1,
    14,
    2048,
    2048
   ],
   "torch.float32",
   63100,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     58720256,
     false,
     0,
     3.725290298461914e-09,
     3.58600562879739e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     9.313225746154785e-10,
     4.650116359079598e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.435542999559061e-08,
     5.2547896673704584e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.435542999559061e-08,
     5.232624191569617e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     6.634688409778988e-09,
     7.437098336823825e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     6.634688409778988e-09,
     7.418626818409743e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "e631f015fc1bd6ffda0171a4dfc093bfa6c2d0c210e4b6749b98ad3c11fab34a",
    "upstream": "efc7cd1fc02f55761fc3903a8e482513b95991d46bade8f91fa90570b325955a",
    "native_forward": "c8a91b9a5950d7eaea1fc3bf80680cb7ce514f7bd1d82c9fd6af1b159395a839",
    "native_backward": "1d60501c542b610d42db5c5886f79ed38c040ed7949f8f60e31fb18056a91487",
    "candidate_forward": "f2d94466b542bafbb6c097a44b83d4a41f3a83ac7828b6e0bc578cc5a1f89890",
    "candidate_backward": "7b13b588dde70dde61acaf4d960d9cd5f3894ad68946dcb80f61067e50632c0b"
   },
   1.50932464748621e-07,
   0,
   true
  ],
  [
   0,
   "target",
   [
    1,
    14,
    4096,
    4096
   ],
   "torch.bfloat16",
   63100,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     234881024,
     false,
     0,
     2.9802322387695312e-08,
     4.7239326375562115e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     7.62939453125e-06,
     1.3179844121984294e-05,
     0.001,
     true
    ],
    [
     234881024,
     false,
     0,
     4.576568635172862e-08,
     5.317207529072065e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.576568635172862e-08,
     5.2909979461476154e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     7.62939453125e-06,
     1.701035920727131e-05,
     0.001,
     true
    ],
    [
     234881024,
     false,
     0,
     3.814697265625e-06,
     1.3434721588220862e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "1822427368be84c7a78249e475a78d4c5011ca0e53979d7466c95b1ac24036f0",
    "upstream": "2b4b8ac25d43a4a6f3e6cd5960ef3e1b4516771b1ad202088bc709a6f2da30a6",
    "native_forward": "9e4b04642a34e116ac6519a9b1b501e774adca0f83f56b804d4c4b81b97cb21e",
    "native_backward": "6f835c86501027243181f1109b356a6c5e92d10ba1e612c55ecae68db0b63e98",
    "candidate_forward": "28720cb3990e2aa4628c496d1ad542682caa4292b2baa484e86a01c2686879e2",
    "candidate_backward": "cc59f7248edf13746d5e2a57bdc5044223ef89f13b6373f83d7ede7d75b6b271"
   },
   1.57160684466362e-07,
   0,
   true
  ],
  [
   0,
   "target",
   [
    1,
    14,
    4096,
    4096
   ],
   "torch.float32",
   63100,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     234881024,
     false,
     0,
     4.656612873077393e-10,
     2.0939859992853893e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     1.7462298274040222e-10,
     2.6826976411967284e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     6.186074019698395e-08,
     5.273726372811987e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     6.186074019698395e-08,
     5.272045513197869e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.464342824236911e-09,
     7.191454206759172e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.464342824236911e-09,
     7.190400214116717e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "9710d4c04d3e81a082a128a46bc2b2c860d240e5a2d9156968538ad7abb17e53",
    "upstream": "2b4b8ac25d43a4a6f3e6cd5960ef3e1b4516771b1ad202088bc709a6f2da30a6",
    "native_forward": "4451f6cf902d0da1d04817474f583d308356c95474874edf0c82b6cd0ff3c05b",
    "native_backward": "0a68326e0f1f180f5d18bd345c60d5094aace15cdc36265ab05fb17c75b37a24",
    "candidate_forward": "2f2430c78faca6fd76dd0c337acad2bb63ca00bc7c15d9897482cefc2213937a",
    "candidate_backward": "51cae9345597de2aef41effe12b9549c4a9c6f979ce6b9cb96dc616bc71ca492"
   },
   1.655425876379013e-07,
   0,
   true
  ],
  [
   0,
   "target",
   [
    1,
    14,
    2048,
    2048
   ],
   "torch.bfloat16",
   63101,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     58720256,
     false,
     0,
     2.9802322387695312e-08,
     4.489604071360259e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     3.814697265625e-06,
     1.05530321369602e-05,
     0.001,
     true
    ],
    [
     58720256,
     false,
     0,
     4.2496641039502236e-08,
     5.2356928144230984e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.2496641039502236e-08,
     5.169549203664316e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     7.62939453125e-06,
     1.4771655151334465e-05,
     0.001,
     true
    ],
    [
     58720256,
     false,
     0,
     7.62939453125e-06,
     1.5236406112271593e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "78ffa1714ca3bf56d8f7f8b434bd3be0af6ba9e05161498cc32aad3a87556796",
    "upstream": "584c8c1ea7d89631486e5cae5c6bc5b573682c6162a291e594e86af3cc34f71b",
    "native_forward": "d310edb1c205953b66dfaa90155546d2d39a17880eeb114f205d19303547da70",
    "native_backward": "0c2d2f7a86f0c36637999e51886937accda9335f0e72092ab6e9961e0aea581f",
    "candidate_forward": "6dc5de53b0c01a5e2644bc875591ee700312f710221ad798b835b41b6c2f3fb2",
    "candidate_backward": "b9acd6b84b70f3df571f152916803a0ac380a0fd98b0a2470ccd07090f2537d9"
   },
   1.3748649507761002e-07,
   0,
   true
  ],
  [
   0,
   "target",
   [
    1,
    14,
    2048,
    2048
   ],
   "torch.float32",
   63101,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     58720256,
     false,
     0,
     2.7939677238464355e-09,
     3.6081517211926234e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     1.1641532182693481e-09,
     4.752618949204581e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.1549625184877925e-08,
     5.249437721437406e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.1549625184877925e-08,
     5.2190896981087585e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     7.76684043468645e-09,
     7.345194938339533e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     7.76684043468645e-09,
     7.322210934062145e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "5a459538917bc6043fd19c7dce618604614879067b2713f310aa0d8faa5a98a7",
    "upstream": "584c8c1ea7d89631486e5cae5c6bc5b573682c6162a291e594e86af3cc34f71b",
    "native_forward": "79df0941d569e6d4a6c20758cf0941a8d18f267ededad9874ebb72d55075eaa5",
    "native_backward": "aa2b7543df98492f1eb63db359ab55b26d0029b3af9cf0d591fb83390a82fee9",
    "candidate_forward": "0c1c3ab5a9cd03a2e073d173b43a52e77b289e05c0a044e54db6832ddf093717",
    "candidate_backward": "b0061c27b08235d723968c1e8d900a22ef10cd0d6802107a903a9ab5cacfd177"
   },
   1.601874828338623e-07,
   0,
   true
  ],
  [
   0,
   "target",
   [
    1,
    14,
    4096,
    4096
   ],
   "torch.bfloat16",
   63101,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     234881024,
     false,
     0,
     2.9802322387695312e-08,
     4.69364830445387e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     3.814697265625e-06,
     1.1351355019900729e-05,
     0.001,
     true
    ],
    [
     234881024,
     false,
     0,
     3.4911159019213045e-08,
     5.2951037353545534e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     3.4911159019213045e-08,
     5.2376635038159436e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     3.814697265625e-06,
     1.3170063466672103e-05,
     0.001,
     true
    ],
    [
     234881024,
     false,
     0,
     3.814697265625e-06,
     1.526232568900658e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "178d445f58b4670a35ef6ef607e9dee7f8322aa1194590b0d5c894231fabc25b",
    "upstream": "f7af1480a2a994dfc0ab747b584b6ccbd1961c3092a08c8a54afb6555db8e3e5",
    "native_forward": "d5ca91dec97da094d26d88641f30a3578789941be29c4f909d53964d004b64de",
    "native_backward": "1bda8c2225ffa42479a88e2c9416418881e8a200ac8b6e628645e8d8c4a44ed9",
    "candidate_forward": "2718b2deb8c34124415c93aeb69dc7e643b099d58b12dbb4fcc5c645ef2df2d7",
    "candidate_backward": "28145ddb8908baa925693074c94d475c27569514590af173002b32e05dd258df"
   },
   1.5486148186028004e-07,
   0,
   true
  ],
  [
   0,
   "target",
   [
    1,
    14,
    4096,
    4096
   ],
   "torch.float32",
   63101,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     234881024,
     false,
     0,
     4.656612873077393e-10,
     2.063061908358854e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     1.7462298274040222e-10,
     2.683868295323416e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.2784851272958946e-08,
     5.300108377000536e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.2784851272958946e-08,
     5.3063415608768204e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     5.373131301000811e-09,
     7.286999999800839e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     5.373131301000811e-09,
     7.292403524900314e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "b994afaa3ca64fb8aa564062a489633186040a846ab52702ff1474b0c028900f",
    "upstream": "f7af1480a2a994dfc0ab747b584b6ccbd1961c3092a08c8a54afb6555db8e3e5",
    "native_forward": "2bebfe885e0c65f8f8b6e7642ce2350b99b36b8465224d0667ebaeb6639f2862",
    "native_backward": "75a5f78bff1b3fb22fbe7e50bd14edb506f15f5119b4476f9b405b4ab4b3020f",
    "candidate_forward": "bb101508e8f56622aa4a5076cbc330c2eeb941548f5712a04f846b21d80c91d0",
    "candidate_backward": "e8f5a960e914a4efccac5bf4fe18c33b591560e047e1a4f8a42bee71a2997ae3"
   },
   1.5326077118515968e-07,
   0,
   true
  ],
  [
   0,
   "target",
   [
    1,
    14,
    2048,
    2048
   ],
   "torch.bfloat16",
   63102,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     58720256,
     false,
     0,
     2.9802322387695312e-08,
     4.878035739475938e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     3.814697265625e-06,
     7.544571638979541e-06,
     0.001,
     true
    ],
    [
     58720256,
     false,
     0,
     4.280985027582673e-08,
     5.3469576908293585e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.113633333968636e-08,
     5.204842975741358e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     3.814697265625e-06,
     1.0177691282413664e-05,
     0.001,
     true
    ],
    [
     58720256,
     false,
     0,
     3.814697265625e-06,
     1.1430430271033896e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "1fe53043d7e0d08b83106a7014cc91477d0601ff4d05db20bdcf20c602f5945c",
    "upstream": "56d22d88f11af2cab3270be3d66039ccb919feebba1f0e226c40f20f2eb89fcb",
    "native_forward": "39cb6b1134a2ccdca3b7e14d5905248d9a0f34c5bb30093d31576c641ed6ebc3",
    "native_backward": "afacd94e65c018f00c19a6460b323c906542871154823c47c51136d873cb6435",
    "candidate_forward": "f572c45772e88fa71d822550cf321379d224c24e5a623ff6a45ffe4626f25864",
    "candidate_backward": "5daa11ff1c772eee90911e85190024720270201bb6db924c1b32229efd62b061"
   },
   1.3899989426136017e-07,
   0,
   true
  ],
  [
   0,
   "target",
   [
    1,
    14,
    2048,
    2048
   ],
   "torch.float32",
   63102,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     58720256,
     false,
     0,
     3.725290298461914e-09,
     3.616061849717798e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     1.1641532182693481e-09,
     4.6753101810563094e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.4250838604487086e-08,
     5.321210387609229e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.4250838604487086e-08,
     5.300181961604132e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     5.590106912090054e-09,
     7.214082168022796e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     5.590106912090054e-09,
     7.199572096263456e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "39084640b3bf5758f774c41374b7fdb25c944e66f26f85b8ef5edfadd57cc17d",
    "upstream": "56d22d88f11af2cab3270be3d66039ccb919feebba1f0e226c40f20f2eb89fcb",
    "native_forward": "af6143c8065d03877906c0905fa8f3ff4d619881074114fb2d20f219daccdc4d",
    "native_backward": "58426ca9cf30697d67c49f7e5db69e0a8312f0e82529728267830b82b2472222",
    "candidate_forward": "ad8061d6eda8f63363597f94708ba1b7d3d3eafc9aec1a5eb448eda7ada36189",
    "candidate_backward": "beb0fc212ffe3e83ae9db10bc9d83e43e867b5dbafbb388062c95757830d9846"
   },
   1.5628756955266e-07,
   0,
   true
  ],
  [
   0,
   "target",
   [
    1,
    14,
    4096,
    4096
   ],
   "torch.bfloat16",
   63102,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     234881024,
     false,
     0,
     2.9802322387695312e-08,
     4.805378670147354e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     3.0517578125e-05,
     3.5450853536185924e-05,
     0.001,
     true
    ],
    [
     234881024,
     false,
     0,
     4.1102273307114245e-08,
     5.29302348207935e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.1102273307114245e-08,
     5.307895904480598e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     3.0517578125e-05,
     3.6359217721770865e-05,
     0.001,
     true
    ],
    [
     234881024,
     false,
     0,
     3.814697265625e-06,
     1.2115806364144605e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "6a87ad940421e31601a8e3bef51e54c059e35c5aa89053f0736360f8a680c67e",
    "upstream": "1239c0d4b07b90008908b59ee22e9ba1f9b66fc4eecd1089cbb321d592e5caeb",
    "native_forward": "14d15e52cb2c66afc914b29caf824b28cfac78da76d878fe82ec6ce3ef325aa2",
    "native_backward": "9441fc1d02f4712edabea594b80519797bb6d4039263ff61467b0bb739ee7ff1",
    "candidate_forward": "943a87af88e7ed970a3df89c21b7fb5da10cceba190f07ab514d1fd368b52022",
    "candidate_backward": "fde9a53a273152945cd1d18f40f601f8d2602317da4f940b538ebdce6ba2ad32"
   },
   1.6827834770083427e-07,
   0,
   true
  ],
  [
   0,
   "target",
   [
    1,
    14,
    4096,
    4096
   ],
   "torch.float32",
   63102,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     234881024,
     false,
     0,
     4.656612873077393e-10,
     2.093322815419162e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     1.4551915228366852e-10,
     2.7478293216517398e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.5011176674769615e-08,
     5.3065916719655186e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.5011176674769615e-08,
     5.30351408785289e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     5.7362278468864325e-09,
     7.325063070507742e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     5.7362278468864325e-09,
     7.32240493549963e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "ef4b6172ea7acc286ff7c3fcc33a6da34aa4913754d235503633297588107607",
    "upstream": "1239c0d4b07b90008908b59ee22e9ba1f9b66fc4eecd1089cbb321d592e5caeb",
    "native_forward": "8244f6eefc5b62349b0ce147eeea01b4081a0b4716e0994e058fc4fd48377c77",
    "native_backward": "f68df1923cb0fa68e6a6732bdc86d8f4367d64f2078d26a1b05b5a2d143fa617",
    "candidate_forward": "a2b576e0c1b4cef10c981bf10d4231345774a33814fb1065130689a945178709",
    "candidate_backward": "8fa54a81ab63c6a5e9c7ad8662e2825cffbfcb9a73c3c981aad3e2268616a28f"
   },
   1.646694727241993e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    1,
    1
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "6e602b749c4fc45c98bc354905c0d1471fa65371932209cf88fc3bafa721d60c",
    "upstream": "2d964498318eb7e9e4e9d7be97b9dfdf54a13b9bf2491c2e46304f3618e56bcf",
    "native_forward": "f46913286d895f62d646314f400fc3d7a543cf34eb4ae56df87f43dc4582a36b",
    "native_backward": "15ec7bf0b50732b49f8228e07d24365338f9e3ab994b00af08e5a3bffe55fd8b",
    "candidate_forward": "f46913286d895f62d646314f400fc3d7a543cf34eb4ae56df87f43dc4582a36b",
    "candidate_backward": "15ec7bf0b50732b49f8228e07d24365338f9e3ab994b00af08e5a3bffe55fd8b"
   },
   0.0,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    2,
    2
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     24,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     24,
     false,
     0,
     3.272206372528785e-08,
     1.9141224837819314e-08,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     3.272206372528785e-08,
     1.9141224837819314e-08,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     24,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "cff1f7beb1042e186caf014a842ffeac5997a872ceda7751a398bdd187110538",
    "upstream": "14a246fc9d7ee38c9c5c2b8930dc100e0f26da450929a8b8f60eb2fb404ea9ab",
    "native_forward": "9575bc814606d65c3a48a26887ac9204110c88109ab86b68e7277b5638b829cf",
    "native_backward": "ec57cd28f04f21c0c234bab08bdb20a2f8882dc0038c8e69d0582d482f64bd41",
    "candidate_forward": "9575bc814606d65c3a48a26887ac9204110c88109ab86b68e7277b5638b829cf",
    "candidate_backward": "ec57cd28f04f21c0c234bab08bdb20a2f8882dc0038c8e69d0582d482f64bd41"
   },
   5.960464477539063e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    3,
    3
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     54,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     54,
     false,
     0,
     3.0784134352757064e-08,
     2.6485539816327432e-08,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     3.0784134352757064e-08,
     2.6485539816327432e-08,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     54,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "f2044477d5f36d599dbc74a1af8123e91e439fb46c89c09d981425a1bde221dd",
    "upstream": "26d5b576c00fb9cc62588aa49b91cb8f93708d4734a8320376862797556b737b",
    "native_forward": "49ae19a988f583cd0b038c2abf9273ca9364a2bd2a858ac377381ab39d669d08",
    "native_backward": "3183435be53227364295fd4b1bfd8bcb496f1ef19ab7d82970673e28539a9259",
    "candidate_forward": "49ae19a988f583cd0b038c2abf9273ca9364a2bd2a858ac377381ab39d669d08",
    "candidate_backward": "3183435be53227364295fd4b1bfd8bcb496f1ef19ab7d82970673e28539a9259"
   },
   5.960464477539063e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     4.127988639801572e-08,
     4.512635312316063e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.127988639801572e-08,
     4.512635312316063e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "3e38f369fb78344a186f2413ecc1d90b5b0892f1302dc888279524c8e7deb667",
    "upstream": "a0ce4e9c30f09ec1299c11efc45e811848edb4ddcdef8942dd34c34f5797e863",
    "native_forward": "1ac3343b11fd794173ae12fb5780aa380a3d10d6f14ecd099d9d81bf6fd7fbb4",
    "native_backward": "44d903e4daea98e85ed15d64d0ba752d8e1442c0be5bb7159c9432a6c49e165c",
    "candidate_forward": "1ac3343b11fd794173ae12fb5780aa380a3d10d6f14ecd099d9d81bf6fd7fbb4",
    "candidate_backward": "44d903e4daea98e85ed15d64d0ba752d8e1442c0be5bb7159c9432a6c49e165c"
   },
   1.0803341865539551e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    127,
    127
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     96774,
     false,
     0,
     3.725290298461914e-09,
     2.062381554774788e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     96774,
     false,
     0,
     4.15412066967491e-08,
     4.893066957853847e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     4.15412066967491e-08,
     4.8773526914231365e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     7.62939453125e-06,
     1.858247258690155e-05,
     0.001,
     true
    ],
    [
     96774,
     false,
     0,
     7.62939453125e-06,
     1.858247258690155e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "f93542f4ff88ee2e655a3f0260a6da128ddde6c3aee65057827e09c0f975cc32",
    "upstream": "b65830f8470cbe97e4c0b6c33a2bb3dea1bdc987c9a6e1cc8faf69720da30004",
    "native_forward": "211cb31d8222cb1d1959c1e431a7d5b83479e79d565e2087ede51a218532d7ce",
    "native_backward": "a78f024e8cf8086b40d3ec3e4e4ab7213aa2962d23e456fdfc326657571b8cd6",
    "candidate_forward": "3aa48958dad0eb29c9628fe4845794507ba0c97bbe973b48c5822be92a64d2d8",
    "candidate_backward": "a78f024e8cf8086b40d3ec3e4e4ab7213aa2962d23e456fdfc326657571b8cd6"
   },
   1.2014061212539673e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    255,
    255
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     390150,
     false,
     0,
     3.725290298461914e-09,
     2.546677396860773e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     1.9073486328125e-06,
     5.993230989489469e-06,
     0.001,
     true
    ],
    [
     390150,
     false,
     0,
     6.409386932304528e-08,
     5.1720863726755754e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     6.409386932304528e-08,
     5.1412324250657485e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     1.52587890625e-05,
     3.5248436372427734e-05,
     0.001,
     true
    ],
    [
     390150,
     false,
     0,
     1.52587890625e-05,
     3.5752357116750226e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "a34940a51f71a4eb50e2f295e12b136372acf007c8944a4dab9bec523c0b7bfd",
    "upstream": "b070bd39c21e86fc6d62afd1c39e5f4417d45474c257f7ce803a8187b317d20c",
    "native_forward": "8baecae3b1c343ce77c49273378da1e66fbe14d4efb1f75a69f19190a1542dfa",
    "native_backward": "5234aa69246c3002c22ffb2106d47b22f4b58f86b116cb3d3a3d1334c09fb321",
    "candidate_forward": "075b7c1fd2255bdd85ad0fd594800eb20c9e01a11bb06638eef56111c568c2f6",
    "candidate_backward": "1840fba6cc13a897129b16488a0f22d307ac23f07f4cffd262cd5ee67e2068dc"
   },
   1.434236764907837e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    257,
    257
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     396294,
     false,
     0,
     3.725290298461914e-09,
     2.6388380584919174e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     2.384185791015625e-07,
     5.308491005517751e-07,
     0.001,
     true
    ],
    [
     396294,
     false,
     0,
     3.580354557275456e-08,
     5.023615774896427e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     3.580354557275456e-08,
     5.0221313053979697e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     2.384185791015625e-07,
     5.29815313627235e-07,
     0.001,
     true
    ],
    [
     396294,
     false,
     0,
     1.4901161193847656e-08,
     3.311345710170219e-08,
     0.001,
     true
    ]
   ],
   {
    "input": "06e86e30ac8a8338241228f2682947f277806d5a847262d1558ab9b87528875f",
    "upstream": "753cd5ef01dfeaad3f23174097a62b537e7e23489140603e2bab214ba39db22e",
    "native_forward": "e8c6a3caa0bbff05f827676bad0c55ad5e1a89f15fbaecfbc6cb8efcbd0805e6",
    "native_backward": "0df397475bf1b2915158a9c2024e92b039a36f924611cb40cd6361033a5bf4a1",
    "candidate_forward": "efb8114334846b134f1c5f382f28d57f41151692ea557e5c35cbcbf0479098d5",
    "candidate_backward": "c6c7d83d179cc7c40d558519d4d685e066d65e3d9a887502d8088839b771c111"
   },
   1.30385160446167e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    1,
    1,
    2047,
    2047
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     4190209,
     false,
     0,
     1.862645149230957e-09,
     3.566419506317351e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     9.5367431640625e-07,
     5.561596047835388e-06,
     0.001,
     true
    ],
    [
     4190209,
     false,
     0,
     3.962185268768792e-08,
     5.217803791889426e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     3.962185268768792e-08,
     5.1742588559388476e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     1.9073486328125e-06,
     9.105341610375104e-06,
     0.001,
     true
    ],
    [
     4190209,
     false,
     0,
     1.9073486328125e-06,
     1.0193884577221881e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "7fb19dd077d46eee121231ee2ee6c7f0d44b3eef333b275903ab098484af3254",
    "upstream": "0a34a4942cf15536dbcf9a06515042c7f0a02f1dfbc2b3cab3f782e4cb6d11df",
    "native_forward": "79a0de0085a6fc720a554dffd6ab116e34c60be78a7db486eba012e9a8b002bc",
    "native_backward": "3acab9d6d6e61891a12492965927c5278c1ef462cd26e8b4c53dbdd082362919",
    "candidate_forward": "624f0e5c5019f68eeb4cc0c661babc7f611abf9c3ad3e80549718fac71190fa1",
    "candidate_backward": "2b5e720a3d8b5d74624e25bffedf8cbe71703fde7bdf544d21647cdfea039d25"
   },
   1.5192199498414993e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    1,
    1,
    2049,
    2049
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     4198401,
     false,
     0,
     4.656612873077393e-10,
     1.7278769281991917e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     4.76837158203125e-07,
     4.032124099761437e-06,
     0.001,
     true
    ],
    [
     4198401,
     false,
     0,
     2.6898947513842586e-08,
     5.302267426013357e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     2.6898947513842586e-08,
     5.2930602221630364e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     9.5367431640625e-07,
     8.315329421041288e-06,
     0.001,
     true
    ],
    [
     4198401,
     false,
     0,
     9.5367431640625e-07,
     8.63466757566927e-06,
     0.001,
     true
    ]
   ],
   {
    "input": "41827f15206567bc72eeadb75a8f4ef30937beff9a4e2320ea2bdfee08d4d801",
    "upstream": "87b7fa131bfc32ed1c4650b6cb84da7a31374a4547d578cc80a3aa679b2a64c4",
    "native_forward": "bcac6913c2569e980ea34d48098d770b6599d6480f01d372f34ca761ef96175a",
    "native_backward": "5a74d77484733a8fa0ffd10907984731e94c054d80c14cb361d2bbaee64f2e91",
    "candidate_forward": "ab52c255fa6705d4b4923da480d31b1e90de7628a7d1e389758a56cc54835798",
    "candidate_backward": "1bded689138d32886efa8ba6e8285d41fce5c00d9ca86a5cfdf856699fc0473d"
   },
   1.5390105545520782e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    1,
    1,
    4095,
    4095
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     16769025,
     false,
     0,
     3.4924596548080444e-10,
     2.0907865334389825e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     4.76837158203125e-07,
     4.537915257182126e-06,
     0.001,
     true
    ],
    [
     16769025,
     false,
     0,
     1.933288917843612e-08,
     5.0899872696029735e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     1.933288917843612e-08,
     5.096742378242412e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     1.9073486328125e-06,
     1.2496518415473971e-05,
     0.001,
     true
    ],
    [
     16769025,
     false,
     0,
     1.9073486328125e-06,
     1.2321424409064232e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "d5ef308725134a84e40d197f496cf83876ac1fdfc8a75a1b7e616c7833c76abe",
    "upstream": "4f3e30055534de9ff84bd1224c6e4ce4a65c0b30834516736522d7edc68d59a3",
    "native_forward": "c79ee07807adac1d739013eb4b0cdc5f87e13b58a8a2f094fe78e358b13f2f12",
    "native_backward": "82c40ec32b5b44880c8f152e8d4f03fc5d70c4ad6b79b205fe7cfeb45aae06d5",
    "candidate_forward": "99884ed4ddd72f820ffa26285d072aa8458ddc26eda3f84da933553213b20700",
    "candidate_backward": "95f5bccb3ffd3fbbe35c80dfea8b4a528a8a820787e10210f92807e7ae62b43e"
   },
   1.5459954738616943e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63104,
   "all-open",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     7.121071524118783e-09,
     5.5078654811627806e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     7.121071524118783e-09,
     5.5078654811627806e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "3b640d7de5c881ed6f691e4cddc9e9d296aef1a13c64dfb35cfaed3ead6afa3f",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "78ccbc250c8e1544b1d2a821ae6869d610f84c7ed4f4c5538bc078c7f77db374",
    "native_backward": "55990b69c814834c10305b7d6ce78e00206bb9277b4390973d31fde273b9f7a5",
    "candidate_forward": "78ccbc250c8e1544b1d2a821ae6869d610f84c7ed4f4c5538bc078c7f77db374",
    "candidate_backward": "55990b69c814834c10305b7d6ce78e00206bb9277b4390973d31fde273b9f7a5"
   },
   9.313225746154785e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63104,
   "all-masked",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     9.613652374174109e-10,
     2.980232235993973e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.613652374174109e-10,
     2.980232235993973e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "3b640d7de5c881ed6f691e4cddc9e9d296aef1a13c64dfb35cfaed3ead6afa3f",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "477a6b9ec23fd18a88caa77a9966a59dab7505a0133223d28cb8185bf4816e3f",
    "native_backward": "d49871a7e88daaf877eec1e123a4442c8acc2f1c992cf2c9cd6e60caf12335ec",
    "candidate_forward": "477a6b9ec23fd18a88caa77a9966a59dab7505a0133223d28cb8185bf4816e3f",
    "candidate_backward": "d49871a7e88daaf877eec1e123a4442c8acc2f1c992cf2c9cd6e60caf12335ec"
   },
   2.9802322387695312e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63104,
   "random-mask",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     1.1128720292519212e-08,
     5.7450048791246e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.1128720292519212e-08,
     5.7450048791246e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.4901161193847656e-08,
     5.686464438916746e-08,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     1.4901161193847656e-08,
     5.686464438916746e-08,
     0.001,
     true
    ]
   ],
   {
    "input": "3b640d7de5c881ed6f691e4cddc9e9d296aef1a13c64dfb35cfaed3ead6afa3f",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "c70787b7246e8f2df331471263f779101fd218756a0400093814e1f5442d0ff0",
    "native_backward": "8811e70054d9b35a24f38f5d044a5002e152f857540f7e9f76c2d243398a95a5",
    "candidate_forward": "c70787b7246e8f2df331471263f779101fd218756a0400093814e1f5442d0ff0",
    "candidate_backward": "8811e70054d9b35a24f38f5d044a5002e152f857540f7e9f76c2d243398a95a5"
   },
   9.685754776000977e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63104,
   "strided-mask",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     1.2135728387485667e-08,
     5.7648735694158225e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.2135728387485667e-08,
     5.7648735694158225e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "3b640d7de5c881ed6f691e4cddc9e9d296aef1a13c64dfb35cfaed3ead6afa3f",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "d462018ea4cddfc77f132e93e561ff3e64313d8558ee29a525dabee6ab1f8ff9",
    "native_backward": "2d4e7039876161d25b9ce87f51132e9e03094d4ee792564ae82d992002e56ead",
    "candidate_forward": "d462018ea4cddfc77f132e93e561ff3e64313d8558ee29a525dabee6ab1f8ff9",
    "candidate_backward": "2d4e7039876161d25b9ce87f51132e9e03094d4ee792564ae82d992002e56ead"
   },
   1.1548399925231934e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63105,
   "causal",
   1.0,
   "large-negative",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "c0117d6fa9629ddf11e97910492711946a6c0c35f1bacf85e0aa7f77b63b6e17",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "94ce1e3bd971720698c51825a5708c4e28ede19252c04e41fd3fb12d3dc2ddb8",
    "native_backward": "0f64be7a07f263e4467f4ecfab166ec04670fe6a50e8a5c17dbe040ae3749633",
    "candidate_forward": "94ce1e3bd971720698c51825a5708c4e28ede19252c04e41fd3fb12d3dc2ddb8",
    "candidate_backward": "0f64be7a07f263e4467f4ecfab166ec04670fe6a50e8a5c17dbe040ae3749633"
   },
   5.21540641784668e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63105,
   "causal",
   0.08838834764831843,
   "zeros",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "d49871a7e88daaf877eec1e123a4442c8acc2f1c992cf2c9cd6e60caf12335ec",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "native_backward": "26a10aa508e7f9fe26976ef3bfbed44c567f3dc78ff9114ce0944409b1534f69",
    "candidate_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "candidate_backward": "26a10aa508e7f9fe26976ef3bfbed44c567f3dc78ff9114ce0944409b1534f69"
   },
   5.21540641784668e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63105,
   "causal",
   0.0,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "e34a384fcf42f7439a56b0a3680f89f7d84b241eed6efce3da533eab10732ff9",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "native_backward": "dfed62fd5ed2643c48baa3333e2ce6c4e448ccd1204b57f78d66501d2d440079",
    "candidate_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "candidate_backward": "dfed62fd5ed2643c48baa3333e2ce6c4e448ccd1204b57f78d66501d2d440079"
   },
   5.21540641784668e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63105,
   "causal",
   -0.25,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     5.960464477539063e-08,
     2.3154470250210007e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     3.0517578125e-05,
     3.14594106217123e-05,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     4.956657762011929e-08,
     4.7384050420991063e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     5.358054766535503e-08,
     4.6265321853876494e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     3.0517578125e-05,
     3.1459405842459247e-05,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "e34a384fcf42f7439a56b0a3680f89f7d84b241eed6efce3da533eab10732ff9",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "8cceff1cce3edc3b1b860b7c530493a10ae3a94ce728ed5bc1d97303d4a50a9d",
    "native_backward": "01286d6dc15af865b4aceb9c7da20f72818de437b771298961ebf08ccc3296b6",
    "candidate_forward": "952ea86d8a893d7c2717582cb3dcd959569645565c57caa532ae3cf8f612c599",
    "candidate_backward": "a0abca5b17b9ec13c3070c1025c665525d6bbaadfcac53bd6279085e6f739a29"
   },
   1.0244548320770264e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63105,
   "causal",
   0.08838834764831843,
   "random",
   true,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     5.315993223398152e-08,
     4.421124443907447e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     5.315993223398152e-08,
     4.421124443907447e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "e34a384fcf42f7439a56b0a3680f89f7d84b241eed6efce3da533eab10732ff9",
    "upstream": "18e156785d6c491434703f68d20be989626977657baf3c8751bce402b0c1c51e",
    "native_forward": "22b97cf3db6db278da28d369ebcfa36042d335fb11e20472029a95cfcd23960f",
    "native_backward": "142ae11683c71dc9e219fe3a80654bb14441ba5535ef62abc8bf68a9df28fbcd",
    "candidate_forward": "22b97cf3db6db278da28d369ebcfa36042d335fb11e20472029a95cfcd23960f",
    "candidate_backward": "142ae11683c71dc9e219fe3a80654bb14441ba5535ef62abc8bf68a9df28fbcd"
   },
   1.0058283805847168e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    1,
    1
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ]
   ],
   {
    "input": "97e4790a364624f7974c756f9de7df2d76952d0f320d1ff278043f1939deb4ff",
    "upstream": "2d964498318eb7e9e4e9d7be97b9dfdf54a13b9bf2491c2e46304f3618e56bcf",
    "native_forward": "f46913286d895f62d646314f400fc3d7a543cf34eb4ae56df87f43dc4582a36b",
    "native_backward": "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0",
    "candidate_forward": "f46913286d895f62d646314f400fc3d7a543cf34eb4ae56df87f43dc4582a36b",
    "candidate_backward": "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0"
   },
   0.0,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    2,
    2
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     24,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     3.569689632776374e-08,
     2.0396721056285102e-08,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     3.569689632776374e-08,
     2.0396721056285102e-08,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     3.289413311180489e-09,
     6.33053948733173e-08,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     3.289413311180489e-09,
     6.33053948733173e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "e3795b488138dd2ab723864fa6ae0d3ec2b30e0df049666dc452c8e01902bac4",
    "upstream": "14a246fc9d7ee38c9c5c2b8930dc100e0f26da450929a8b8f60eb2fb404ea9ab",
    "native_forward": "d41c386d1bb2b638d43928a525fcc91342d3765e3247eb4f385dab2dc8915804",
    "native_backward": "2dc879e04ea129375728d4c95cedd72cd86422d9f27663451225761b74f760ef",
    "candidate_forward": "d41c386d1bb2b638d43928a525fcc91342d3765e3247eb4f385dab2dc8915804",
    "candidate_backward": "2dc879e04ea129375728d4c95cedd72cd86422d9f27663451225761b74f760ef"
   },
   2.9802322387695312e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    3,
    3
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     54,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     4.4220734851663224e-08,
     2.7966713410464728e-08,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     4.4220734851663224e-08,
     2.7966713410464728e-08,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     2.685563195437002e-09,
     5.1669630310457116e-08,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     2.685563195437002e-09,
     5.1669630310457116e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "56b73b2b26e25644fa325bb527c28b2feeb162c08aa5ef83dad6dee1130af1af",
    "upstream": "26d5b576c00fb9cc62588aa49b91cb8f93708d4734a8320376862797556b737b",
    "native_forward": "e213ab1aa9eb6c7fb2cb4db9e64085acf83cd8b80eb346c8fcd0d0f028147bde",
    "native_backward": "244a07e7c38c29badfdf33e919c4fff12413ac48fc02832aa4c57575cb82032a",
    "candidate_forward": "e213ab1aa9eb6c7fb2cb4db9e64085acf83cd8b80eb346c8fcd0d0f028147bde",
    "candidate_backward": "244a07e7c38c29badfdf33e919c4fff12413ac48fc02832aa4c57575cb82032a"
   },
   5.960464477539063e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.623894045430177e-08,
     4.345270130559963e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.623894045430177e-08,
     4.345270130559963e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     6.53414781830175e-09,
     7.457748853466223e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     6.53414781830175e-09,
     7.457748853466223e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "0c21cac1b425f49ac893a40723bed000efeffe15e3b2e4a42792b13c916bef2f",
    "upstream": "a0ce4e9c30f09ec1299c11efc45e811848edb4ddcdef8942dd34c34f5797e863",
    "native_forward": "13722a83fdc848dcd8597fa9cd9aabf6a1f419cf5ae47d4e55405d74fa79e372",
    "native_backward": "13491b793ac473dd49d74f5f914788af12531a3bf4f37d97a415d60aaafec662",
    "candidate_forward": "13722a83fdc848dcd8597fa9cd9aabf6a1f419cf5ae47d4e55405d74fa79e372",
    "candidate_backward": "13491b793ac473dd49d74f5f914788af12531a3bf4f37d97a415d60aaafec662"
   },
   1.1548399925231934e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    127,
    127
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     96774,
     false,
     0,
     3.725290298461914e-09,
     1.9975713310104676e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     9.313225746154785e-10,
     3.028928598025468e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     3.5059435132822614e-08,
     4.856882737853011e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     3.5059435132822614e-08,
     4.852829541202915e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     3.568572371631662e-09,
     7.289135270559189e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     3.568572371631662e-09,
     7.291605463581857e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "5a9f6e3f5ea82e7136235671f7b96d201352a0c8236aaf7d8d9569479c84d311",
    "upstream": "b65830f8470cbe97e4c0b6c33a2bb3dea1bdc987c9a6e1cc8faf69720da30004",
    "native_forward": "c56bb1fd873604476f887f5da252d79056ea33436418295f84fcb1a0cd0cdbe7",
    "native_backward": "ab6ae51538215060ed143e72e6f850e14b34ddf4ac1bebce85463b0a5a82e09e",
    "candidate_forward": "4e16272d7974bc0bdcc3e7e14a8e1af13b1326cbf163ecb9aee57c8ab2b709b9",
    "candidate_backward": "64f8dc348b9388bfe1d8a4f10a61916d7abf889f0b0ea67e632d99bbe5c4b8bb"
   },
   1.5832483768463135e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    255,
    255
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     390150,
     false,
     0,
     3.725290298461914e-09,
     2.627396427656501e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     6.984919309616089e-10,
     3.799797733729154e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     4.031189537734292e-08,
     5.018001317495489e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     4.031189537734292e-08,
     5.002096734627235e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     3.3278994421293007e-09,
     7.183678493388644e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     3.3278994421293007e-09,
     7.182632647554721e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "22060da8660015e1e3357416bb8d9c4b0350eaeb601462245b7434d6c9d731c7",
    "upstream": "b070bd39c21e86fc6d62afd1c39e5f4417d45474c257f7ce803a8187b317d20c",
    "native_forward": "fd9d11a2c9cc72f68b8c2caf781c0057905dfa44e11508817dc73decde4c916d",
    "native_backward": "c5f2cf180d7e4afd131cf3c5feaaca80f5f8a27ab070bb9e5f23a5a27b628f1f",
    "candidate_forward": "95dac3bcf60705af0a787d5bfe0e601e979b2bd20dea11a7d2fb1b5ab7fbf68a",
    "candidate_backward": "140854b15dd6b3d8091c58b08573e19c192d7669ea26af003edc4afc0f4e0a59"
   },
   1.2293457984924316e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    257,
    257
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     396294,
     false,
     0,
     3.725290298461914e-09,
     2.703644932059261e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     6.984919309616089e-10,
     3.8853535699831246e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     4.993713187406712e-08,
     4.991066638856359e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     4.993713187406712e-08,
     4.989444619128417e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     7.612985658544957e-09,
     7.337705163512448e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     7.612985658544957e-09,
     7.339394960915919e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "055fab1dbfa19ef68fe1f3560fffc0a93f78676960f91e1d15544251e6d72be0",
    "upstream": "753cd5ef01dfeaad3f23174097a62b537e7e23489140603e2bab214ba39db22e",
    "native_forward": "c3823718861d695d2f6a16dfadb2e8e97a1e1862e1c8facc4a90e49b005a8bb5",
    "native_backward": "2dca9f65d5de4253e029a40a4b6d9b8c62d19d8b164d625831ca13f87ad50c11",
    "candidate_forward": "45ff216f31fe0bfd9b2c35fd4aa84030d91afd25aec0a2a65413d4128a5cff65",
    "candidate_backward": "a6aa4663154100981006453ff3347e4fd1c281ed229884e4530dc55280599d89"
   },
   1.1920928955078125e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    1,
    1,
    2047,
    2047
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     4190209,
     false,
     0,
     3.725290298461914e-09,
     3.747845830738651e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     6.984919309616089e-10,
     4.909472447309809e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     1.936420404580197e-08,
     5.291629786251054e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     1.936420404580197e-08,
     5.277109211377292e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     3.759538159953113e-09,
     7.500797484583873e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     3.759538159953113e-09,
     7.506176227290741e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "133b34e22bb48d8539bfac1862a7fa6ab86ab72688670e6407d9a4ba4431eb3c",
    "upstream": "0a34a4942cf15536dbcf9a06515042c7f0a02f1dfbc2b3cab3f782e4cb6d11df",
    "native_forward": "829a714bc9e376f462fb741e3907e2d4f6df2fda041f14f7a42a50a9913f170f",
    "native_backward": "2ef322173f0a1abdf434345210ed392039a710f12cb6ee954a27aa11b5c7e02a",
    "candidate_forward": "eb69ef16d275805a26ab2abc706aa4942a95e5ef58515b53a1d625ad351e24b0",
    "candidate_backward": "0cd2fca182c63d2fd5f7d664f542c2a6fa1ad353273f6fedfb14ae485bade7c8"
   },
   1.4193938113749027e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    1,
    1,
    2049,
    2049
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     4198401,
     false,
     0,
     3.4924596548080444e-10,
     1.5868487554703808e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     1.1641532182693481e-10,
     2.0543491438466058e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     2.5684546212101367e-08,
     5.319121273569074e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     2.5684546212101367e-08,
     5.321010761122377e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     3.7701664776235155e-09,
     7.482357332318883e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     3.7701664776235155e-09,
     7.483414391251959e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "ca1ff011c0567f8d7964969a081615f97c0624c336d7c62d583281efa69a7d9c",
    "upstream": "87b7fa131bfc32ed1c4650b6cb84da7a31374a4547d578cc80a3aa679b2a64c4",
    "native_forward": "ce56a9439a59dde98c69d604fe6f104c08d4be146447dc02cfc9088e33110557",
    "native_backward": "af4789b8348ab36020da4fe4aad86dc6aac0764df5484cc71a76c207d560f32e",
    "candidate_forward": "d4635f644066f6cf06fa99a49c0529880cb2bf7bccaffcba1f431f1ff475d79c",
    "candidate_backward": "e846481b8dd80b409f6aa80b8a9cdd1891c542a674023f4f14e1c505a41edea8"
   },
   1.2962846085429192e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    1,
    1,
    4095,
    4095
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     16769025,
     false,
     0,
     4.656612873077393e-10,
     2.140361109160751e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     1.1641532182693481e-10,
     2.6674458599638703e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     2.9149262559258204e-08,
     5.4062414696177244e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     2.9149262559258204e-08,
     5.393988722562874e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     3.934552843254391e-09,
     7.126396762106084e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     3.934552843254391e-09,
     7.115964050469328e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "9b54860febae149f30424c33d1bdbb2b3142a4c6bba2a0dc5118483a91ab806f",
    "upstream": "4f3e30055534de9ff84bd1224c6e4ce4a65c0b30834516736522d7edc68d59a3",
    "native_forward": "f2de3935f05cb7e7469bdb2d1440f35ae2c2ac19eb2c963ce87a7a9e69da4432",
    "native_backward": "ce04da4673c5622cda5fb6e5e9c0b0323c8551f620ba4301bdd8d59b6afa9272",
    "candidate_forward": "d280d0f1c424123181f0d6ae5e946c82e7d88b0c5c5d4cc13eea0bd47d4bfd00",
    "candidate_backward": "fe17efbcdc4c9382ede5e25ff30d6e7ddda01e29f8d7055a386a5f6ec941af3f"
   },
   1.2996315490454435e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63104,
   "all-open",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     6.539937660865469e-09,
     5.513767966299546e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     6.539937660865469e-09,
     5.513767966299546e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.6156578592357862e-09,
     7.124098973620619e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.6156578592357862e-09,
     7.124098973620619e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "9eca1b7249e56a01756880dc7f06561aec9f1684af14837396b7618a5e2a98e7",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "5ce81a4bcbb6dcc020d7c5589e33270c817fa236dfd913d3e03445da4c3e4af9",
    "native_backward": "bc06df06b58bd3146c61146b12ca35834448a1ba9192f76fa5e345d443c92043",
    "candidate_forward": "5ce81a4bcbb6dcc020d7c5589e33270c817fa236dfd913d3e03445da4c3e4af9",
    "candidate_backward": "bc06df06b58bd3146c61146b12ca35834448a1ba9192f76fa5e345d443c92043"
   },
   8.568167686462402e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63104,
   "all-masked",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.613652374174109e-10,
     2.980232235993973e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.613652374174109e-10,
     2.980232235993973e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ]
   ],
   {
    "input": "9eca1b7249e56a01756880dc7f06561aec9f1684af14837396b7618a5e2a98e7",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "477a6b9ec23fd18a88caa77a9966a59dab7505a0133223d28cb8185bf4816e3f",
    "native_backward": "dbb056b04cf7e09668b449f9280e43ac832356e3fb4bc3590ed5a3f017dc7cf7",
    "candidate_forward": "477a6b9ec23fd18a88caa77a9966a59dab7505a0133223d28cb8185bf4816e3f",
    "candidate_backward": "dbb056b04cf7e09668b449f9280e43ac832356e3fb4bc3590ed5a3f017dc7cf7"
   },
   2.9802322387695312e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63104,
   "random-mask",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.1345216779923284e-08,
     5.788858753768595e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.1345216779923284e-08,
     5.788858753768595e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.7910227884110341e-09,
     7.35516604833963e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.7910227884110341e-09,
     7.35516604833963e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "9eca1b7249e56a01756880dc7f06561aec9f1684af14837396b7618a5e2a98e7",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "5250aec63c3ec0835b3eb998450904b4c8e097d4b768b492cf27c52bea3ac803",
    "native_backward": "e669c646516fc7044388e7e1bd8d74988954867ab075f06818ad37f72c49addf",
    "candidate_forward": "5250aec63c3ec0835b3eb998450904b4c8e097d4b768b492cf27c52bea3ac803",
    "candidate_backward": "e669c646516fc7044388e7e1bd8d74988954867ab075f06818ad37f72c49addf"
   },
   1.0803341865539551e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63104,
   "strided-mask",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.0911934647528376e-08,
     5.9313710461098403e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.0911934647528376e-08,
     5.9313710461098403e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     2.274017656470062e-09,
     7.437920816800971e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     2.274017656470062e-09,
     7.437920816800971e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "9eca1b7249e56a01756880dc7f06561aec9f1684af14837396b7618a5e2a98e7",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "3ab68c4334cbc2b19800ded5a11073e6be98880cc60b6bd13d3083d9bc63eacf",
    "native_backward": "355bc9d24adddc79ad524be6f15fba9413f6a7c15f6f269120e69024b23847ac",
    "candidate_forward": "3ab68c4334cbc2b19800ded5a11073e6be98880cc60b6bd13d3083d9bc63eacf",
    "candidate_backward": "355bc9d24adddc79ad524be6f15fba9413f6a7c15f6f269120e69024b23847ac"
   },
   1.1920928955078125e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63105,
   "causal",
   1.0,
   "large-negative",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     8.766643092017468e-09,
     6.14927687905896e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     8.766643092017468e-09,
     6.14927687905896e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "ca8d88b595aa52d814b8686c9c5d68452c23ac0c2e066331269cd4e711505071",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "94ce1e3bd971720698c51825a5708c4e28ede19252c04e41fd3fb12d3dc2ddb8",
    "native_backward": "ec64099a42259eb7de473f26599dd2993d1418655252d3522f70b317127401ed",
    "candidate_forward": "94ce1e3bd971720698c51825a5708c4e28ede19252c04e41fd3fb12d3dc2ddb8",
    "candidate_backward": "ec64099a42259eb7de473f26599dd2993d1418655252d3522f70b317127401ed"
   },
   5.21540641784668e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63105,
   "causal",
   0.08838834764831843,
   "zeros",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.345741801137226e-09,
     5.386282923363045e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.345741801137226e-09,
     5.386282923363045e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "dbb056b04cf7e09668b449f9280e43ac832356e3fb4bc3590ed5a3f017dc7cf7",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "native_backward": "47678db904c9fdbb0e5f5ad805a60f4ea82d8a6363b6f28e0bd67e106bcf0d7f",
    "candidate_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "candidate_backward": "47678db904c9fdbb0e5f5ad805a60f4ea82d8a6363b6f28e0bd67e106bcf0d7f"
   },
   5.21540641784668e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63105,
   "causal",
   0.0,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ]
   ],
   {
    "input": "619bceb7cf51993999287904e9a9a0324bb1fbfcf0293a55a299c918da6ecc8f",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "native_backward": "6e2e536f6b068bdc2e5edbf1f3d3d57e2e56c72a2876dcd93d4a078dadc8633f",
    "candidate_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "candidate_backward": "6e2e536f6b068bdc2e5edbf1f3d3d57e2e56c72a2876dcd93d4a078dadc8633f"
   },
   5.21540641784668e-08,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63105,
   "causal",
   -0.25,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     2.9802322387695312e-08,
     1.908662774645302e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     7.450580596923828e-09,
     3.009303014932518e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.034320033596828e-08,
     4.757451667412049e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.034320033596828e-08,
     4.596470680630464e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.447217118366595e-08,
     7.187394792396369e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.447217118366595e-08,
     7.01590177081028e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "619bceb7cf51993999287904e9a9a0324bb1fbfcf0293a55a299c918da6ecc8f",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "40923b5a902f5dc5e39641d2cf07c7353f5421b7df4745800713473ab05c67c8",
    "native_backward": "f07bb50e115d3f20ee3fa4b124df8ff553ffba9c34d2c4739cfe3229763e552e",
    "candidate_forward": "f2bb587407fc7532b60d6bf258edacebc685057ccfcae86c338a45fda2fb4e0b",
    "candidate_backward": "2bda107648ec6e2a7f7326031abab22e96980c44d5bdfc2db6556a9aba85eb14"
   },
   1.2479722499847412e-07,
   0,
   true
  ],
  [
   0,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63105,
   "causal",
   0.08838834764831843,
   "random",
   true,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     2.8564231702166154e-08,
     4.2374034187812844e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     2.8564231702166154e-08,
     4.2374034187812844e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.765884228929984e-09,
     6.681776809527146e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.765884228929984e-09,
     6.681776809527146e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "619bceb7cf51993999287904e9a9a0324bb1fbfcf0293a55a299c918da6ecc8f",
    "upstream": "18e156785d6c491434703f68d20be989626977657baf3c8751bce402b0c1c51e",
    "native_forward": "cfb942e3d9a75d6b32cde65acaf53deaf8cccc49db47a7b502d89cdc7dbda881",
    "native_backward": "fc6816e93dc80bdabca44612cb97e20736c4c3c175a80396ff6391187fedbdbe",
    "candidate_forward": "cfb942e3d9a75d6b32cde65acaf53deaf8cccc49db47a7b502d89cdc7dbda881",
    "candidate_backward": "fc6816e93dc80bdabca44612cb97e20736c4c3c175a80396ff6391187fedbdbe"
   },
   1.043081283569336e-07,
   0,
   true
  ],
  [
   1,
   "target",
   [
    1,
    14,
    2048,
    2048
   ],
   "torch.bfloat16",
   63100,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     58720256,
     false,
     0,
     2.9802322387695312e-08,
     4.672940860860489e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     1.9073486328125e-06,
     7.414175452977063e-06,
     0.001,
     true
    ],
    [
     58720256,
     false,
     0,
     5.4706867369525014e-08,
     5.328086906700259e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     5.4706867369525014e-08,
     5.3041651406504105e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     1.52587890625e-05,
     2.0788746203636328e-05,
     0.001,
     true
    ],
    [
     58720256,
     false,
     0,
     1.52587890625e-05,
     2.101007929912348e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "d2eae38d4762531df6d369e8175fec9ed4ee340a38966281b03e46af5958b930",
    "upstream": "efc7cd1fc02f55761fc3903a8e482513b95991d46bade8f91fa90570b325955a",
    "native_forward": "85a5b825fdbaae4cdd4fd097dc5e6e4600f42129f8a785ffb58137c72be3ef4c",
    "native_backward": "02c62f2a64bd3e750a4ec72f61b82d8a5a0ade6fed581bb69f5d414c1109cbaa",
    "candidate_forward": "0f8ad05b3ee327d7640e835d79d7a09a93053b41fc1429318a9e756fb989ba65",
    "candidate_backward": "ac5ecd61424fefeadbb412299ccb5c6b8f57ffdedd55f3d9889eee9ba87e65b5"
   },
   1.4045508578419685e-07,
   0,
   true
  ],
  [
   1,
   "target",
   [
    1,
    14,
    2048,
    2048
   ],
   "torch.float32",
   63100,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     58720256,
     false,
     0,
     3.725290298461914e-09,
     3.58600562879739e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     9.313225746154785e-10,
     4.650116359079598e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.435542999559061e-08,
     5.2547896673704584e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.435542999559061e-08,
     5.232624191569617e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     6.634688409778988e-09,
     7.437098336823825e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     6.634688409778988e-09,
     7.418626818409743e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "e631f015fc1bd6ffda0171a4dfc093bfa6c2d0c210e4b6749b98ad3c11fab34a",
    "upstream": "efc7cd1fc02f55761fc3903a8e482513b95991d46bade8f91fa90570b325955a",
    "native_forward": "c8a91b9a5950d7eaea1fc3bf80680cb7ce514f7bd1d82c9fd6af1b159395a839",
    "native_backward": "1d60501c542b610d42db5c5886f79ed38c040ed7949f8f60e31fb18056a91487",
    "candidate_forward": "f2d94466b542bafbb6c097a44b83d4a41f3a83ac7828b6e0bc578cc5a1f89890",
    "candidate_backward": "7b13b588dde70dde61acaf4d960d9cd5f3894ad68946dcb80f61067e50632c0b"
   },
   1.50932464748621e-07,
   0,
   true
  ],
  [
   1,
   "target",
   [
    1,
    14,
    4096,
    4096
   ],
   "torch.bfloat16",
   63100,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     234881024,
     false,
     0,
     2.9802322387695312e-08,
     4.7239326375562115e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     7.62939453125e-06,
     1.3179844121984294e-05,
     0.001,
     true
    ],
    [
     234881024,
     false,
     0,
     4.576568635172862e-08,
     5.317207529072065e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.576568635172862e-08,
     5.2909979461476154e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     7.62939453125e-06,
     1.701035920727131e-05,
     0.001,
     true
    ],
    [
     234881024,
     false,
     0,
     3.814697265625e-06,
     1.3434721588220862e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "1822427368be84c7a78249e475a78d4c5011ca0e53979d7466c95b1ac24036f0",
    "upstream": "2b4b8ac25d43a4a6f3e6cd5960ef3e1b4516771b1ad202088bc709a6f2da30a6",
    "native_forward": "9e4b04642a34e116ac6519a9b1b501e774adca0f83f56b804d4c4b81b97cb21e",
    "native_backward": "6f835c86501027243181f1109b356a6c5e92d10ba1e612c55ecae68db0b63e98",
    "candidate_forward": "28720cb3990e2aa4628c496d1ad542682caa4292b2baa484e86a01c2686879e2",
    "candidate_backward": "cc59f7248edf13746d5e2a57bdc5044223ef89f13b6373f83d7ede7d75b6b271"
   },
   1.57160684466362e-07,
   0,
   true
  ],
  [
   1,
   "target",
   [
    1,
    14,
    4096,
    4096
   ],
   "torch.float32",
   63100,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     234881024,
     false,
     0,
     4.656612873077393e-10,
     2.0939859992853893e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     1.7462298274040222e-10,
     2.6826976411967284e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     6.186074019698395e-08,
     5.273726372811987e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     6.186074019698395e-08,
     5.272045513197869e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.464342824236911e-09,
     7.191454206759172e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.464342824236911e-09,
     7.190400214116717e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "9710d4c04d3e81a082a128a46bc2b2c860d240e5a2d9156968538ad7abb17e53",
    "upstream": "2b4b8ac25d43a4a6f3e6cd5960ef3e1b4516771b1ad202088bc709a6f2da30a6",
    "native_forward": "4451f6cf902d0da1d04817474f583d308356c95474874edf0c82b6cd0ff3c05b",
    "native_backward": "0a68326e0f1f180f5d18bd345c60d5094aace15cdc36265ab05fb17c75b37a24",
    "candidate_forward": "2f2430c78faca6fd76dd0c337acad2bb63ca00bc7c15d9897482cefc2213937a",
    "candidate_backward": "51cae9345597de2aef41effe12b9549c4a9c6f979ce6b9cb96dc616bc71ca492"
   },
   1.655425876379013e-07,
   0,
   true
  ],
  [
   1,
   "target",
   [
    1,
    14,
    2048,
    2048
   ],
   "torch.bfloat16",
   63101,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     58720256,
     false,
     0,
     2.9802322387695312e-08,
     4.489604071360259e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     3.814697265625e-06,
     1.05530321369602e-05,
     0.001,
     true
    ],
    [
     58720256,
     false,
     0,
     4.2496641039502236e-08,
     5.2356928144230984e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.2496641039502236e-08,
     5.169549203664316e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     7.62939453125e-06,
     1.4771655151334465e-05,
     0.001,
     true
    ],
    [
     58720256,
     false,
     0,
     7.62939453125e-06,
     1.5236406112271593e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "78ffa1714ca3bf56d8f7f8b434bd3be0af6ba9e05161498cc32aad3a87556796",
    "upstream": "584c8c1ea7d89631486e5cae5c6bc5b573682c6162a291e594e86af3cc34f71b",
    "native_forward": "d310edb1c205953b66dfaa90155546d2d39a17880eeb114f205d19303547da70",
    "native_backward": "0c2d2f7a86f0c36637999e51886937accda9335f0e72092ab6e9961e0aea581f",
    "candidate_forward": "6dc5de53b0c01a5e2644bc875591ee700312f710221ad798b835b41b6c2f3fb2",
    "candidate_backward": "b9acd6b84b70f3df571f152916803a0ac380a0fd98b0a2470ccd07090f2537d9"
   },
   1.3748649507761002e-07,
   0,
   true
  ],
  [
   1,
   "target",
   [
    1,
    14,
    2048,
    2048
   ],
   "torch.float32",
   63101,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     58720256,
     false,
     0,
     2.7939677238464355e-09,
     3.6081517211926234e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     1.1641532182693481e-09,
     4.752618949204581e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.1549625184877925e-08,
     5.249437721437406e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.1549625184877925e-08,
     5.2190896981087585e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     7.76684043468645e-09,
     7.345194938339533e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     7.76684043468645e-09,
     7.322210934062145e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "5a459538917bc6043fd19c7dce618604614879067b2713f310aa0d8faa5a98a7",
    "upstream": "584c8c1ea7d89631486e5cae5c6bc5b573682c6162a291e594e86af3cc34f71b",
    "native_forward": "79df0941d569e6d4a6c20758cf0941a8d18f267ededad9874ebb72d55075eaa5",
    "native_backward": "aa2b7543df98492f1eb63db359ab55b26d0029b3af9cf0d591fb83390a82fee9",
    "candidate_forward": "0c1c3ab5a9cd03a2e073d173b43a52e77b289e05c0a044e54db6832ddf093717",
    "candidate_backward": "b0061c27b08235d723968c1e8d900a22ef10cd0d6802107a903a9ab5cacfd177"
   },
   1.601874828338623e-07,
   0,
   true
  ],
  [
   1,
   "target",
   [
    1,
    14,
    4096,
    4096
   ],
   "torch.bfloat16",
   63101,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     234881024,
     false,
     0,
     2.9802322387695312e-08,
     4.69364830445387e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     3.814697265625e-06,
     1.1351355019900729e-05,
     0.001,
     true
    ],
    [
     234881024,
     false,
     0,
     3.4911159019213045e-08,
     5.2951037353545534e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     3.4911159019213045e-08,
     5.2376635038159436e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     3.814697265625e-06,
     1.3170063466672103e-05,
     0.001,
     true
    ],
    [
     234881024,
     false,
     0,
     3.814697265625e-06,
     1.526232568900658e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "178d445f58b4670a35ef6ef607e9dee7f8322aa1194590b0d5c894231fabc25b",
    "upstream": "f7af1480a2a994dfc0ab747b584b6ccbd1961c3092a08c8a54afb6555db8e3e5",
    "native_forward": "d5ca91dec97da094d26d88641f30a3578789941be29c4f909d53964d004b64de",
    "native_backward": "1bda8c2225ffa42479a88e2c9416418881e8a200ac8b6e628645e8d8c4a44ed9",
    "candidate_forward": "2718b2deb8c34124415c93aeb69dc7e643b099d58b12dbb4fcc5c645ef2df2d7",
    "candidate_backward": "28145ddb8908baa925693074c94d475c27569514590af173002b32e05dd258df"
   },
   1.5486148186028004e-07,
   0,
   true
  ],
  [
   1,
   "target",
   [
    1,
    14,
    4096,
    4096
   ],
   "torch.float32",
   63101,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     234881024,
     false,
     0,
     4.656612873077393e-10,
     2.063061908358854e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     1.7462298274040222e-10,
     2.683868295323416e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.2784851272958946e-08,
     5.300108377000536e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.2784851272958946e-08,
     5.3063415608768204e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     5.373131301000811e-09,
     7.286999999800839e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     5.373131301000811e-09,
     7.292403524900314e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "b994afaa3ca64fb8aa564062a489633186040a846ab52702ff1474b0c028900f",
    "upstream": "f7af1480a2a994dfc0ab747b584b6ccbd1961c3092a08c8a54afb6555db8e3e5",
    "native_forward": "2bebfe885e0c65f8f8b6e7642ce2350b99b36b8465224d0667ebaeb6639f2862",
    "native_backward": "75a5f78bff1b3fb22fbe7e50bd14edb506f15f5119b4476f9b405b4ab4b3020f",
    "candidate_forward": "bb101508e8f56622aa4a5076cbc330c2eeb941548f5712a04f846b21d80c91d0",
    "candidate_backward": "e8f5a960e914a4efccac5bf4fe18c33b591560e047e1a4f8a42bee71a2997ae3"
   },
   1.5326077118515968e-07,
   0,
   true
  ],
  [
   1,
   "target",
   [
    1,
    14,
    2048,
    2048
   ],
   "torch.bfloat16",
   63102,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     58720256,
     false,
     0,
     2.9802322387695312e-08,
     4.878035739475938e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     3.814697265625e-06,
     7.544571638979541e-06,
     0.001,
     true
    ],
    [
     58720256,
     false,
     0,
     4.280985027582673e-08,
     5.3469576908293585e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.113633333968636e-08,
     5.204842975741358e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     3.814697265625e-06,
     1.0177691282413664e-05,
     0.001,
     true
    ],
    [
     58720256,
     false,
     0,
     3.814697265625e-06,
     1.1430430271033896e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "1fe53043d7e0d08b83106a7014cc91477d0601ff4d05db20bdcf20c602f5945c",
    "upstream": "56d22d88f11af2cab3270be3d66039ccb919feebba1f0e226c40f20f2eb89fcb",
    "native_forward": "39cb6b1134a2ccdca3b7e14d5905248d9a0f34c5bb30093d31576c641ed6ebc3",
    "native_backward": "afacd94e65c018f00c19a6460b323c906542871154823c47c51136d873cb6435",
    "candidate_forward": "f572c45772e88fa71d822550cf321379d224c24e5a623ff6a45ffe4626f25864",
    "candidate_backward": "5daa11ff1c772eee90911e85190024720270201bb6db924c1b32229efd62b061"
   },
   1.3899989426136017e-07,
   0,
   true
  ],
  [
   1,
   "target",
   [
    1,
    14,
    2048,
    2048
   ],
   "torch.float32",
   63102,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     58720256,
     false,
     0,
     3.725290298461914e-09,
     3.616061849717798e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     1.1641532182693481e-09,
     4.6753101810563094e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.4250838604487086e-08,
     5.321210387609229e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     4.4250838604487086e-08,
     5.300181961604132e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     5.590106912090054e-09,
     7.214082168022796e-08,
     1e-05,
     true
    ],
    [
     58720256,
     false,
     0,
     5.590106912090054e-09,
     7.199572096263456e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "39084640b3bf5758f774c41374b7fdb25c944e66f26f85b8ef5edfadd57cc17d",
    "upstream": "56d22d88f11af2cab3270be3d66039ccb919feebba1f0e226c40f20f2eb89fcb",
    "native_forward": "af6143c8065d03877906c0905fa8f3ff4d619881074114fb2d20f219daccdc4d",
    "native_backward": "58426ca9cf30697d67c49f7e5db69e0a8312f0e82529728267830b82b2472222",
    "candidate_forward": "ad8061d6eda8f63363597f94708ba1b7d3d3eafc9aec1a5eb448eda7ada36189",
    "candidate_backward": "beb0fc212ffe3e83ae9db10bc9d83e43e867b5dbafbb388062c95757830d9846"
   },
   1.5628756955266e-07,
   0,
   true
  ],
  [
   1,
   "target",
   [
    1,
    14,
    4096,
    4096
   ],
   "torch.bfloat16",
   63102,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     234881024,
     false,
     0,
     2.9802322387695312e-08,
     4.805378670147354e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     3.0517578125e-05,
     3.5450853536185924e-05,
     0.001,
     true
    ],
    [
     234881024,
     false,
     0,
     4.1102273307114245e-08,
     5.29302348207935e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.1102273307114245e-08,
     5.307895904480598e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     3.0517578125e-05,
     3.6359217721770865e-05,
     0.001,
     true
    ],
    [
     234881024,
     false,
     0,
     3.814697265625e-06,
     1.2115806364144605e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "6a87ad940421e31601a8e3bef51e54c059e35c5aa89053f0736360f8a680c67e",
    "upstream": "1239c0d4b07b90008908b59ee22e9ba1f9b66fc4eecd1089cbb321d592e5caeb",
    "native_forward": "14d15e52cb2c66afc914b29caf824b28cfac78da76d878fe82ec6ce3ef325aa2",
    "native_backward": "9441fc1d02f4712edabea594b80519797bb6d4039263ff61467b0bb739ee7ff1",
    "candidate_forward": "943a87af88e7ed970a3df89c21b7fb5da10cceba190f07ab514d1fd368b52022",
    "candidate_backward": "fde9a53a273152945cd1d18f40f601f8d2602317da4f940b538ebdce6ba2ad32"
   },
   1.6827834770083427e-07,
   0,
   true
  ],
  [
   1,
   "target",
   [
    1,
    14,
    4096,
    4096
   ],
   "torch.float32",
   63102,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     234881024,
     false,
     0,
     4.656612873077393e-10,
     2.093322815419162e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     1.4551915228366852e-10,
     2.7478293216517398e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.5011176674769615e-08,
     5.3065916719655186e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     4.5011176674769615e-08,
     5.30351408785289e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     5.7362278468864325e-09,
     7.325063070507742e-08,
     1e-05,
     true
    ],
    [
     234881024,
     false,
     0,
     5.7362278468864325e-09,
     7.32240493549963e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "ef4b6172ea7acc286ff7c3fcc33a6da34aa4913754d235503633297588107607",
    "upstream": "1239c0d4b07b90008908b59ee22e9ba1f9b66fc4eecd1089cbb321d592e5caeb",
    "native_forward": "8244f6eefc5b62349b0ce147eeea01b4081a0b4716e0994e058fc4fd48377c77",
    "native_backward": "f68df1923cb0fa68e6a6732bdc86d8f4367d64f2078d26a1b05b5a2d143fa617",
    "candidate_forward": "a2b576e0c1b4cef10c981bf10d4231345774a33814fb1065130689a945178709",
    "candidate_backward": "8fa54a81ab63c6a5e9c7ad8662e2825cffbfcb9a73c3c981aad3e2268616a28f"
   },
   1.646694727241993e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    1,
    1
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "6e602b749c4fc45c98bc354905c0d1471fa65371932209cf88fc3bafa721d60c",
    "upstream": "2d964498318eb7e9e4e9d7be97b9dfdf54a13b9bf2491c2e46304f3618e56bcf",
    "native_forward": "f46913286d895f62d646314f400fc3d7a543cf34eb4ae56df87f43dc4582a36b",
    "native_backward": "15ec7bf0b50732b49f8228e07d24365338f9e3ab994b00af08e5a3bffe55fd8b",
    "candidate_forward": "f46913286d895f62d646314f400fc3d7a543cf34eb4ae56df87f43dc4582a36b",
    "candidate_backward": "15ec7bf0b50732b49f8228e07d24365338f9e3ab994b00af08e5a3bffe55fd8b"
   },
   0.0,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    2,
    2
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     24,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     24,
     false,
     0,
     3.272206372528785e-08,
     1.9141224837819314e-08,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     3.272206372528785e-08,
     1.9141224837819314e-08,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     24,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "cff1f7beb1042e186caf014a842ffeac5997a872ceda7751a398bdd187110538",
    "upstream": "14a246fc9d7ee38c9c5c2b8930dc100e0f26da450929a8b8f60eb2fb404ea9ab",
    "native_forward": "9575bc814606d65c3a48a26887ac9204110c88109ab86b68e7277b5638b829cf",
    "native_backward": "ec57cd28f04f21c0c234bab08bdb20a2f8882dc0038c8e69d0582d482f64bd41",
    "candidate_forward": "9575bc814606d65c3a48a26887ac9204110c88109ab86b68e7277b5638b829cf",
    "candidate_backward": "ec57cd28f04f21c0c234bab08bdb20a2f8882dc0038c8e69d0582d482f64bd41"
   },
   5.960464477539063e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    3,
    3
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     54,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     54,
     false,
     0,
     3.0784134352757064e-08,
     2.6485539816327432e-08,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     3.0784134352757064e-08,
     2.6485539816327432e-08,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     54,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "f2044477d5f36d599dbc74a1af8123e91e439fb46c89c09d981425a1bde221dd",
    "upstream": "26d5b576c00fb9cc62588aa49b91cb8f93708d4734a8320376862797556b737b",
    "native_forward": "49ae19a988f583cd0b038c2abf9273ca9364a2bd2a858ac377381ab39d669d08",
    "native_backward": "3183435be53227364295fd4b1bfd8bcb496f1ef19ab7d82970673e28539a9259",
    "candidate_forward": "49ae19a988f583cd0b038c2abf9273ca9364a2bd2a858ac377381ab39d669d08",
    "candidate_backward": "3183435be53227364295fd4b1bfd8bcb496f1ef19ab7d82970673e28539a9259"
   },
   5.960464477539063e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     4.127988639801572e-08,
     4.512635312316063e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.127988639801572e-08,
     4.512635312316063e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "3e38f369fb78344a186f2413ecc1d90b5b0892f1302dc888279524c8e7deb667",
    "upstream": "a0ce4e9c30f09ec1299c11efc45e811848edb4ddcdef8942dd34c34f5797e863",
    "native_forward": "1ac3343b11fd794173ae12fb5780aa380a3d10d6f14ecd099d9d81bf6fd7fbb4",
    "native_backward": "44d903e4daea98e85ed15d64d0ba752d8e1442c0be5bb7159c9432a6c49e165c",
    "candidate_forward": "1ac3343b11fd794173ae12fb5780aa380a3d10d6f14ecd099d9d81bf6fd7fbb4",
    "candidate_backward": "44d903e4daea98e85ed15d64d0ba752d8e1442c0be5bb7159c9432a6c49e165c"
   },
   1.0803341865539551e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    127,
    127
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     96774,
     false,
     0,
     3.725290298461914e-09,
     2.062381554774788e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     96774,
     false,
     0,
     4.15412066967491e-08,
     4.893066957853847e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     4.15412066967491e-08,
     4.8773526914231365e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     7.62939453125e-06,
     1.858247258690155e-05,
     0.001,
     true
    ],
    [
     96774,
     false,
     0,
     7.62939453125e-06,
     1.858247258690155e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "f93542f4ff88ee2e655a3f0260a6da128ddde6c3aee65057827e09c0f975cc32",
    "upstream": "b65830f8470cbe97e4c0b6c33a2bb3dea1bdc987c9a6e1cc8faf69720da30004",
    "native_forward": "211cb31d8222cb1d1959c1e431a7d5b83479e79d565e2087ede51a218532d7ce",
    "native_backward": "a78f024e8cf8086b40d3ec3e4e4ab7213aa2962d23e456fdfc326657571b8cd6",
    "candidate_forward": "3aa48958dad0eb29c9628fe4845794507ba0c97bbe973b48c5822be92a64d2d8",
    "candidate_backward": "a78f024e8cf8086b40d3ec3e4e4ab7213aa2962d23e456fdfc326657571b8cd6"
   },
   1.2014061212539673e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    255,
    255
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     390150,
     false,
     0,
     3.725290298461914e-09,
     2.546677396860773e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     1.9073486328125e-06,
     5.993230989489469e-06,
     0.001,
     true
    ],
    [
     390150,
     false,
     0,
     6.409386932304528e-08,
     5.1720863726755754e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     6.409386932304528e-08,
     5.1412324250657485e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     1.52587890625e-05,
     3.5248436372427734e-05,
     0.001,
     true
    ],
    [
     390150,
     false,
     0,
     1.52587890625e-05,
     3.5752357116750226e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "a34940a51f71a4eb50e2f295e12b136372acf007c8944a4dab9bec523c0b7bfd",
    "upstream": "b070bd39c21e86fc6d62afd1c39e5f4417d45474c257f7ce803a8187b317d20c",
    "native_forward": "8baecae3b1c343ce77c49273378da1e66fbe14d4efb1f75a69f19190a1542dfa",
    "native_backward": "5234aa69246c3002c22ffb2106d47b22f4b58f86b116cb3d3a3d1334c09fb321",
    "candidate_forward": "075b7c1fd2255bdd85ad0fd594800eb20c9e01a11bb06638eef56111c568c2f6",
    "candidate_backward": "1840fba6cc13a897129b16488a0f22d307ac23f07f4cffd262cd5ee67e2068dc"
   },
   1.434236764907837e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    257,
    257
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     396294,
     false,
     0,
     3.725290298461914e-09,
     2.6388380584919174e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     2.384185791015625e-07,
     5.308491005517751e-07,
     0.001,
     true
    ],
    [
     396294,
     false,
     0,
     3.580354557275456e-08,
     5.023615774896427e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     3.580354557275456e-08,
     5.0221313053979697e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     2.384185791015625e-07,
     5.29815313627235e-07,
     0.001,
     true
    ],
    [
     396294,
     false,
     0,
     1.4901161193847656e-08,
     3.311345710170219e-08,
     0.001,
     true
    ]
   ],
   {
    "input": "06e86e30ac8a8338241228f2682947f277806d5a847262d1558ab9b87528875f",
    "upstream": "753cd5ef01dfeaad3f23174097a62b537e7e23489140603e2bab214ba39db22e",
    "native_forward": "e8c6a3caa0bbff05f827676bad0c55ad5e1a89f15fbaecfbc6cb8efcbd0805e6",
    "native_backward": "0df397475bf1b2915158a9c2024e92b039a36f924611cb40cd6361033a5bf4a1",
    "candidate_forward": "efb8114334846b134f1c5f382f28d57f41151692ea557e5c35cbcbf0479098d5",
    "candidate_backward": "c6c7d83d179cc7c40d558519d4d685e066d65e3d9a887502d8088839b771c111"
   },
   1.30385160446167e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    1,
    1,
    2047,
    2047
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     4190209,
     false,
     0,
     1.862645149230957e-09,
     3.566419506317351e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     9.5367431640625e-07,
     5.561596047835388e-06,
     0.001,
     true
    ],
    [
     4190209,
     false,
     0,
     3.962185268768792e-08,
     5.217803791889426e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     3.962185268768792e-08,
     5.1742588559388476e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     1.9073486328125e-06,
     9.105341610375104e-06,
     0.001,
     true
    ],
    [
     4190209,
     false,
     0,
     1.9073486328125e-06,
     1.0193884577221881e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "7fb19dd077d46eee121231ee2ee6c7f0d44b3eef333b275903ab098484af3254",
    "upstream": "0a34a4942cf15536dbcf9a06515042c7f0a02f1dfbc2b3cab3f782e4cb6d11df",
    "native_forward": "79a0de0085a6fc720a554dffd6ab116e34c60be78a7db486eba012e9a8b002bc",
    "native_backward": "3acab9d6d6e61891a12492965927c5278c1ef462cd26e8b4c53dbdd082362919",
    "candidate_forward": "624f0e5c5019f68eeb4cc0c661babc7f611abf9c3ad3e80549718fac71190fa1",
    "candidate_backward": "2b5e720a3d8b5d74624e25bffedf8cbe71703fde7bdf544d21647cdfea039d25"
   },
   1.5192199498414993e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    1,
    1,
    2049,
    2049
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     4198401,
     false,
     0,
     4.656612873077393e-10,
     1.7278769281991917e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     4.76837158203125e-07,
     4.032124099761437e-06,
     0.001,
     true
    ],
    [
     4198401,
     false,
     0,
     2.6898947513842586e-08,
     5.302267426013357e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     2.6898947513842586e-08,
     5.2930602221630364e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     9.5367431640625e-07,
     8.315329421041288e-06,
     0.001,
     true
    ],
    [
     4198401,
     false,
     0,
     9.5367431640625e-07,
     8.63466757566927e-06,
     0.001,
     true
    ]
   ],
   {
    "input": "41827f15206567bc72eeadb75a8f4ef30937beff9a4e2320ea2bdfee08d4d801",
    "upstream": "87b7fa131bfc32ed1c4650b6cb84da7a31374a4547d578cc80a3aa679b2a64c4",
    "native_forward": "bcac6913c2569e980ea34d48098d770b6599d6480f01d372f34ca761ef96175a",
    "native_backward": "5a74d77484733a8fa0ffd10907984731e94c054d80c14cb361d2bbaee64f2e91",
    "candidate_forward": "ab52c255fa6705d4b4923da480d31b1e90de7628a7d1e389758a56cc54835798",
    "candidate_backward": "1bded689138d32886efa8ba6e8285d41fce5c00d9ca86a5cfdf856699fc0473d"
   },
   1.5390105545520782e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    1,
    1,
    4095,
    4095
   ],
   "torch.bfloat16",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     16769025,
     false,
     0,
     3.4924596548080444e-10,
     2.0907865334389825e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     4.76837158203125e-07,
     4.537915257182126e-06,
     0.001,
     true
    ],
    [
     16769025,
     false,
     0,
     1.933288917843612e-08,
     5.0899872696029735e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     1.933288917843612e-08,
     5.096742378242412e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     1.9073486328125e-06,
     1.2496518415473971e-05,
     0.001,
     true
    ],
    [
     16769025,
     false,
     0,
     1.9073486328125e-06,
     1.2321424409064232e-05,
     0.001,
     true
    ]
   ],
   {
    "input": "d5ef308725134a84e40d197f496cf83876ac1fdfc8a75a1b7e616c7833c76abe",
    "upstream": "4f3e30055534de9ff84bd1224c6e4ce4a65c0b30834516736522d7edc68d59a3",
    "native_forward": "c79ee07807adac1d739013eb4b0cdc5f87e13b58a8a2f094fe78e358b13f2f12",
    "native_backward": "82c40ec32b5b44880c8f152e8d4f03fc5d70c4ad6b79b205fe7cfeb45aae06d5",
    "candidate_forward": "99884ed4ddd72f820ffa26285d072aa8458ddc26eda3f84da933553213b20700",
    "candidate_backward": "95f5bccb3ffd3fbbe35c80dfea8b4a528a8a820787e10210f92807e7ae62b43e"
   },
   1.5459954738616943e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63104,
   "all-open",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     7.121071524118783e-09,
     5.5078654811627806e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     7.121071524118783e-09,
     5.5078654811627806e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "3b640d7de5c881ed6f691e4cddc9e9d296aef1a13c64dfb35cfaed3ead6afa3f",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "78ccbc250c8e1544b1d2a821ae6869d610f84c7ed4f4c5538bc078c7f77db374",
    "native_backward": "55990b69c814834c10305b7d6ce78e00206bb9277b4390973d31fde273b9f7a5",
    "candidate_forward": "78ccbc250c8e1544b1d2a821ae6869d610f84c7ed4f4c5538bc078c7f77db374",
    "candidate_backward": "55990b69c814834c10305b7d6ce78e00206bb9277b4390973d31fde273b9f7a5"
   },
   9.313225746154785e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63104,
   "all-masked",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     9.613652374174109e-10,
     2.980232235993973e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.613652374174109e-10,
     2.980232235993973e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "3b640d7de5c881ed6f691e4cddc9e9d296aef1a13c64dfb35cfaed3ead6afa3f",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "477a6b9ec23fd18a88caa77a9966a59dab7505a0133223d28cb8185bf4816e3f",
    "native_backward": "d49871a7e88daaf877eec1e123a4442c8acc2f1c992cf2c9cd6e60caf12335ec",
    "candidate_forward": "477a6b9ec23fd18a88caa77a9966a59dab7505a0133223d28cb8185bf4816e3f",
    "candidate_backward": "d49871a7e88daaf877eec1e123a4442c8acc2f1c992cf2c9cd6e60caf12335ec"
   },
   2.9802322387695312e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63104,
   "random-mask",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     1.1128720292519212e-08,
     5.7450048791246e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.1128720292519212e-08,
     5.7450048791246e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.4901161193847656e-08,
     5.686464438916746e-08,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     1.4901161193847656e-08,
     5.686464438916746e-08,
     0.001,
     true
    ]
   ],
   {
    "input": "3b640d7de5c881ed6f691e4cddc9e9d296aef1a13c64dfb35cfaed3ead6afa3f",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "c70787b7246e8f2df331471263f779101fd218756a0400093814e1f5442d0ff0",
    "native_backward": "8811e70054d9b35a24f38f5d044a5002e152f857540f7e9f76c2d243398a95a5",
    "candidate_forward": "c70787b7246e8f2df331471263f779101fd218756a0400093814e1f5442d0ff0",
    "candidate_backward": "8811e70054d9b35a24f38f5d044a5002e152f857540f7e9f76c2d243398a95a5"
   },
   9.685754776000977e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63104,
   "strided-mask",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     1.2135728387485667e-08,
     5.7648735694158225e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.2135728387485667e-08,
     5.7648735694158225e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "3b640d7de5c881ed6f691e4cddc9e9d296aef1a13c64dfb35cfaed3ead6afa3f",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "d462018ea4cddfc77f132e93e561ff3e64313d8558ee29a525dabee6ab1f8ff9",
    "native_backward": "2d4e7039876161d25b9ce87f51132e9e03094d4ee792564ae82d992002e56ead",
    "candidate_forward": "d462018ea4cddfc77f132e93e561ff3e64313d8558ee29a525dabee6ab1f8ff9",
    "candidate_backward": "2d4e7039876161d25b9ce87f51132e9e03094d4ee792564ae82d992002e56ead"
   },
   1.1548399925231934e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63105,
   "causal",
   1.0,
   "large-negative",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "c0117d6fa9629ddf11e97910492711946a6c0c35f1bacf85e0aa7f77b63b6e17",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "94ce1e3bd971720698c51825a5708c4e28ede19252c04e41fd3fb12d3dc2ddb8",
    "native_backward": "0f64be7a07f263e4467f4ecfab166ec04670fe6a50e8a5c17dbe040ae3749633",
    "candidate_forward": "94ce1e3bd971720698c51825a5708c4e28ede19252c04e41fd3fb12d3dc2ddb8",
    "candidate_backward": "0f64be7a07f263e4467f4ecfab166ec04670fe6a50e8a5c17dbe040ae3749633"
   },
   5.21540641784668e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63105,
   "causal",
   0.08838834764831843,
   "zeros",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "d49871a7e88daaf877eec1e123a4442c8acc2f1c992cf2c9cd6e60caf12335ec",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "native_backward": "26a10aa508e7f9fe26976ef3bfbed44c567f3dc78ff9114ce0944409b1534f69",
    "candidate_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "candidate_backward": "26a10aa508e7f9fe26976ef3bfbed44c567f3dc78ff9114ce0944409b1534f69"
   },
   5.21540641784668e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63105,
   "causal",
   0.0,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "e34a384fcf42f7439a56b0a3680f89f7d84b241eed6efce3da533eab10732ff9",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "native_backward": "dfed62fd5ed2643c48baa3333e2ce6c4e448ccd1204b57f78d66501d2d440079",
    "candidate_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "candidate_backward": "dfed62fd5ed2643c48baa3333e2ce6c4e448ccd1204b57f78d66501d2d440079"
   },
   5.21540641784668e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63105,
   "causal",
   -0.25,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     5.960464477539063e-08,
     2.3154470250210007e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     3.0517578125e-05,
     3.14594106217123e-05,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     4.956657762011929e-08,
     4.7384050420991063e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     5.358054766535503e-08,
     4.6265321853876494e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     3.0517578125e-05,
     3.1459405842459247e-05,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "e34a384fcf42f7439a56b0a3680f89f7d84b241eed6efce3da533eab10732ff9",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "8cceff1cce3edc3b1b860b7c530493a10ae3a94ce728ed5bc1d97303d4a50a9d",
    "native_backward": "01286d6dc15af865b4aceb9c7da20f72818de437b771298961ebf08ccc3296b6",
    "candidate_forward": "952ea86d8a893d7c2717582cb3dcd959569645565c57caa532ae3cf8f612c599",
    "candidate_backward": "a0abca5b17b9ec13c3070c1025c665525d6bbaadfcac53bd6279085e6f739a29"
   },
   1.0244548320770264e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.bfloat16",
   63105,
   "causal",
   0.08838834764831843,
   "random",
   true,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     5.315993223398152e-08,
     4.421124443907447e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     5.315993223398152e-08,
     4.421124443907447e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     0.001,
     true
    ]
   ],
   {
    "input": "e34a384fcf42f7439a56b0a3680f89f7d84b241eed6efce3da533eab10732ff9",
    "upstream": "18e156785d6c491434703f68d20be989626977657baf3c8751bce402b0c1c51e",
    "native_forward": "22b97cf3db6db278da28d369ebcfa36042d335fb11e20472029a95cfcd23960f",
    "native_backward": "142ae11683c71dc9e219fe3a80654bb14441ba5535ef62abc8bf68a9df28fbcd",
    "candidate_forward": "22b97cf3db6db278da28d369ebcfa36042d335fb11e20472029a95cfcd23960f",
    "candidate_backward": "142ae11683c71dc9e219fe3a80654bb14441ba5535ef62abc8bf68a9df28fbcd"
   },
   1.0058283805847168e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    1,
    1
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     6,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ]
   ],
   {
    "input": "97e4790a364624f7974c756f9de7df2d76952d0f320d1ff278043f1939deb4ff",
    "upstream": "2d964498318eb7e9e4e9d7be97b9dfdf54a13b9bf2491c2e46304f3618e56bcf",
    "native_forward": "f46913286d895f62d646314f400fc3d7a543cf34eb4ae56df87f43dc4582a36b",
    "native_backward": "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0",
    "candidate_forward": "f46913286d895f62d646314f400fc3d7a543cf34eb4ae56df87f43dc4582a36b",
    "candidate_backward": "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0"
   },
   0.0,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    2,
    2
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     24,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     3.569689632776374e-08,
     2.0396721056285102e-08,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     3.569689632776374e-08,
     2.0396721056285102e-08,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     3.289413311180489e-09,
     6.33053948733173e-08,
     1e-05,
     true
    ],
    [
     24,
     false,
     0,
     3.289413311180489e-09,
     6.33053948733173e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "e3795b488138dd2ab723864fa6ae0d3ec2b30e0df049666dc452c8e01902bac4",
    "upstream": "14a246fc9d7ee38c9c5c2b8930dc100e0f26da450929a8b8f60eb2fb404ea9ab",
    "native_forward": "d41c386d1bb2b638d43928a525fcc91342d3765e3247eb4f385dab2dc8915804",
    "native_backward": "2dc879e04ea129375728d4c95cedd72cd86422d9f27663451225761b74f760ef",
    "candidate_forward": "d41c386d1bb2b638d43928a525fcc91342d3765e3247eb4f385dab2dc8915804",
    "candidate_backward": "2dc879e04ea129375728d4c95cedd72cd86422d9f27663451225761b74f760ef"
   },
   2.9802322387695312e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    3,
    3
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     54,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     4.4220734851663224e-08,
     2.7966713410464728e-08,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     4.4220734851663224e-08,
     2.7966713410464728e-08,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     2.685563195437002e-09,
     5.1669630310457116e-08,
     1e-05,
     true
    ],
    [
     54,
     false,
     0,
     2.685563195437002e-09,
     5.1669630310457116e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "56b73b2b26e25644fa325bb527c28b2feeb162c08aa5ef83dad6dee1130af1af",
    "upstream": "26d5b576c00fb9cc62588aa49b91cb8f93708d4734a8320376862797556b737b",
    "native_forward": "e213ab1aa9eb6c7fb2cb4db9e64085acf83cd8b80eb346c8fcd0d0f028147bde",
    "native_backward": "244a07e7c38c29badfdf33e919c4fff12413ac48fc02832aa4c57575cb82032a",
    "candidate_forward": "e213ab1aa9eb6c7fb2cb4db9e64085acf83cd8b80eb346c8fcd0d0f028147bde",
    "candidate_backward": "244a07e7c38c29badfdf33e919c4fff12413ac48fc02832aa4c57575cb82032a"
   },
   5.960464477539063e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.623894045430177e-08,
     4.345270130559963e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.623894045430177e-08,
     4.345270130559963e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     6.53414781830175e-09,
     7.457748853466223e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     6.53414781830175e-09,
     7.457748853466223e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "0c21cac1b425f49ac893a40723bed000efeffe15e3b2e4a42792b13c916bef2f",
    "upstream": "a0ce4e9c30f09ec1299c11efc45e811848edb4ddcdef8942dd34c34f5797e863",
    "native_forward": "13722a83fdc848dcd8597fa9cd9aabf6a1f419cf5ae47d4e55405d74fa79e372",
    "native_backward": "13491b793ac473dd49d74f5f914788af12531a3bf4f37d97a415d60aaafec662",
    "candidate_forward": "13722a83fdc848dcd8597fa9cd9aabf6a1f419cf5ae47d4e55405d74fa79e372",
    "candidate_backward": "13491b793ac473dd49d74f5f914788af12531a3bf4f37d97a415d60aaafec662"
   },
   1.1548399925231934e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    127,
    127
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     96774,
     false,
     0,
     3.725290298461914e-09,
     1.9975713310104676e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     9.313225746154785e-10,
     3.028928598025468e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     3.5059435132822614e-08,
     4.856882737853011e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     3.5059435132822614e-08,
     4.852829541202915e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     3.568572371631662e-09,
     7.289135270559189e-08,
     1e-05,
     true
    ],
    [
     96774,
     false,
     0,
     3.568572371631662e-09,
     7.291605463581857e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "5a9f6e3f5ea82e7136235671f7b96d201352a0c8236aaf7d8d9569479c84d311",
    "upstream": "b65830f8470cbe97e4c0b6c33a2bb3dea1bdc987c9a6e1cc8faf69720da30004",
    "native_forward": "c56bb1fd873604476f887f5da252d79056ea33436418295f84fcb1a0cd0cdbe7",
    "native_backward": "ab6ae51538215060ed143e72e6f850e14b34ddf4ac1bebce85463b0a5a82e09e",
    "candidate_forward": "4e16272d7974bc0bdcc3e7e14a8e1af13b1326cbf163ecb9aee57c8ab2b709b9",
    "candidate_backward": "64f8dc348b9388bfe1d8a4f10a61916d7abf889f0b0ea67e632d99bbe5c4b8bb"
   },
   1.5832483768463135e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    255,
    255
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     390150,
     false,
     0,
     3.725290298461914e-09,
     2.627396427656501e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     6.984919309616089e-10,
     3.799797733729154e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     4.031189537734292e-08,
     5.018001317495489e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     4.031189537734292e-08,
     5.002096734627235e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     3.3278994421293007e-09,
     7.183678493388644e-08,
     1e-05,
     true
    ],
    [
     390150,
     false,
     0,
     3.3278994421293007e-09,
     7.182632647554721e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "22060da8660015e1e3357416bb8d9c4b0350eaeb601462245b7434d6c9d731c7",
    "upstream": "b070bd39c21e86fc6d62afd1c39e5f4417d45474c257f7ce803a8187b317d20c",
    "native_forward": "fd9d11a2c9cc72f68b8c2caf781c0057905dfa44e11508817dc73decde4c916d",
    "native_backward": "c5f2cf180d7e4afd131cf3c5feaaca80f5f8a27ab070bb9e5f23a5a27b628f1f",
    "candidate_forward": "95dac3bcf60705af0a787d5bfe0e601e979b2bd20dea11a7d2fb1b5ab7fbf68a",
    "candidate_backward": "140854b15dd6b3d8091c58b08573e19c192d7669ea26af003edc4afc0f4e0a59"
   },
   1.2293457984924316e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    257,
    257
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     396294,
     false,
     0,
     3.725290298461914e-09,
     2.703644932059261e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     6.984919309616089e-10,
     3.8853535699831246e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     4.993713187406712e-08,
     4.991066638856359e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     4.993713187406712e-08,
     4.989444619128417e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     7.612985658544957e-09,
     7.337705163512448e-08,
     1e-05,
     true
    ],
    [
     396294,
     false,
     0,
     7.612985658544957e-09,
     7.339394960915919e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "055fab1dbfa19ef68fe1f3560fffc0a93f78676960f91e1d15544251e6d72be0",
    "upstream": "753cd5ef01dfeaad3f23174097a62b537e7e23489140603e2bab214ba39db22e",
    "native_forward": "c3823718861d695d2f6a16dfadb2e8e97a1e1862e1c8facc4a90e49b005a8bb5",
    "native_backward": "2dca9f65d5de4253e029a40a4b6d9b8c62d19d8b164d625831ca13f87ad50c11",
    "candidate_forward": "45ff216f31fe0bfd9b2c35fd4aa84030d91afd25aec0a2a65413d4128a5cff65",
    "candidate_backward": "a6aa4663154100981006453ff3347e4fd1c281ed229884e4530dc55280599d89"
   },
   1.1920928955078125e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    1,
    1,
    2047,
    2047
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     4190209,
     false,
     0,
     3.725290298461914e-09,
     3.747845830738651e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     6.984919309616089e-10,
     4.909472447309809e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     1.936420404580197e-08,
     5.291629786251054e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     1.936420404580197e-08,
     5.277109211377292e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     3.759538159953113e-09,
     7.500797484583873e-08,
     1e-05,
     true
    ],
    [
     4190209,
     false,
     0,
     3.759538159953113e-09,
     7.506176227290741e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "133b34e22bb48d8539bfac1862a7fa6ab86ab72688670e6407d9a4ba4431eb3c",
    "upstream": "0a34a4942cf15536dbcf9a06515042c7f0a02f1dfbc2b3cab3f782e4cb6d11df",
    "native_forward": "829a714bc9e376f462fb741e3907e2d4f6df2fda041f14f7a42a50a9913f170f",
    "native_backward": "2ef322173f0a1abdf434345210ed392039a710f12cb6ee954a27aa11b5c7e02a",
    "candidate_forward": "eb69ef16d275805a26ab2abc706aa4942a95e5ef58515b53a1d625ad351e24b0",
    "candidate_backward": "0cd2fca182c63d2fd5f7d664f542c2a6fa1ad353273f6fedfb14ae485bade7c8"
   },
   1.4193938113749027e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    1,
    1,
    2049,
    2049
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     4198401,
     false,
     0,
     3.4924596548080444e-10,
     1.5868487554703808e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     1.1641532182693481e-10,
     2.0543491438466058e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     2.5684546212101367e-08,
     5.319121273569074e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     2.5684546212101367e-08,
     5.321010761122377e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     3.7701664776235155e-09,
     7.482357332318883e-08,
     1e-05,
     true
    ],
    [
     4198401,
     false,
     0,
     3.7701664776235155e-09,
     7.483414391251959e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "ca1ff011c0567f8d7964969a081615f97c0624c336d7c62d583281efa69a7d9c",
    "upstream": "87b7fa131bfc32ed1c4650b6cb84da7a31374a4547d578cc80a3aa679b2a64c4",
    "native_forward": "ce56a9439a59dde98c69d604fe6f104c08d4be146447dc02cfc9088e33110557",
    "native_backward": "af4789b8348ab36020da4fe4aad86dc6aac0764df5484cc71a76c207d560f32e",
    "candidate_forward": "d4635f644066f6cf06fa99a49c0529880cb2bf7bccaffcba1f431f1ff475d79c",
    "candidate_backward": "e846481b8dd80b409f6aa80b8a9cdd1891c542a674023f4f14e1c505a41edea8"
   },
   1.2962846085429192e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    1,
    1,
    4095,
    4095
   ],
   "torch.float32",
   63103,
   "causal",
   0.08838834764831843,
   "random",
   false,
   [
    [
     16769025,
     false,
     0,
     4.656612873077393e-10,
     2.140361109160751e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     1.1641532182693481e-10,
     2.6674458599638703e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     2.9149262559258204e-08,
     5.4062414696177244e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     2.9149262559258204e-08,
     5.393988722562874e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     3.934552843254391e-09,
     7.126396762106084e-08,
     1e-05,
     true
    ],
    [
     16769025,
     false,
     0,
     3.934552843254391e-09,
     7.115964050469328e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "9b54860febae149f30424c33d1bdbb2b3142a4c6bba2a0dc5118483a91ab806f",
    "upstream": "4f3e30055534de9ff84bd1224c6e4ce4a65c0b30834516736522d7edc68d59a3",
    "native_forward": "f2de3935f05cb7e7469bdb2d1440f35ae2c2ac19eb2c963ce87a7a9e69da4432",
    "native_backward": "ce04da4673c5622cda5fb6e5e9c0b0323c8551f620ba4301bdd8d59b6afa9272",
    "candidate_forward": "d280d0f1c424123181f0d6ae5e946c82e7d88b0c5c5d4cc13eea0bd47d4bfd00",
    "candidate_backward": "fe17efbcdc4c9382ede5e25ff30d6e7ddda01e29f8d7055a386a5f6ec941af3f"
   },
   1.2996315490454435e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63104,
   "all-open",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     6.539937660865469e-09,
     5.513767966299546e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     6.539937660865469e-09,
     5.513767966299546e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.6156578592357862e-09,
     7.124098973620619e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.6156578592357862e-09,
     7.124098973620619e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "9eca1b7249e56a01756880dc7f06561aec9f1684af14837396b7618a5e2a98e7",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "5ce81a4bcbb6dcc020d7c5589e33270c817fa236dfd913d3e03445da4c3e4af9",
    "native_backward": "bc06df06b58bd3146c61146b12ca35834448a1ba9192f76fa5e345d443c92043",
    "candidate_forward": "5ce81a4bcbb6dcc020d7c5589e33270c817fa236dfd913d3e03445da4c3e4af9",
    "candidate_backward": "bc06df06b58bd3146c61146b12ca35834448a1ba9192f76fa5e345d443c92043"
   },
   8.568167686462402e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63104,
   "all-masked",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.613652374174109e-10,
     2.980232235993973e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.613652374174109e-10,
     2.980232235993973e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ]
   ],
   {
    "input": "9eca1b7249e56a01756880dc7f06561aec9f1684af14837396b7618a5e2a98e7",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "477a6b9ec23fd18a88caa77a9966a59dab7505a0133223d28cb8185bf4816e3f",
    "native_backward": "dbb056b04cf7e09668b449f9280e43ac832356e3fb4bc3590ed5a3f017dc7cf7",
    "candidate_forward": "477a6b9ec23fd18a88caa77a9966a59dab7505a0133223d28cb8185bf4816e3f",
    "candidate_backward": "dbb056b04cf7e09668b449f9280e43ac832356e3fb4bc3590ed5a3f017dc7cf7"
   },
   2.9802322387695312e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63104,
   "random-mask",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.1345216779923284e-08,
     5.788858753768595e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.1345216779923284e-08,
     5.788858753768595e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.7910227884110341e-09,
     7.35516604833963e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.7910227884110341e-09,
     7.35516604833963e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "9eca1b7249e56a01756880dc7f06561aec9f1684af14837396b7618a5e2a98e7",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "5250aec63c3ec0835b3eb998450904b4c8e097d4b768b492cf27c52bea3ac803",
    "native_backward": "e669c646516fc7044388e7e1bd8d74988954867ab075f06818ad37f72c49addf",
    "candidate_forward": "5250aec63c3ec0835b3eb998450904b4c8e097d4b768b492cf27c52bea3ac803",
    "candidate_backward": "e669c646516fc7044388e7e1bd8d74988954867ab075f06818ad37f72c49addf"
   },
   1.0803341865539551e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63104,
   "strided-mask",
   0.08838834764831843,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.0911934647528376e-08,
     5.9313710461098403e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.0911934647528376e-08,
     5.9313710461098403e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     2.274017656470062e-09,
     7.437920816800971e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     2.274017656470062e-09,
     7.437920816800971e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "9eca1b7249e56a01756880dc7f06561aec9f1684af14837396b7618a5e2a98e7",
    "upstream": "f2f359fc939d88cb6958de6709a5e1054c9d874fd07d51a4c0df3275cb24f790",
    "native_forward": "3ab68c4334cbc2b19800ded5a11073e6be98880cc60b6bd13d3083d9bc63eacf",
    "native_backward": "355bc9d24adddc79ad524be6f15fba9413f6a7c15f6f269120e69024b23847ac",
    "candidate_forward": "3ab68c4334cbc2b19800ded5a11073e6be98880cc60b6bd13d3083d9bc63eacf",
    "candidate_backward": "355bc9d24adddc79ad524be6f15fba9413f6a7c15f6f269120e69024b23847ac"
   },
   1.1920928955078125e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63105,
   "causal",
   1.0,
   "large-negative",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     8.766643092017468e-09,
     6.14927687905896e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     8.766643092017468e-09,
     6.14927687905896e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "ca8d88b595aa52d814b8686c9c5d68452c23ac0c2e066331269cd4e711505071",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "94ce1e3bd971720698c51825a5708c4e28ede19252c04e41fd3fb12d3dc2ddb8",
    "native_backward": "ec64099a42259eb7de473f26599dd2993d1418655252d3522f70b317127401ed",
    "candidate_forward": "94ce1e3bd971720698c51825a5708c4e28ede19252c04e41fd3fb12d3dc2ddb8",
    "candidate_backward": "ec64099a42259eb7de473f26599dd2993d1418655252d3522f70b317127401ed"
   },
   5.21540641784668e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63105,
   "causal",
   0.08838834764831843,
   "zeros",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.345741801137226e-09,
     5.386282923363045e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.345741801137226e-09,
     5.386282923363045e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "dbb056b04cf7e09668b449f9280e43ac832356e3fb4bc3590ed5a3f017dc7cf7",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "native_backward": "47678db904c9fdbb0e5f5ad805a60f4ea82d8a6363b6f28e0bd67e106bcf0d7f",
    "candidate_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "candidate_backward": "47678db904c9fdbb0e5f5ad805a60f4ea82d8a6363b6f28e0bd67e106bcf0d7f"
   },
   5.21540641784668e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63105,
   "causal",
   0.0,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     9.934107481068821e-09,
     2.1032052104304662e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ]
   ],
   {
    "input": "619bceb7cf51993999287904e9a9a0324bb1fbfcf0293a55a299c918da6ecc8f",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "native_backward": "6e2e536f6b068bdc2e5edbf1f3d3d57e2e56c72a2876dcd93d4a078dadc8633f",
    "candidate_forward": "11f2dc4db027a7dc13843ae010776f4cd225cb05623e6e491dbad06dbbacd117",
    "candidate_backward": "6e2e536f6b068bdc2e5edbf1f3d3d57e2e56c72a2876dcd93d4a078dadc8633f"
   },
   5.21540641784668e-08,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63105,
   "causal",
   -0.25,
   "random",
   false,
   [
    [
     5766,
     false,
     0,
     2.9802322387695312e-08,
     1.908662774645302e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     7.450580596923828e-09,
     3.009303014932518e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.034320033596828e-08,
     4.757451667412049e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.034320033596828e-08,
     4.596470680630464e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.447217118366595e-08,
     7.187394792396369e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     1.447217118366595e-08,
     7.01590177081028e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "619bceb7cf51993999287904e9a9a0324bb1fbfcf0293a55a299c918da6ecc8f",
    "upstream": "c1032635503b572bd17f7f434d25bec29a57a44b993e28d9869de857c3edb311",
    "native_forward": "40923b5a902f5dc5e39641d2cf07c7353f5421b7df4745800713473ab05c67c8",
    "native_backward": "f07bb50e115d3f20ee3fa4b124df8ff553ffba9c34d2c4739cfe3229763e552e",
    "candidate_forward": "f2bb587407fc7532b60d6bf258edacebc685057ccfcae86c338a45fda2fb4e0b",
    "candidate_backward": "2bda107648ec6e2a7f7326031abab22e96980c44d5bdfc2db6556a9aba85eb14"
   },
   1.2479722499847412e-07,
   0,
   true
  ],
  [
   1,
   "boundary",
   [
    2,
    3,
    31,
    31
   ],
   "torch.float32",
   63105,
   "causal",
   0.08838834764831843,
   "random",
   true,
   [
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     0.0,
     0.0,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     2.8564231702166154e-08,
     4.2374034187812844e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     2.8564231702166154e-08,
     4.2374034187812844e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.765884228929984e-09,
     6.681776809527146e-08,
     1e-05,
     true
    ],
    [
     5766,
     false,
     0,
     4.765884228929984e-09,
     6.681776809527146e-08,
     1e-05,
     true
    ]
   ],
   {
    "input": "619bceb7cf51993999287904e9a9a0324bb1fbfcf0293a55a299c918da6ecc8f",
    "upstream": "18e156785d6c491434703f68d20be989626977657baf3c8751bce402b0c1c51e",
    "native_forward": "cfb942e3d9a75d6b32cde65acaf53deaf8cccc49db47a7b502d89cdc7dbda881",
    "native_backward": "fc6816e93dc80bdabca44612cb97e20736c4c3c175a80396ff6391187fedbdbe",
    "candidate_forward": "cfb942e3d9a75d6b32cde65acaf53deaf8cccc49db47a7b502d89cdc7dbda881",
    "candidate_backward": "fc6816e93dc80bdabca44612cb97e20736c4c3c175a80396ff6391187fedbdbe"
   },
   1.043081283569336e-07,
   0,
   true
  ]
 ],
 "negative_controls": [
  {
   "device": 0,
   "controls": [
    {
     "name": "CPU-input",
     "rejected": true
    },
    {
     "name": "FP16-input",
     "rejected": true
    },
    {
     "name": "FP64-input",
     "rejected": true
    },
    {
     "name": "noncontiguous-input",
     "rejected": true
    },
    {
     "name": "nonsquare",
     "rejected": true
    },
    {
     "name": "empty",
     "rejected": true
    },
    {
     "name": "NaN-scale",
     "rejected": true
    },
    {
     "name": "infinite-scale",
     "rejected": true
    },
    {
     "name": "nonboolean-mask",
     "rejected": true
    },
    {
     "name": "CPU-mask",
     "rejected": true
    },
    {
     "name": "invalid-broadcast",
     "rejected": true
    },
    {
     "name": "zero-forward",
     "rejected": true
    },
    {
     "name": "zero-gradient",
     "rejected": true
    },
    {
     "name": "reversed-gradient",
     "rejected": true
    },
    {
     "name": "corrupt-one-element",
     "rejected": true
    }
   ]
  },
  {
   "device": 1,
   "controls": [
    {
     "name": "CPU-input",
     "rejected": true
    },
    {
     "name": "FP16-input",
     "rejected": true
    },
    {
     "name": "FP64-input",
     "rejected": true
    },
    {
     "name": "noncontiguous-input",
     "rejected": true
    },
    {
     "name": "nonsquare",
     "rejected": true
    },
    {
     "name": "empty",
     "rejected": true
    },
    {
     "name": "NaN-scale",
     "rejected": true
    },
    {
     "name": "infinite-scale",
     "rejected": true
    },
    {
     "name": "nonboolean-mask",
     "rejected": true
    },
    {
     "name": "CPU-mask",
     "rejected": true
    },
    {
     "name": "invalid-broadcast",
     "rejected": true
    },
    {
     "name": "zero-forward",
     "rejected": true
    },
    {
     "name": "zero-gradient",
     "rejected": true
    },
    {
     "name": "reversed-gradient",
     "rejected": true
    },
    {
     "name": "corrupt-one-element",
     "rejected": true
    }
   ]
  }
 ],
 "paired_timing_details": [
  {
   "device": 0,
   "seed": 63100,
   "round": "primary",
   "cases": [
    {
     "shape": [
      1,
      14,
      2048,
      2048
     ],
     "dtype": "torch.bfloat16",
     "seed": 63100,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.01171090453863144,
        0.01133331935852766
       ],
       "candidate_seconds": [
        0.0028489381074905396,
        0.002629866823554039
       ],
       "throughput_ratio": 4.206067598169722
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011384954676032066,
        0.01145703624933958
       ],
       "candidate_seconds": [
        0.002751455642282963,
        0.0028125373646616936
       ],
       "throughput_ratio": 4.105323442510008
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011368476785719395,
        0.011331752873957157
       ],
       "candidate_seconds": [
        0.002660481259226799,
        0.0025993622839450836
       ],
       "throughput_ratio": 4.315761385934203
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011419035494327545,
        0.01134618278592825
       ],
       "candidate_seconds": [
        0.0028216252103447914,
        0.0028080688789486885
       ],
       "throughput_ratio": 4.0437753666847644
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011364282108843327,
        0.011281609535217285
       ],
       "candidate_seconds": [
        0.0027128132060170174,
        0.002587178722023964
       ],
       "throughput_ratio": 4.272816251709111
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011385981924831867,
        0.01143903099000454
       ],
       "candidate_seconds": [
        0.0026466213166713715,
        0.002766380086541176
       ],
       "throughput_ratio": 4.2167018285437825
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011351736262440681,
        0.011333138681948185
       ],
       "candidate_seconds": [
        0.002643323503434658,
        0.002600565552711487
       ],
       "throughput_ratio": 4.325963936594133
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011386371217668056,
        0.011309679597616196
       ],
       "candidate_seconds": [
        0.002786746248602867,
        0.002659830264747143
       ],
       "throughput_ratio": 4.167030566752225
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011309364810585976,
        0.011300574988126755
       ],
       "candidate_seconds": [
        0.002775474451482296,
        0.002537800930440426
       ],
       "throughput_ratio": 4.255367579033866
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011324839666485786,
        0.01141305174678564
       ],
       "candidate_seconds": [
        0.002669883891940117,
        0.0027818335220217705
       ],
       "throughput_ratio": 4.170775865058509
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011318104341626167,
        0.011277706362307072
       ],
       "candidate_seconds": [
        0.0027450881898403168,
        0.0025636497884988785
       ],
       "throughput_ratio": 4.256343182905063
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011337844654917717,
        0.011477711610496044
       ],
       "candidate_seconds": [
        0.0027449168264865875,
        0.0028161518275737762
       ],
       "throughput_ratio": 4.102728753178616
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.01131769735366106,
        0.011294491589069366
       ],
       "candidate_seconds": [
        0.002704160287976265,
        0.002607869915664196
       ],
       "throughput_ratio": 4.256788473686342
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011400841176509857,
        0.011380605399608612
       ],
       "candidate_seconds": [
        0.0027731461450457573,
        0.0027747778221964836
       ],
       "throughput_ratio": 4.1063011516797445
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.01129720639437437,
        0.011269094422459602
       ],
       "candidate_seconds": [
        0.00273042730987072,
        0.002620106562972069
       ],
       "throughput_ratio": 4.217579283325663
      }
     ],
     "native_median_seconds": 0.011342013720422983,
     "candidate_median_seconds": 0.0027216202579438686,
     "geomean_paired_ratio": 4.200506077653176,
     "every_ABBA_round_faster": true,
     "native_CV": 0.007334772578403405,
     "candidate_CV": 0.03195834915894403,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 1077936128
    },
    {
     "shape": [
      1,
      14,
      2048,
      2048
     ],
     "dtype": "torch.float32",
     "seed": 63100,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.00951372180134058,
        0.00932726263999939
       ],
       "candidate_seconds": [
        0.0033295396715402603,
        0.0031585693359375
       ],
       "throughput_ratio": 2.9039253840564503
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009066365659236908,
        0.009415021166205406
       ],
       "candidate_seconds": [
        0.003275499679148197,
        0.003282698802649975
       ],
       "throughput_ratio": 2.8180584769942736
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.00942879170179367,
        0.009362520650029182
       ],
       "candidate_seconds": [
        0.0032863160595297813,
        0.0032072262838482857
       ],
       "throughput_ratio": 2.8938461256028782
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009413976222276688,
        0.009351124055683613
       ],
       "candidate_seconds": [
        0.00328864436596632,
        0.003300054930150509
       ],
       "throughput_ratio": 2.848073562716674
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009376941248774529,
        0.009342922829091549
       ],
       "candidate_seconds": [
        0.0032975217327475548,
        0.0031849388033151627
       ],
       "throughput_ratio": 2.8877713907744127
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009404320269823074,
        0.009422496892511845
       ],
       "candidate_seconds": [
        0.0032906243577599525,
        0.0031182197853922844
       ],
       "throughput_ratio": 2.9376306775147776
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009441285394132137,
        0.009311623871326447
       ],
       "candidate_seconds": [
        0.0032696807757019997,
        0.0032177316024899483
       ],
       "throughput_ratio": 2.89066089408749
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009398816153407097,
        0.009430141188204288
       ],
       "candidate_seconds": [
        0.003268980421125889,
        0.0032783737406134605
       ],
       "throughput_ratio": 2.8758116448995246
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.0094205467030406,
        0.009321576915681362
       ],
       "candidate_seconds": [
        0.0032764384523034096,
        0.003187502734363079
       ],
       "throughput_ratio": 2.8994885747695736
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009412015788257122,
        0.009405224584043026
       ],
       "candidate_seconds": [
        0.003269001841545105,
        0.003276018425822258
       ],
       "throughput_ratio": 2.8750469217216197
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009421016089618206,
        0.009290552698075771
       ],
       "candidate_seconds": [
        0.0033252574503421783,
        0.0031405938789248466
       ],
       "throughput_ratio": 2.893906437810895
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009400792419910431,
        0.009422718547284603
       ],
       "candidate_seconds": [
        0.0032863523811101913,
        0.003277764655649662
       ],
       "throughput_ratio": 2.8676379262863665
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009410744532942772,
        0.009320776909589767
       ],
       "candidate_seconds": [
        0.0032519707456231117,
        0.003202359192073345
       ],
       "throughput_ratio": 2.902163605416459
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009441894479095936,
        0.00942497793585062
       ],
       "candidate_seconds": [
        0.00312119722366333,
        0.0032607512548565865
       ],
       "throughput_ratio": 2.9562871713000898
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009468994103372097,
        0.009383948519825935
       ],
       "candidate_seconds": [
        0.0032495642080903053,
        0.0031576035544276237
       ],
       "throughput_ratio": 2.942476820021501
      }
     ],
     "native_median_seconds": 0.009407984558492899,
     "candidate_median_seconds": 0.0032693413086235523,
     "geomean_paired_ratio": 2.892649133872891,
     "every_ABBA_round_faster": true,
     "native_CV": 0.008164275747471483,
     "candidate_CV": 0.01811656150564929,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 1312817152
    },
    {
     "shape": [
      1,
      14,
      4096,
      4096
     ],
     "dtype": "torch.bfloat16",
     "seed": 63100,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04450158216059208,
        0.044420309364795685
       ],
       "candidate_seconds": [
        0.009748912416398525,
        0.009297901764512062
       ],
       "throughput_ratio": 4.668596578975844
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04453136958181858,
        0.0443279892206192
       ],
       "candidate_seconds": [
        0.010121503844857216,
        0.009544318541884422
       ],
       "throughput_ratio": 4.518466456930133
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04440526105463505,
        0.04431496001780033
       ],
       "candidate_seconds": [
        0.009656963869929314,
        0.009441843256354332
       ],
       "throughput_ratio": 4.64532787235383
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04388538375496864,
        0.04446171876043081
       ],
       "candidate_seconds": [
        0.00954147893935442,
        0.009753893129527569
       ],
       "throughput_ratio": 4.578667993548489
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.044364532455801964,
        0.04454283323138952
       ],
       "candidate_seconds": [
        0.0094556650146842,
        0.009300801903009415
       ],
       "throughput_ratio": 4.740091301700436
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04435931611806154,
        0.04426420573145151
       ],
       "candidate_seconds": [
        0.009739416651427746,
        0.00964542105793953
       ],
       "throughput_ratio": 4.571795914839554
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04429815895855427,
        0.04420599155128002
       ],
       "candidate_seconds": [
        0.009408737532794476,
        0.00939658097922802
       ],
       "throughput_ratio": 4.7063361598077895
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04425879009068012,
        0.04468206409364939
       ],
       "candidate_seconds": [
        0.009410073049366474,
        0.009792216122150421
       ],
       "throughput_ratio": 4.631783918568266
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04424323234707117,
        0.04419488552957773
       ],
       "candidate_seconds": [
        0.009449672885239124,
        0.00938329752534628
       ],
       "throughput_ratio": 4.695919759261168
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04431261122226715,
        0.044219049625098705
       ],
       "candidate_seconds": [
        0.009472555480897427,
        0.009507270529866219
       ],
       "throughput_ratio": 4.664513826267885
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.044311233796179295,
        0.04421322327107191
       ],
       "candidate_seconds": [
        0.009435673244297504,
        0.009330276399850845
       ],
       "throughput_ratio": 4.717291623707151
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04420525673776865,
        0.04422696027904749
       ],
       "candidate_seconds": [
        0.009435596875846386,
        0.00949651375412941
       ],
       "throughput_ratio": 4.671017339017588
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04436036944389343,
        0.044001735746860504
       ],
       "candidate_seconds": [
        0.00962923839688301,
        0.009476610459387302
       ],
       "throughput_ratio": 4.624871988441097
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04430327098816633,
        0.04432433471083641
       ],
       "candidate_seconds": [
        0.009603717364370823,
        0.009586287662386894
       ],
       "throughput_ratio": 4.618425350875324
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.044370874762535095,
        0.044340393505990505
       ],
       "candidate_seconds": [
        0.009536994621157646,
        0.009484556503593922
       ],
       "throughput_ratio": 4.663724198238024
      }
     ],
     "native_median_seconds": 0.04431378562003374,
     "candidate_median_seconds": 0.009490535128861666,
     "geomean_paired_ratio": 4.6474318670088826,
     "every_ABBA_round_faster": true,
     "native_CV": 0.003395468747161165,
     "candidate_CV": 0.017855726138467854,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 5540675584
    },
    {
     "shape": [
      1,
      14,
      4096,
      4096
     ],
     "dtype": "torch.float32",
     "seed": 63100,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036580780521035194,
        0.036512671038508415
       ],
       "candidate_seconds": [
        0.011574015952646732,
        0.011477215215563774
       ],
       "throughput_ratio": 3.1709131293752906
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03645092714577913,
        0.036407108418643475
       ],
       "candidate_seconds": [
        0.011580104008316994,
        0.011557912454009056
       ],
       "throughput_ratio": 3.1488453508126786
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03645288199186325,
        0.03641669172793627
       ],
       "candidate_seconds": [
        0.011499624699354172,
        0.011460433714091778
       ],
       "throughput_ratio": 3.173753847121111
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036444926634430885,
        0.03650884795933962
       ],
       "candidate_seconds": [
        0.011617605574429035,
        0.011457507498562336
       ],
       "throughput_ratio": 3.161578206052667
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03661159332841635,
        0.03646642714738846
       ],
       "candidate_seconds": [
        0.011378107592463493,
        0.011331337504088879
       ],
       "throughput_ratio": 3.2179571171864136
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036471910774707794,
        0.03633779287338257
       ],
       "candidate_seconds": [
        0.011683736927807331,
        0.011546437628567219
       ],
       "throughput_ratio": 3.1342727740334944
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03639921918511391,
        0.03640128672122955
       ],
       "candidate_seconds": [
        0.011486451141536236,
        0.011431052349507809
       ],
       "throughput_ratio": 3.1766333507833107
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03638477623462677,
        0.036352936178445816
       ],
       "candidate_seconds": [
        0.011465553194284439,
        0.011447726748883724
       ],
       "throughput_ratio": 3.1744784069973404
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03638674598187208,
        0.03646672237664461
       ],
       "candidate_seconds": [
        0.011355558410286903,
        0.011435753665864468
       ],
       "throughput_ratio": 3.1965456010209223
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036413817666471004,
        0.03633701242506504
       ],
       "candidate_seconds": [
        0.011471624486148357,
        0.011481249704957008
       ],
       "throughput_ratio": 3.169573861896923
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03637005854398012,
        0.036403270438313484
       ],
       "candidate_seconds": [
        0.011437810957431793,
        0.011379769071936607
       ],
       "throughput_ratio": 3.1893535111360363
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036470639519393444,
        0.03647029772400856
       ],
       "candidate_seconds": [
        0.011604584753513336,
        0.011653749272227287
       ],
       "throughput_ratio": 3.1361204617096097
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036503068171441555,
        0.03652196563780308
       ],
       "candidate_seconds": [
        0.011627343483269215,
        0.01153270062059164
       ],
       "throughput_ratio": 3.153061085797808
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036449321545660496,
        0.03645669016987085
       ],
       "candidate_seconds": [
        0.011598755605518818,
        0.011947558261454105
       ],
       "throughput_ratio": 3.0962813172125627
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03641982190310955,
        0.03651160094887018
       ],
       "candidate_seconds": [
        0.011725117452442646,
        0.011497985571622849
       ],
       "throughput_ratio": 3.140468471263414
      }
     ],
     "native_median_seconds": 0.036450124345719814,
     "candidate_median_seconds": 0.011492218356579542,
     "geomean_paired_ratio": 3.1625270669718764,
     "every_ABBA_round_faster": true,
     "native_CV": 0.0017753972596279864,
     "candidate_CV": 0.010740404369492383,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 6480199680
    }
   ]
  },
  {
   "device": 0,
   "seed": 63101,
   "round": "primary",
   "cases": [
    {
     "shape": [
      1,
      14,
      2048,
      2048
     ],
     "dtype": "torch.bfloat16",
     "seed": 63101,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011797877959907055,
        0.011413197964429855
       ],
       "candidate_seconds": [
        0.002979523502290249,
        0.0025937361642718315
       ],
       "throughput_ratio": 4.164721780970757
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.01146156806498766,
        0.011574073694646358
       ],
       "candidate_seconds": [
        0.002975884824991226,
        0.002969609573483467
       ],
       "throughput_ratio": 3.874470349437008
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011395355686545372,
        0.011470461264252663
       ],
       "candidate_seconds": [
        0.002994086593389511,
        0.002630479633808136
       ],
       "throughput_ratio": 4.0653476245386075
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.01144507434219122,
        0.01150259468704462
       ],
       "candidate_seconds": [
        0.002786152996122837,
        0.0029306570068001747
       ],
       "throughput_ratio": 4.014068863142674
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011469095945358276,
        0.011418591253459454
       ],
       "candidate_seconds": [
        0.002763117663562298,
        0.002693159505724907
       ],
       "throughput_ratio": 4.19474423470458
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011455662548542023,
        0.011534121818840504
       ],
       "candidate_seconds": [
        0.0028680218383669853,
        0.002846774645149708
       ],
       "throughput_ratio": 4.022852683151962
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011547685600817204,
        0.01146703865379095
       ],
       "candidate_seconds": [
        0.0030245091766119003,
        0.0026764804497361183
       ],
       "throughput_ratio": 4.036970028544165
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011600133031606674,
        0.011598481796681881
       ],
       "candidate_seconds": [
        0.0029245857149362564,
        0.0029098549857735634
       ],
       "throughput_ratio": 3.976150588944405
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011518515646457672,
        0.011394178494811058
       ],
       "candidate_seconds": [
        0.0029493076726794243,
        0.002643730491399765
       ],
       "throughput_ratio": 4.096645413296758
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011546026915311813,
        0.011571350507438183
       ],
       "candidate_seconds": [
        0.002881470136344433,
        0.0028831632807850838
       ],
       "throughput_ratio": 4.010207718335927
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011504145339131355,
        0.011372270993888378
       ],
       "candidate_seconds": [
        0.0028021233156323433,
        0.0027036871761083603
       ],
       "throughput_ratio": 4.154958905203289
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011476827785372734,
        0.011466852389276028
       ],
       "candidate_seconds": [
        0.002804654650390148,
        0.0028341757133603096
       ],
       "throughput_ratio": 4.068872211894069
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011469043791294098,
        0.011416046880185604
       ],
       "candidate_seconds": [
        0.002790769562125206,
        0.002743818797171116
       ],
       "throughput_ratio": 4.134921910324214
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011538044549524784,
        0.011548817157745361
       ],
       "candidate_seconds": [
        0.0027992473915219307,
        0.0028596781194210052
       ],
       "throughput_ratio": 4.07972532287728
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011514203622937202,
        0.011501839384436607
       ],
       "candidate_seconds": [
        0.002882278524339199,
        0.0027340464293956757
       ],
       "throughput_ratio": 4.09806113374335
      }
     ],
     "native_median_seconds": 0.01148933358490467,
     "candidate_median_seconds": 0.0028404751792550087,
     "geomean_paired_ratio": 4.065401410237658,
     "every_ABBA_round_faster": true,
     "native_CV": 0.0071113787751405436,
     "candidate_CV": 0.040500211302743006,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 1077936128
    },
    {
     "shape": [
      1,
      14,
      2048,
      2048
     ],
     "dtype": "torch.float32",
     "seed": 63101,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.00953590590506792,
        0.009347299113869667
       ],
       "candidate_seconds": [
        0.003280399367213249,
        0.0032032933086156845
       ],
       "throughput_ratio": 2.91241518731043
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009390033781528473,
        0.009399537928402424
       ],
       "candidate_seconds": [
        0.00329548679292202,
        0.0032701492309570312
       ],
       "throughput_ratio": 2.8618052602358253
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009448623284697533,
        0.009367518126964569
       ],
       "candidate_seconds": [
        0.0032489830628037453,
        0.0032005198299884796
       ],
       "throughput_ratio": 2.917456077535909
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009310292080044746,
        0.00940086878836155
       ],
       "candidate_seconds": [
        0.0030871592462062836,
        0.003265650011599064
       ],
       "throughput_ratio": 2.945336481717426
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009412745013833046,
        0.009370776824653149
       ],
       "candidate_seconds": [
        0.003256530500948429,
        0.0031664129346609116
       ],
       "throughput_ratio": 2.9244414226581483
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009447482414543629,
        0.009391963481903076
       ],
       "candidate_seconds": [
        0.0032430971041321754,
        0.003269360400736332
       ],
       "throughput_ratio": 2.892832065677654
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009447784163057804,
        0.009391674771904945
       ],
       "candidate_seconds": [
        0.0032494766637682915,
        0.003165380097925663
       ],
       "throughput_ratio": 2.9368479507542213
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009452144615352154,
        0.009399525821208954
       ],
       "candidate_seconds": [
        0.0032477369531989098,
        0.0032426975667476654
       ],
       "throughput_ratio": 2.9045313343237122
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009361373260617256,
        0.009376723319292068
       ],
       "candidate_seconds": [
        0.003218008205294609,
        0.0031785834580659866
       ],
       "throughput_ratio": 2.9293876436165753
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009452342055737972,
        0.009415468201041222
       ],
       "candidate_seconds": [
        0.0032939817756414413,
        0.0032837558537721634
       ],
       "throughput_ratio": 2.8684346077302005
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009424833580851555,
        0.009319114498794079
       ],
       "candidate_seconds": [
        0.0032592397183179855,
        0.003235097974538803
       ],
       "throughput_ratio": 2.886198557285119
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009457841515541077,
        0.009411602281033993
       ],
       "candidate_seconds": [
        0.0032533518970012665,
        0.003246830776333809
       ],
       "throughput_ratio": 2.902909771133193
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009425287134945393,
        0.009367440827190876
       ],
       "candidate_seconds": [
        0.00325214397162199,
        0.0032373303547501564
       ],
       "throughput_ratio": 2.895878312634005
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009406181052327156,
        0.009406915865838528
       ],
       "candidate_seconds": [
        0.003260933794081211,
        0.003269459120929241
       ],
       "throughput_ratio": 2.880852218696181
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009451650083065033,
        0.009367421269416809
       ],
       "candidate_seconds": [
        0.0032577775418758392,
        0.0032442426308989525
       ],
       "throughput_ratio": 2.8943421971037417
      }
     ],
     "native_median_seconds": 0.009403524920344353,
     "candidate_median_seconds": 0.0032492298632860184,
     "geomean_paired_ratio": 2.903482856958327,
     "every_ABBA_round_faster": true,
     "native_CV": 0.0048160746078884575,
     "candidate_CV": 0.013384536464949602,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 1312817152
    },
    {
     "shape": [
      1,
      14,
      4096,
      4096
     ],
     "dtype": "torch.bfloat16",
     "seed": 63101,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04425352532416582,
        0.04424480628222227
       ],
       "candidate_seconds": [
        0.009491105563938618,
        0.009394658729434013
       ],
       "throughput_ratio": 4.685980944781981
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.0442449739202857,
        0.04413564316928387
       ],
       "candidate_seconds": [
        0.009407834149897099,
        0.009405184537172318
       ],
       "throughput_ratio": 4.69784347529062
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04417306836694479,
        0.044048785232007504
       ],
       "candidate_seconds": [
        0.009397645480930805,
        0.009333327412605286
       ],
       "throughput_ratio": 4.709945078688195
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04429593309760094,
        0.044431501999497414
       ],
       "candidate_seconds": [
        0.00965268723666668,
        0.009669968858361244
       ],
       "throughput_ratio": 4.591886056489385
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04438336566090584,
        0.044363231398165226
       ],
       "candidate_seconds": [
        0.009560462087392807,
        0.009425451047718525
       ],
       "throughput_ratio": 4.6743391496376745
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044289628975093365,
        0.044497073628008366
       ],
       "candidate_seconds": [
        0.0094068618491292,
        0.009680632501840591
       ],
       "throughput_ratio": 4.651564053958223
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.044122910127043724,
        0.04442755598574877
       ],
       "candidate_seconds": [
        0.009672452695667744,
        0.009570231661200523
       ],
       "throughput_ratio": 4.601773041149858
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04432923998683691,
        0.04473407845944166
       ],
       "candidate_seconds": [
        0.009554141201078892,
        0.009924405254423618
       ],
       "throughput_ratio": 4.572380113153618
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.0441830912604928,
        0.044266846030950546
       ],
       "candidate_seconds": [
        0.009528810158371925,
        0.009362542070448399
       ],
       "throughput_ratio": 4.682033145118412
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04425783734768629,
        0.04422530345618725
       ],
       "candidate_seconds": [
        0.009555988945066929,
        0.009553136304020882
       ],
       "throughput_ratio": 4.63041293887052
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04420242737978697,
        0.043910834938287735
       ],
       "candidate_seconds": [
        0.0093629639595747,
        0.009341376833617687
       ],
       "throughput_ratio": 4.710845642319794
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04418424144387245,
        0.044171255081892014
       ],
       "candidate_seconds": [
        0.00937698781490326,
        0.00944023672491312
       ],
       "throughput_ratio": 4.695458479480239
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.044203463941812515,
        0.04422326199710369
       ],
       "candidate_seconds": [
        0.009454255923628807,
        0.009317122399806976
       ],
       "throughput_ratio": 4.710720993168454
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044181122444570065,
        0.044155897572636604
       ],
       "candidate_seconds": [
        0.009275846183300018,
        0.009433669038116932
       ],
       "throughput_ratio": 4.721502346361518
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04409403447061777,
        0.04410731792449951
       ],
       "candidate_seconds": [
        0.009631055407226086,
        0.00941083487123251
       ],
       "throughput_ratio": 4.631964112034418
      }
     ],
     "native_median_seconds": 0.04422428272664547,
     "candidate_median_seconds": 0.009436952881515026,
     "geomean_paired_ratio": 4.664342589550286,
     "every_ABBA_round_faster": true,
     "native_CV": 0.0033768977438703362,
     "candidate_CV": 0.014725826703756651,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 5540675584
    },
    {
     "shape": [
      1,
      14,
      4096,
      4096
     ],
     "dtype": "torch.float32",
     "seed": 63101,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036476452834904194,
        0.03639225009828806
       ],
       "candidate_seconds": [
        0.011519871652126312,
        0.011420507915318012
       ],
       "throughput_ratio": 3.176438415892793
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036460140720009804,
        0.036532205529510975
       ],
       "candidate_seconds": [
        0.011269848793745041,
        0.011676100082695484
       ],
       "throughput_ratio": 3.181055908499159
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03652701899409294,
        0.03644819650799036
       ],
       "candidate_seconds": [
        0.011636599898338318,
        0.011536523699760437
       ],
       "throughput_ratio": 3.149131587425295
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036476679146289825,
        0.036390842869877815
       ],
       "candidate_seconds": [
        0.011609251610934734,
        0.011627888306975365
       ],
       "throughput_ratio": 3.135821459679931
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036481644958257675,
        0.03660536836832762
       ],
       "candidate_seconds": [
        0.011637706309556961,
        0.011551769450306892
       ],
       "throughput_ratio": 3.151732022035775
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03650259133428335,
        0.0363986361771822
       ],
       "candidate_seconds": [
        0.011650540865957737,
        0.01115046814084053
       ],
       "throughput_ratio": 3.1972807646244767
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03646001871675253,
        0.036448219791054726
       ],
       "candidate_seconds": [
        0.011614996939897537,
        0.01153903640806675
       ],
       "throughput_ratio": 3.1488353416497685
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.0364470724016428,
        0.03641996439546347
       ],
       "candidate_seconds": [
        0.01158875785768032,
        0.01159055158495903
       ],
       "throughput_ratio": 3.1436241436539163
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03637722786515951,
        0.03640499897301197
       ],
       "candidate_seconds": [
        0.011497949250042439,
        0.011469625867903233
       ],
       "throughput_ratio": 3.168912106063092
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03636666480451822,
        0.036330304108560085
       ],
       "candidate_seconds": [
        0.011463107541203499,
        0.011582277715206146
       ],
       "throughput_ratio": 3.154513066465617
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03640471212565899,
        0.03633051086217165
       ],
       "candidate_seconds": [
        0.01144254021346569,
        0.011418502777814865
       ],
       "throughput_ratio": 3.181623122600864
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03631902113556862,
        0.036280419677495956
       ],
       "candidate_seconds": [
        0.011421061120927334,
        0.011412646621465683
       ],
       "throughput_ratio": 3.1794854183176127
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036351258866488934,
        0.036364663392305374
       ],
       "candidate_seconds": [
        0.011462284252047539,
        0.011438746005296707
       ],
       "throughput_ratio": 3.175224932750555
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03635442443192005,
        0.03630609530955553
       ],
       "candidate_seconds": [
        0.011298038996756077,
        0.011539578437805176
       ],
       "throughput_ratio": 3.1816155932061005
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03635419812053442,
        0.036262896843254566
       ],
       "candidate_seconds": [
        0.011450709775090218,
        0.011702151969075203
       ],
       "throughput_ratio": 3.136419841581297
      }
     ],
     "native_median_seconds": 0.03640167415142059,
     "candidate_median_seconds": 0.011528197675943375,
     "geomean_paired_ratio": 3.164059193113272,
     "every_ABBA_round_faster": true,
     "native_CV": 0.0021536204226985317,
     "candidate_CV": 0.01074161564612437,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 6480199680
    }
   ]
  },
  {
   "device": 0,
   "seed": 63102,
   "round": "primary",
   "cases": [
    {
     "shape": [
      1,
      14,
      2048,
      2048
     ],
     "dtype": "torch.bfloat16",
     "seed": 63102,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.01158794667571783,
        0.011341853067278862
       ],
       "candidate_seconds": [
        0.002830060198903084,
        0.002626306377351284
       ],
       "throughput_ratio": 4.202393556691221
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011471234261989594,
        0.011496828868985176
       ],
       "candidate_seconds": [
        0.002788242883980274,
        0.002787649631500244
       ],
       "throughput_ratio": 4.119172503273305
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011414526961743832,
        0.011387179605662823
       ],
       "candidate_seconds": [
        0.0027215545997023582,
        0.0026211366057395935
       ],
       "throughput_ratio": 4.267831639639087
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011348892003297806,
        0.011384543962776661
       ],
       "candidate_seconds": [
        0.002785680815577507,
        0.002875356003642082
       ],
       "throughput_ratio": 4.015772497520767
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011137516237795353,
        0.011455533094704151
       ],
       "candidate_seconds": [
        0.002846384420990944,
        0.0026693660765886307
       ],
       "throughput_ratio": 4.096097048337085
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011496554128825665,
        0.011446453630924225
       ],
       "candidate_seconds": [
        0.0028318678960204124,
        0.002830471843481064
       ],
       "throughput_ratio": 4.051859975779171
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011489955708384514,
        0.011568081565201283
       ],
       "candidate_seconds": [
        0.00278683565557003,
        0.002694421447813511
       ],
       "throughput_ratio": 4.206706023578466
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011544900946319103,
        0.011436374858021736
       ],
       "candidate_seconds": [
        0.0028075259178876877,
        0.0027269329875707626
       ],
       "throughput_ratio": 4.152397948365861
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.01140044815838337,
        0.01144169457256794
       ],
       "candidate_seconds": [
        0.002719367854297161,
        0.0025652600452303886
       ],
       "throughput_ratio": 4.32237485121581
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011411950923502445,
        0.01138690672814846
       ],
       "candidate_seconds": [
        0.002754751592874527,
        0.002720431424677372
       ],
       "throughput_ratio": 4.1640357187265105
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.01139330305159092,
        0.011402796022593975
       ],
       "candidate_seconds": [
        0.0026953406631946564,
        0.0026319120079278946
       ],
       "throughput_ratio": 4.279147335690637
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.01138248573988676,
        0.011591455899178982
       ],
       "candidate_seconds": [
        0.0026957672089338303,
        0.00293613038957119
       ],
       "throughput_ratio": 4.079254147867345
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011462729424238205,
        0.01134380791336298
       ],
       "candidate_seconds": [
        0.00273129902780056,
        0.0025304146111011505
       ],
       "throughput_ratio": 4.334431499461389
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011407459154725075,
        0.011602490209043026
       ],
       "candidate_seconds": [
        0.002937711775302887,
        0.002796838991343975
       ],
       "throughput_ratio": 4.012511232370274
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011417693458497524,
        0.011485961265861988
       ],
       "candidate_seconds": [
        0.0027914419770240784,
        0.002698684111237526
       ],
       "throughput_ratio": 4.171790293364962
      }
     ],
     "native_median_seconds": 0.01142703415825963,
     "candidate_median_seconds": 0.0027430253103375435,
     "geomean_paired_ratio": 4.163829860830047,
     "every_ABBA_round_faster": true,
     "native_CV": 0.007983292601020684,
     "candidate_CV": 0.03496552784013344,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 1077936128
    },
    {
     "shape": [
      1,
      14,
      2048,
      2048
     ],
     "dtype": "torch.float32",
     "seed": 63102,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009529937990009785,
        0.009399840608239174
       ],
       "candidate_seconds": [
        0.003281128592789173,
        0.003164786845445633
       ],
       "throughput_ratio": 2.936709111317914
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009305659681558609,
        0.009423886425793171
       ],
       "candidate_seconds": [
        0.0031005218625068665,
        0.0032909251749515533
       ],
       "throughput_ratio": 2.9304077773911503
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.00938809011131525,
        0.009309020824730396
       ],
       "candidate_seconds": [
        0.003097762353718281,
        0.0030449116602540016
       ],
       "throughput_ratio": 3.0438064747562255
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009382294490933418,
        0.009293188340961933
       ],
       "candidate_seconds": [
        0.0033357003703713417,
        0.003299298696219921
       ],
       "throughput_ratio": 2.8146926087647364
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009292690083384514,
        0.009298366494476795
       ],
       "candidate_seconds": [
        0.003164983354508877,
        0.0031049372628331184
       ],
       "throughput_ratio": 2.9651183344236034
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.00938069261610508,
        0.009426341392099857
       ],
       "candidate_seconds": [
        0.003207717090845108,
        0.003240738995373249
       ],
       "throughput_ratio": 2.916517342561941
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009335869923233986,
        0.00930606946349144
       ],
       "candidate_seconds": [
        0.0031887833029031754,
        0.0031172605231404305
       ],
       "throughput_ratio": 2.95620200255115
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009330843575298786,
        0.009387197904288769
       ],
       "candidate_seconds": [
        0.0031771184876561165,
        0.0032725511118769646
       ],
       "throughput_ratio": 2.90217059815632
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009333203546702862,
        0.00929634552448988
       ],
       "candidate_seconds": [
        0.003123607486486435,
        0.003130176104605198
       ],
       "throughput_ratio": 2.978924486247029
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009377089329063892,
        0.009304283186793327
       ],
       "candidate_seconds": [
        0.003233294002711773,
        0.003271978348493576
       ],
       "throughput_ratio": 2.871727962687955
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.00932393129914999,
        0.009336648508906364
       ],
       "candidate_seconds": [
        0.003237612545490265,
        0.0031358664855360985
       ],
       "throughput_ratio": 2.9278483097246997
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009350134991109371,
        0.009409831836819649
       ],
       "candidate_seconds": [
        0.003102431073784828,
        0.003264819271862507
       ],
       "throughput_ratio": 2.9463215374833456
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009350670501589775,
        0.009311663918197155
       ],
       "candidate_seconds": [
        0.0031310785561800003,
        0.0030907420441508293
       ],
       "throughput_ratio": 2.9994973527193323
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009352780878543854,
        0.009409132413566113
       ],
       "candidate_seconds": [
        0.0032012127339839935,
        0.003239039331674576
       ],
       "throughput_ratio": 2.9132265477859685
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009380349889397621,
        0.009277240373194218
       ],
       "candidate_seconds": [
        0.0032142559066414833,
        0.0030641350895166397
       ],
       "throughput_ratio": 2.971715249019821
      }
     ],
     "native_median_seconds": 0.009343391750007868,
     "candidate_median_seconds": 0.003182950895279646,
     "geomean_paired_ratio": 2.9378580303217876,
     "every_ABBA_round_faster": true,
     "native_CV": 0.005775411388927042,
     "candidate_CV": 0.024274545177207705,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 1312817152
    },
    {
     "shape": [
      1,
      14,
      4096,
      4096
     ],
     "dtype": "torch.bfloat16",
     "seed": 63102,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04435337986797094,
        0.04481012932956219
       ],
       "candidate_seconds": [
        0.009537385776638985,
        0.009347368031740189
       ],
       "throughput_ratio": 4.721454677263055
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044184498488903046,
        0.04420837573707104
       ],
       "candidate_seconds": [
        0.009572367183864117,
        0.009505245834589005
       ],
       "throughput_ratio": 4.6333298689136155
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.043760208413004875,
        0.04422919172793627
       ],
       "candidate_seconds": [
        0.009496429935097694,
        0.00927727110683918
       ],
       "throughput_ratio": 4.68684357678806
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044056160375475883,
        0.043907737359404564
       ],
       "candidate_seconds": [
        0.009427757933735847,
        0.009471503086388111
       ],
       "throughput_ratio": 4.654356466171686
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04404884856194258,
        0.04399533849209547
       ],
       "candidate_seconds": [
        0.009208699688315392,
        0.009147711098194122
       ],
       "throughput_ratio": 4.796372672088132
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044206131249666214,
        0.044362060725688934
       ],
       "candidate_seconds": [
        0.009545325301587582,
        0.009670215658843517
       ],
       "throughput_ratio": 4.609195866915012
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04419337399303913,
        0.044391389936208725
       ],
       "candidate_seconds": [
        0.009561048820614815,
        0.009489846415817738
       ],
       "throughput_ratio": 4.649900323835707
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044252476654946804,
        0.044327071867883205
       ],
       "candidate_seconds": [
        0.010265184566378593,
        0.009688322432339191
       ],
       "throughput_ratio": 4.4392972387521725
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04416168574243784,
        0.04422268085181713
       ],
       "candidate_seconds": [
        0.009636623784899712,
        0.00927162915468216
       ],
       "throughput_ratio": 4.674380381766221
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04416144639253616,
        0.04427262023091316
       ],
       "candidate_seconds": [
        0.009606433100998402,
        0.00958506390452385
       ],
       "throughput_ratio": 4.607981680532941
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04429418407380581,
        0.04413488879799843
       ],
       "candidate_seconds": [
        0.009449372999370098,
        0.00937608815729618
       ],
       "throughput_ratio": 4.6973124395675505
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04405444674193859,
        0.04423135705292225
       ],
       "candidate_seconds": [
        0.009377587586641312,
        0.009397190995514393
       ],
       "throughput_ratio": 4.702361916468681
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.044217295944690704,
        0.04420285765081644
       ],
       "candidate_seconds": [
        0.009419317357242107,
        0.00935703981667757
       ],
       "throughput_ratio": 4.709121837452185
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04410083591938019,
        0.044165859930217266
       ],
       "candidate_seconds": [
        0.009434443898499012,
        0.00945306196808815
       ],
       "throughput_ratio": 4.673284893890895
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04410812631249428,
        0.044175163842737675
       ],
       "candidate_seconds": [
        0.009406207129359245,
        0.009233001619577408
       ],
       "throughput_ratio": 4.736429069730142
      }
     ],
     "native_median_seconds": 0.044198115821927786,
     "candidate_median_seconds": 0.009451217483729124,
     "geomean_paired_ratio": 4.665461117577323,
     "every_ABBA_round_faster": true,
     "native_CV": 0.003932313987973891,
     "candidate_CV": 0.020900743165567386,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 5540675584
    },
    {
     "shape": [
      1,
      14,
      4096,
      4096
     ],
     "dtype": "torch.float32",
     "seed": 63102,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03653138969093561,
        0.03636815678328276
       ],
       "candidate_seconds": [
        0.011614258401095867,
        0.011450729332864285
       ],
       "throughput_ratio": 3.1606150115954064
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03639706876128912,
        0.036321213468909264
       ],
       "candidate_seconds": [
        0.011479261331260204,
        0.011471749283373356
       ],
       "throughput_ratio": 3.168413079981464
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.0363030256703496,
        0.03646776732057333
       ],
       "candidate_seconds": [
        0.011542195454239845,
        0.011479022912681103
       ],
       "throughput_ratio": 3.1610313507771095
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03638213872909546,
        0.036484479904174805
       ],
       "candidate_seconds": [
        0.01126779429614544,
        0.011496096849441528
       ],
       "throughput_ratio": 3.200973777604637
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03635679930448532,
        0.0364506971091032
       ],
       "candidate_seconds": [
        0.011463011614978313,
        0.01142841950058937
       ],
       "throughput_ratio": 3.180556778910805
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03642648831009865,
        0.03648054786026478
       ],
       "candidate_seconds": [
        0.011652193032205105,
        0.011645279824733734
       ],
       "throughput_ratio": 3.1293967641065
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03637741133570671,
        0.03641255293041468
       ],
       "candidate_seconds": [
        0.011621243320405483,
        0.011462623253464699
       ],
       "throughput_ratio": 3.1532830097240336
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036348084919154644,
        0.03635517414659262
       ],
       "candidate_seconds": [
        0.011656839400529861,
        0.011657170951366425
       ],
       "throughput_ratio": 3.118436423780425
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03633155673742294,
        0.03644431009888649
       ],
       "candidate_seconds": [
        0.011616833508014679,
        0.011423430405557156
       ],
       "throughput_ratio": 3.158638594996341
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03640755079686642,
        0.036355575546622276
       ],
       "candidate_seconds": [
        0.011559291742742062,
        0.011604229919612408
       ],
       "throughput_ratio": 3.1412808209445924
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03633988928049803,
        0.036539290100336075
       ],
       "candidate_seconds": [
        0.0116629833355546,
        0.011452732607722282
       ],
       "throughput_ratio": 3.1527978436692434
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.0363870607689023,
        0.03642271179705858
       ],
       "candidate_seconds": [
        0.011585790663957596,
        0.011623826809227467
       ],
       "throughput_ratio": 3.1370518126841485
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036269331350922585,
        0.03607284277677536
       ],
       "candidate_seconds": [
        0.011528600007295609,
        0.011359233409166336
       ],
       "throughput_ratio": 3.160726173219447
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036282531917095184,
        0.03632059879601002
       ],
       "candidate_seconds": [
        0.011509379372000694,
        0.011134453117847443
       ],
       "throughput_ratio": 3.206309300585721
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036267587915062904,
        0.03639715537428856
       ],
       "candidate_seconds": [
        0.011490658856928349,
        0.011403106153011322
       ],
       "throughput_ratio": 3.1739970798950274
      }
     ],
     "native_median_seconds": 0.036379775032401085,
     "candidate_median_seconds": 0.011502738110721111,
     "geomean_paired_ratio": 3.1601479672340504,
     "every_ABBA_round_faster": true,
     "native_CV": 0.0024628037351938524,
     "candidate_CV": 0.010452400737104225,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 6480199680
    }
   ]
  },
  {
   "device": 1,
   "seed": 63100,
   "round": "primary",
   "cases": [
    {
     "shape": [
      1,
      14,
      2048,
      2048
     ],
     "dtype": "torch.bfloat16",
     "seed": 63100,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011638084426522255,
        0.01141876820474863
       ],
       "candidate_seconds": [
        0.002825050614774227,
        0.002680663950741291
       ],
       "throughput_ratio": 4.187803845786909
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011493658646941185,
        0.011494837701320648
       ],
       "candidate_seconds": [
        0.002792131155729294,
        0.0027820803225040436
       ],
       "throughput_ratio": 4.124080408149081
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011509707197546959,
        0.011464562267065048
       ],
       "candidate_seconds": [
        0.0027649328112602234,
        0.0027284370735287666
       ],
       "throughput_ratio": 4.182181419865284
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011482865549623966,
        0.011455966159701347
       ],
       "candidate_seconds": [
        0.002760940231382847,
        0.0027556167915463448
       ],
       "throughput_ratio": 4.1581790261537455
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.01149462629109621,
        0.011329462751746178
       ],
       "candidate_seconds": [
        0.002748834900557995,
        0.002678520046174526
       ],
       "throughput_ratio": 4.205379833611837
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011508111841976643,
        0.011471353471279144
       ],
       "candidate_seconds": [
        0.0027514416724443436,
        0.0027517126873135567
       ],
       "throughput_ratio": 4.175689761002219
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011501329950988293,
        0.011355480179190636
       ],
       "candidate_seconds": [
        0.002742251381278038,
        0.0026828087866306305
       ],
       "throughput_ratio": 4.213190162458623
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011311153881251812,
        0.011457175016403198
       ],
       "candidate_seconds": [
        0.0027428073808550835,
        0.002752285450696945
       ],
       "throughput_ratio": 4.1433929499648405
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011443332768976688,
        0.011373158544301987
       ],
       "candidate_seconds": [
        0.0027304599061608315,
        0.0027121854946017265
       ],
       "throughput_ratio": 4.192169364934541
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011455019935965538,
        0.011444365605711937
       ],
       "candidate_seconds": [
        0.002730553038418293,
        0.0027575017884373665
       ],
       "throughput_ratio": 4.172586875338763
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011471339501440525,
        0.011374319903552532
       ],
       "candidate_seconds": [
        0.002760709263384342,
        0.0026785731315612793
       ],
       "throughput_ratio": 4.200123793208838
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011451316066086292,
        0.011447136290371418
       ],
       "candidate_seconds": [
        0.0027445973828434944,
        0.002758610062301159
       ],
       "throughput_ratio": 4.1609284375896936
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011498000472784042,
        0.01137081440538168
       ],
       "candidate_seconds": [
        0.0027312133461236954,
        0.002656945027410984
       ],
       "throughput_ratio": 4.244272957990947
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011482605710625648,
        0.011433265171945095
       ],
       "candidate_seconds": [
        0.002732047811150551,
        0.002741742879152298
       ],
       "throughput_ratio": 4.18647189472692
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011478085070848465,
        0.011424287222325802
       ],
       "candidate_seconds": [
        0.0027633430436253548,
        0.0027055377140641212
       ],
       "throughput_ratio": 4.187762232879656
      }
     ],
     "native_median_seconds": 0.011456570588052273,
     "candidate_median_seconds": 0.002743702381849289,
     "geomean_paired_ratio": 4.182186021408187,
     "every_ABBA_round_faster": true,
     "native_CV": 0.005521696890208335,
     "candidate_CV": 0.013124305096300469,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 1077936128
    },
    {
     "shape": [
      1,
      14,
      2048,
      2048
     ],
     "dtype": "torch.float32",
     "seed": 63100,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009496508166193962,
        0.009363619610667229
       ],
       "candidate_seconds": [
        0.003282615914940834,
        0.0032326076179742813
       ],
       "throughput_ratio": 2.8947783113778414
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009440882131457329,
        0.009456799365580082
       ],
       "candidate_seconds": [
        0.0032928911969065666,
        0.003537488169968128
       ],
       "throughput_ratio": 2.7667103804929982
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009609527885913849,
        0.009417038410902023
       ],
       "candidate_seconds": [
        0.003438052721321583,
        0.0032323114573955536
       ],
       "throughput_ratio": 2.852402925393965
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009501696564257145,
        0.009595829993486404
       ],
       "candidate_seconds": [
        0.003375350497663021,
        0.0034229783341288567
       ],
       "throughput_ratio": 2.8091501647339254
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009532885625958443,
        0.009448880329728127
       ],
       "candidate_seconds": [
        0.0034638606011867523,
        0.0032695112749934196
       ],
       "throughput_ratio": 2.819058015024545
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.00954357162117958,
        0.00947096198797226
       ],
       "candidate_seconds": [
        0.0030585145577788353,
        0.0033531999215483665
       ],
       "throughput_ratio": 2.9655926929464718
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009524268098175526,
        0.009451886638998985
       ],
       "candidate_seconds": [
        0.003367261029779911,
        0.0032027140259742737
       ],
       "throughput_ratio": 2.8883145789959452
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009475268423557281,
        0.009459208697080612
       ],
       "candidate_seconds": [
        0.003386802040040493,
        0.003386947326362133
       ],
       "throughput_ratio": 2.7952727649699765
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.00953004602342844,
        0.00942927598953247
       ],
       "candidate_seconds": [
        0.0033599697053432465,
        0.0032040001824498177
       ],
       "throughput_ratio": 2.888392594277334
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009543824940919876,
        0.009539901278913021
       ],
       "candidate_seconds": [
        0.0034395232796669006,
        0.003439202904701233
       ],
       "throughput_ratio": 2.774311072768176
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009552128612995148,
        0.009449023753404617
       ],
       "candidate_seconds": [
        0.00343517679721117,
        0.0031928177922964096
       ],
       "throughput_ratio": 2.866802636875936
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009498772211372852,
        0.009535664692521095
       ],
       "candidate_seconds": [
        0.0033899089321494102,
        0.003431891091167927
       ],
       "throughput_ratio": 2.790236717410809
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009351453743875027,
        0.009381202980875969
       ],
       "candidate_seconds": [
        0.003659701906144619,
        0.003128817304968834
       ],
       "throughput_ratio": 2.759461399782717
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.00938143115490675,
        0.009427479468286037
       ],
       "candidate_seconds": [
        0.0032146377488970757,
        0.0033468352630734444
       ],
       "throughput_ratio": 2.866568313072152
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009357046335935593,
        0.009409988299012184
       ],
       "candidate_seconds": [
        0.0033260779455304146,
        0.003193122334778309
       ],
       "throughput_ratio": 2.878732640203385
      }
     ],
     "native_median_seconds": 0.009465085342526436,
     "candidate_median_seconds": 0.0033565848134458065,
     "geomean_paired_ratio": 2.840484874014521,
     "every_ABBA_round_faster": true,
     "native_CV": 0.007239114671946349,
     "candidate_CV": 0.03765003497169289,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 1312817152
    },
    {
     "shape": [
      1,
      14,
      4096,
      4096
     ],
     "dtype": "torch.bfloat16",
     "seed": 63100,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04451610613614321,
        0.0442291060462594
       ],
       "candidate_seconds": [
        0.009631671942770481,
        0.009205285459756851
       ],
       "throughput_ratio": 4.711228585700139
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044312103651463985,
        0.044277857057750225
       ],
       "candidate_seconds": [
        0.009602528996765614,
        0.009501234628260136
       ],
       "throughput_ratio": 4.637304064690279
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04413340613245964,
        0.044133152812719345
       ],
       "candidate_seconds": [
        0.009484956972301006,
        0.009387567639350891
       ],
       "throughput_ratio": 4.676987353916773
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04405762068927288,
        0.044220488518476486
       ],
       "candidate_seconds": [
        0.009448728524148464,
        0.009519699960947037
       ],
       "throughput_ratio": 4.653949549753905
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04406768921762705,
        0.04426681995391846
       ],
       "candidate_seconds": [
        0.009468482807278633,
        0.009145401418209076
       ],
       "throughput_ratio": 4.745624723000608
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04414874315261841,
        0.044245156459510326
       ],
       "candidate_seconds": [
        0.009448968805372715,
        0.00942483451217413
       ],
       "throughput_ratio": 4.683417439766872
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04399446491152048,
        0.04421788174659014
       ],
       "candidate_seconds": [
        0.009429307654500008,
        0.009167601354420185
       ],
       "throughput_ratio": 4.743387549823398
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04414429422467947,
        0.04416555352509022
       ],
       "candidate_seconds": [
        0.010081269778311253,
        0.009523958899080753
       ],
       "throughput_ratio": 4.504402840840372
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04400908853858709,
        0.044049857184290886
       ],
       "candidate_seconds": [
        0.009359655901789665,
        0.009438426233828068
       ],
       "throughput_ratio": 4.684464355862558
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04421385005116463,
        0.04402306769043207
       ],
       "candidate_seconds": [
        0.009443383663892746,
        0.009624730795621872
       ],
       "throughput_ratio": 4.627458993333669
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.044041247107088566,
        0.04430607985705137
       ],
       "candidate_seconds": [
        0.009688401594758034,
        0.009361186064779758
       ],
       "throughput_ratio": 4.637755343743936
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04429752565920353,
        0.044163141399621964
       ],
       "candidate_seconds": [
        0.009591535665094852,
        0.009389440529048443
       ],
       "throughput_ratio": 4.660490912270393
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04417866840958595,
        0.044328054413199425
       ],
       "candidate_seconds": [
        0.00961926206946373,
        0.009241674095392227
       ],
       "throughput_ratio": 4.692594368019871
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04420365858823061,
        0.04420234262943268
       ],
       "candidate_seconds": [
        0.009487009607255459,
        0.009590947069227695
       ],
       "throughput_ratio": 4.633934478247286
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.0441937604919076,
        0.04436296224594116
       ],
       "candidate_seconds": [
        0.009645214304327965,
        0.00929704774171114
       ],
       "throughput_ratio": 4.675086983941619
      }
     ],
     "native_median_seconds": 0.04419805156067014,
     "candidate_median_seconds": 0.009458725806325674,
     "geomean_paired_ratio": 4.6642049101766405,
     "every_ABBA_round_faster": true,
     "native_CV": 0.0026157700251229748,
     "candidate_CV": 0.018978094582707942,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 5540675584
    },
    {
     "shape": [
      1,
      14,
      4096,
      4096
     ],
     "dtype": "torch.float32",
     "seed": 63100,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036483561620116234,
        0.03647398855537176
       ],
       "candidate_seconds": [
        0.011694762855768204,
        0.011326143518090248
       ],
       "throughput_ratio": 3.1691866945054534
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036338294856250286,
        0.036394149996340275
       ],
       "candidate_seconds": [
        0.01148987002670765,
        0.011475179344415665
       ],
       "throughput_ratio": 3.167092901792134
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036378901451826096,
        0.03639115206897259
       ],
       "candidate_seconds": [
        0.011452107690274715,
        0.011401615105569363
       ],
       "throughput_ratio": 3.184166280954096
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03640245459973812,
        0.036344983614981174
       ],
       "candidate_seconds": [
        0.011454969644546509,
        0.011468272656202316
       ],
       "throughput_ratio": 3.173523067124884
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.0363379567861557,
        0.03637554403394461
       ],
       "candidate_seconds": [
        0.0114669818431139,
        0.011443422175943851
       ],
       "throughput_ratio": 3.1738201019770074
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03622711915522814,
        0.03649136796593666
       ],
       "candidate_seconds": [
        0.011480221524834633,
        0.011552215553820133
       ],
       "throughput_ratio": 3.1572207002165835
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036417847499251366,
        0.036399747245013714
       ],
       "candidate_seconds": [
        0.011501376517117023,
        0.011437611654400826
       ],
       "throughput_ratio": 3.174403081766218
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03640397358685732,
        0.03636983875185251
       ],
       "candidate_seconds": [
        0.011495145969092846,
        0.01151509489864111
       ],
       "throughput_ratio": 3.1626706020603503
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036223278380930424,
        0.036500100046396255
       ],
       "candidate_seconds": [
        0.011757947504520416,
        0.011426773853600025
       ],
       "throughput_ratio": 3.136694088490968
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036514087580144405,
        0.03650172706693411
       ],
       "candidate_seconds": [
        0.01173957996070385,
        0.011640631593763828
       ],
       "throughput_ratio": 3.1229749344644433
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03651993814855814,
        0.03648482542484999
       ],
       "candidate_seconds": [
        0.011675945483148098,
        0.011205470189452171
       ],
       "throughput_ratio": 3.1905702259860123
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03640442714095116,
        0.03646716009825468
       ],
       "candidate_seconds": [
        0.011687101796269417,
        0.011692793108522892
       ],
       "throughput_ratio": 3.11684836634013
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03639522194862366,
        0.036528744734823704
       ],
       "candidate_seconds": [
        0.011232836171984673,
        0.011380995623767376
       ],
       "throughput_ratio": 3.224750557185357
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03627691511064768,
        0.036329216323792934
       ],
       "candidate_seconds": [
        0.011633090674877167,
        0.011582338251173496
       ],
       "throughput_ratio": 3.1274947219677385
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036288813687860966,
        0.036479150876402855
       ],
       "candidate_seconds": [
        0.011595522984862328,
        0.011372744105756283
       ],
       "throughput_ratio": 3.1681956795942128
      }
     ],
     "native_median_seconds": 0.03640110092237592,
     "candidate_median_seconds": 0.011485045775771141,
     "geomean_paired_ratio": 3.1631897464729732,
     "every_ABBA_round_faster": true,
     "native_CV": 0.0022938166148952495,
     "candidate_CV": 0.011946524505826752,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 6480199680
    }
   ]
  },
  {
   "device": 1,
   "seed": 63101,
   "round": "primary",
   "cases": [
    {
     "shape": [
      1,
      14,
      2048,
      2048
     ],
     "dtype": "torch.bfloat16",
     "seed": 63101,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011733323335647583,
        0.011435072869062424
       ],
       "candidate_seconds": [
        0.0029217274859547615,
        0.00269866269081831
       ],
       "throughput_ratio": 4.122204237786933
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.0116030378267169,
        0.01153254322707653
       ],
       "candidate_seconds": [
        0.0028737597167491913,
        0.002805315889418125
       ],
       "throughput_ratio": 4.073828675333859
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011544930748641491,
        0.011452962644398212
       ],
       "candidate_seconds": [
        0.002760147675871849,
        0.0026807887479662895
       ],
       "throughput_ratio": 4.226826340458608
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011456616222858429,
        0.011464645154774189
       ],
       "candidate_seconds": [
        0.002770727500319481,
        0.002779528498649597
       ],
       "throughput_ratio": 4.129766515614791
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011495898477733135,
        0.011415460146963596
       ],
       "candidate_seconds": [
        0.0028936034068465233,
        0.002680327743291855
       ],
       "throughput_ratio": 4.110448803108724
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011477802880108356,
        0.011529087089002132
       ],
       "candidate_seconds": [
        0.0027694208547472954,
        0.002847210504114628
       ],
       "throughput_ratio": 4.096207940157975
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011553245596587658,
        0.01136928889900446
       ],
       "candidate_seconds": [
        0.002818141132593155,
        0.0026675965636968613
       ],
       "throughput_ratio": 4.178569185160738
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011524470522999763,
        0.011440328322350979
       ],
       "candidate_seconds": [
        0.002795075997710228,
        0.0027512991800904274
       ],
       "throughput_ratio": 4.140505845559684
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011467400006949902,
        0.011385315097868443
       ],
       "candidate_seconds": [
        0.0027364343404769897,
        0.0026537012308835983
       ],
       "throughput_ratio": 4.23972918719182
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011442339979112148,
        0.01144979428499937
       ],
       "candidate_seconds": [
        0.0027845995500683784,
        0.0027553504332900047
       ],
       "throughput_ratio": 4.132191505857971
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011040928773581982,
        0.011373423971235752
       ],
       "candidate_seconds": [
        0.002726549282670021,
        0.002676473930478096
       ],
       "throughput_ratio": 4.148483517574566
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.01145316194742918,
        0.011437688022851944
       ],
       "candidate_seconds": [
        0.0027414169162511826,
        0.002814173698425293
       ],
       "throughput_ratio": 4.120326992742994
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.01144885178655386,
        0.011369222775101662
       ],
       "candidate_seconds": [
        0.0027568768709897995,
        0.0026466865092515945
       ],
       "throughput_ratio": 4.2227828112633645
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011523459106683731,
        0.011443144641816616
       ],
       "candidate_seconds": [
        0.002739422023296356,
        0.0027434220537543297
       ],
       "throughput_ratio": 4.18881212482965
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011505752801895142,
        0.011328699067234993
       ],
       "candidate_seconds": [
        0.002760055474936962,
        0.002713494934141636
       ],
       "throughput_ratio": 4.171780683933451
      }
     ],
     "native_median_seconds": 0.011453062295913696,
     "candidate_median_seconds": 0.002756113652139902,
     "geomean_paired_ratio": 4.153220677719906,
     "every_ABBA_round_faster": true,
     "native_CV": 0.009555656883128259,
     "candidate_CV": 0.02443486296203206,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 1077936128
    },
    {
     "shape": [
      1,
      14,
      2048,
      2048
     ],
     "dtype": "torch.float32",
     "seed": 63101,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.00947837345302105,
        0.00935437623411417
       ],
       "candidate_seconds": [
        0.0032972292974591255,
        0.00325075164437294
       ],
       "throughput_ratio": 2.876115531555898
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009444985538721085,
        0.009407599456608295
       ],
       "candidate_seconds": [
        0.0032540401443839073,
        0.0032389331609010696
       ],
       "throughput_ratio": 2.9035365015260814
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009442283771932125,
        0.009355229325592518
       ],
       "candidate_seconds": [
        0.0032487669959664345,
        0.0031593330204486847
       ],
       "throughput_ratio": 2.9333988310688897
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009441378526389599,
        0.009385406039655209
       ],
       "candidate_seconds": [
        0.0032411301508545876,
        0.003230980597436428
       ],
       "throughput_ratio": 2.908909519358285
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009444521740078926,
        0.00936852302402258
       ],
       "candidate_seconds": [
        0.003232046961784363,
        0.00318110641092062
       ],
       "throughput_ratio": 2.933509253680364
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009405962191522121,
        0.009394210763275623
       ],
       "candidate_seconds": [
        0.003232507035136223,
        0.0032367128878831863
       ],
       "throughput_ratio": 2.9060958165761432
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009415857493877411,
        0.009346752427518368
       ],
       "candidate_seconds": [
        0.0032372726127505302,
        0.0031811408698558807
       ],
       "throughput_ratio": 2.9232473059333963
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009419139474630356,
        0.009088678285479546
       ],
       "candidate_seconds": [
        0.003229408524930477,
        0.00323420949280262
       ],
       "throughput_ratio": 2.863383589397338
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009437471628189087,
        0.009386660531163216
       ],
       "candidate_seconds": [
        0.0032395804300904274,
        0.003228912129998207
       ],
       "throughput_ratio": 2.9101265842832422
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009412448853254318,
        0.009417898952960968
       ],
       "candidate_seconds": [
        0.0032418053597211838,
        0.0031357361003756523
       ],
       "throughput_ratio": 2.9526029621341525
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009431947953999043,
        0.009362488053739071
       ],
       "candidate_seconds": [
        0.00324929878115654,
        0.003222612664103508
       ],
       "throughput_ratio": 2.9040007989452543
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009393473155796528,
        0.009400809183716774
       ],
       "candidate_seconds": [
        0.0032349843531847,
        0.0032587889581918716
       ],
       "throughput_ratio": 2.8942005577230763
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009427308104932308,
        0.009357993490993977
       ],
       "candidate_seconds": [
        0.003240811638534069,
        0.003220559097826481
       ],
       "throughput_ratio": 2.90732452329571
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009456971660256386,
        0.009411321021616459
       ],
       "candidate_seconds": [
        0.0032417401671409607,
        0.0032654674723744392
       ],
       "throughput_ratio": 2.8995989873281482
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009423445910215378,
        0.009321467950940132
       ],
       "candidate_seconds": [
        0.003246883861720562,
        0.003178585320711136
       ],
       "throughput_ratio": 2.917283287640259
      }
     ],
     "native_median_seconds": 0.009409460239112377,
     "candidate_median_seconds": 0.0032369927503168583,
     "geomean_paired_ratio": 2.9088097537984052,
     "every_ABBA_round_faster": true,
     "native_CV": 0.0071580315158684595,
     "candidate_CV": 0.009938568456170398,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 1312817152
    },
    {
     "shape": [
      1,
      14,
      4096,
      4096
     ],
     "dtype": "torch.bfloat16",
     "seed": 63101,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.044428613036870956,
        0.04432860668748617
       ],
       "candidate_seconds": [
        0.009616478346288204,
        0.009513463824987411
       ],
       "throughput_ratio": 4.6397014130878915
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04430302418768406,
        0.04437092412263155
       ],
       "candidate_seconds": [
        0.00953933596611023,
        0.009611918590962887
       ],
       "throughput_ratio": 4.630190050790471
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04435786884278059,
        0.04431185219436884
       ],
       "candidate_seconds": [
        0.009616483002901077,
        0.009453528560698032
       ],
       "throughput_ratio": 4.649694141056655
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04428361542522907,
        0.04427463002502918
       ],
       "candidate_seconds": [
        0.009543453343212605,
        0.009505607187747955
       ],
       "throughput_ratio": 4.648956063020744
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04407814797013998,
        0.044352754950523376
       ],
       "candidate_seconds": [
        0.009578092023730278,
        0.009432878345251083
       ],
       "throughput_ratio": 4.651572287175241
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044254787266254425,
        0.04418078996241093
       ],
       "candidate_seconds": [
        0.009478381834924221,
        0.009497469291090965
       ],
       "throughput_ratio": 4.660427437029345
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.044253995642066,
        0.04420715104788542
       ],
       "candidate_seconds": [
        0.009425798431038857,
        0.009317158721387386
       ],
       "throughput_ratio": 4.719700630511246
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04423338919878006,
        0.04451998136937618
       ],
       "candidate_seconds": [
        0.009349441155791283,
        0.009781861677765846
       ],
       "throughput_ratio": 4.639170230083808
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04418884124606848,
        0.044191704131662846
       ],
       "candidate_seconds": [
        0.009411587379872799,
        0.009355240501463413
       ],
       "throughput_ratio": 4.7094024593056885
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044189262203872204,
        0.044177920557558537
       ],
       "candidate_seconds": [
        0.009393313899636269,
        0.009463838301599026
       ],
       "throughput_ratio": 4.686136157698403
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.044194805435836315,
        0.04421553295105696
       ],
       "candidate_seconds": [
        0.009382345713675022,
        0.009363389573991299
       ],
       "throughput_ratio": 4.7162907738840465
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04422399401664734,
        0.044107080437242985
       ],
       "candidate_seconds": [
        0.009469105862081051,
        0.00938415713608265
       ],
       "throughput_ratio": 4.685187623091755
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04419241938740015,
        0.04424182511866093
       ],
       "candidate_seconds": [
        0.00940560270100832,
        0.009314707480370998
       ],
       "throughput_ratio": 4.7239732487993
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044165441766381264,
        0.04417870659381151
       ],
       "candidate_seconds": [
        0.009361632168293,
        0.00938743632286787
       ],
       "throughput_ratio": 4.711922003050022
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04423814918845892,
        0.04425332788378
       ],
       "candidate_seconds": [
        0.00945745687931776,
        0.00936682615429163
       ],
       "throughput_ratio": 4.700921512614521
      }
     ],
     "native_median_seconds": 0.04423576919361949,
     "candidate_median_seconds": 0.009443203452974558,
     "geomean_paired_ratio": 4.678102325656373,
     "every_ABBA_round_faster": true,
     "native_CV": 0.002062008831071187,
     "candidate_CV": 0.011100505603617509,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 5540675584
    },
    {
     "shape": [
      1,
      14,
      4096,
      4096
     ],
     "dtype": "torch.float32",
     "seed": 63101,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03660461958497763,
        0.03658487927168608
       ],
       "candidate_seconds": [
        0.011690789833664894,
        0.011580617167055607
       ],
       "throughput_ratio": 3.145039698476233
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036509910598397255,
        0.03639056906104088
       ],
       "candidate_seconds": [
        0.011640144512057304,
        0.01156664453446865
       ],
       "throughput_ratio": 3.1413427990095553
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03645410854369402,
        0.036406743340194225
       ],
       "candidate_seconds": [
        0.011592295952141285,
        0.011511893011629581
       ],
       "throughput_ratio": 3.1535775611141177
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03642372041940689,
        0.0363386906683445
       ],
       "candidate_seconds": [
        0.011579473502933979,
        0.011466743424534798
       ],
       "throughput_ratio": 3.157238835195806
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03641010262072086,
        0.036362833343446255
       ],
       "candidate_seconds": [
        0.011426056735217571,
        0.01141440775245428
       ],
       "throughput_ratio": 3.1861408074011077
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03634379059076309,
        0.03627115394920111
       ],
       "candidate_seconds": [
        0.011455575935542583,
        0.011448836885392666
       ],
       "throughput_ratio": 3.1703473521745202
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03635282255709171,
        0.03637282643467188
       ],
       "candidate_seconds": [
        0.011415311135351658,
        0.011280996724963188
       ],
       "throughput_ratio": 3.204294259637115
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03639872092753649,
        0.03633053880184889
       ],
       "candidate_seconds": [
        0.011492489837110043,
        0.011450165882706642
       ],
       "throughput_ratio": 3.170045378249981
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.0364071624353528,
        0.03634138684719801
       ],
       "candidate_seconds": [
        0.01147401425987482,
        0.011421465314924717
       ],
       "throughput_ratio": 3.177419762922252
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03634896222501993,
        0.03631965070962906
       ],
       "candidate_seconds": [
        0.01142456941306591,
        0.011459884233772755
       ],
       "throughput_ratio": 3.1754576297121986
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03635098226368427,
        0.03635554388165474
       ],
       "candidate_seconds": [
        0.011438491754233837,
        0.011384820565581322
       ],
       "throughput_ratio": 3.185625518615689
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036340441554784775,
        0.036314718425273895
       ],
       "candidate_seconds": [
        0.011430035345256329,
        0.011518423445522785
       ],
       "throughput_ratio": 3.166014791775565
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03638027887791395,
        0.03642837330698967
       ],
       "candidate_seconds": [
        0.011474658735096455,
        0.011408762075006962
       ],
       "throughput_ratio": 3.181720634738202
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03645670227706432,
        0.036591763608157635
       ],
       "candidate_seconds": [
        0.011365477927029133,
        0.011554315686225891
       ],
       "throughput_ratio": 3.187134540468829
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03645692765712738,
        0.036418505012989044
       ],
       "candidate_seconds": [
        0.011577601544559002,
        0.011425938457250595
       ],
       "throughput_ratio": 3.168009474384534
      }
     ],
     "native_median_seconds": 0.036385423969477415,
     "candidate_median_seconds": 0.011457730084657669,
     "geomean_paired_ratio": 3.171251161007136,
     "every_ABBA_round_faster": true,
     "native_CV": 0.0022255853590324244,
     "candidate_CV": 0.007508737013139382,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 6480199680
    }
   ]
  },
  {
   "device": 1,
   "seed": 63102,
   "round": "primary",
   "cases": [
    {
     "shape": [
      1,
      14,
      2048,
      2048
     ],
     "dtype": "torch.bfloat16",
     "seed": 63102,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.01176492776721716,
        0.011412730440497398
       ],
       "candidate_seconds": [
        0.002842946909368038,
        0.002660459838807583
       ],
       "throughput_ratio": 4.21151102731012
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011427675373852253,
        0.011487525887787342
       ],
       "candidate_seconds": [
        0.002800160087645054,
        0.002786729484796524
       ],
       "throughput_ratio": 4.101602683302224
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011425713077187538,
        0.011433183215558529
       ],
       "candidate_seconds": [
        0.0028647873550653458,
        0.0026084883138537407
       ],
       "throughput_ratio": 4.176456234893145
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.01147119514644146,
        0.011433805339038372
       ],
       "candidate_seconds": [
        0.0029614418745040894,
        0.0029154596850275993
       ],
       "throughput_ratio": 3.897461996505699
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011460385285317898,
        0.011381606571376324
       ],
       "candidate_seconds": [
        0.002757517620921135,
        0.0026192497462034225
       ],
       "throughput_ratio": 4.248276017362807
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011362667195498943,
        0.01151133794337511
       ],
       "candidate_seconds": [
        0.0027198083698749542,
        0.0028751231729984283
       ],
       "throughput_ratio": 4.088344059903667
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.01140343863517046,
        0.011355916038155556
       ],
       "candidate_seconds": [
        0.0027867164462804794,
        0.002553093247115612
       ],
       "throughput_ratio": 4.262203333102529
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011460217647254467,
        0.011385577730834484
       ],
       "candidate_seconds": [
        0.0027274461463093758,
        0.0026604468002915382
       ],
       "throughput_ratio": 4.2402095966850615
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011351967230439186,
        0.01143431756645441
       ],
       "candidate_seconds": [
        0.0026289401575922966,
        0.00260845385491848
       ],
       "throughput_ratio": 4.350691344295096
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011413638480007648,
        0.011329900473356247
       ],
       "candidate_seconds": [
        0.0029023373499512672,
        0.002646340988576412
       ],
       "throughput_ratio": 4.0989110497615915
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011337356641888618,
        0.011335893534123898
       ],
       "candidate_seconds": [
        0.0029815640300512314,
        0.0025613969191908836
       ],
       "throughput_ratio": 4.090458219647499
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.01136722695082426,
        0.011445708572864532
       ],
       "candidate_seconds": [
        0.0026911934837698936,
        0.002896299585700035
       ],
       "throughput_ratio": 4.082857059517211
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.01140686683356762,
        0.011308813467621803
       ],
       "candidate_seconds": [
        0.002713443711400032,
        0.002569626085460186
       ],
       "throughput_ratio": 4.299712321553954
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.011413437314331532,
        0.011364535428583622
       ],
       "candidate_seconds": [
        0.002764785662293434,
        0.0027738064527511597
       ],
       "throughput_ratio": 4.112592563197219
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.011368395760655403,
        0.011307278648018837
       ],
       "candidate_seconds": [
        0.002634047530591488,
        0.002614564262330532
       ],
       "throughput_ratio": 4.320318458159425
      }
     ],
     "native_median_seconds": 0.011409798637032509,
     "candidate_median_seconds": 0.002723627258092165,
     "geomean_paired_ratio": 4.170483914436361,
     "every_ABBA_round_faster": true,
     "native_CV": 0.00729773097723098,
     "candidate_CV": 0.04487126631058786,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 1077936128
    },
    {
     "shape": [
      1,
      14,
      2048,
      2048
     ],
     "dtype": "torch.float32",
     "seed": 63102,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009489952586591244,
        0.00928663183003664
       ],
       "candidate_seconds": [
        0.003312908113002777,
        0.003029205836355686
       ],
       "throughput_ratio": 2.960619214122955
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009317311458289623,
        0.009326906874775887
       ],
       "candidate_seconds": [
        0.003168513998389244,
        0.003301512449979782
       ],
       "throughput_ratio": 2.881629384647318
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009312471374869347,
        0.009309562854468822
       ],
       "candidate_seconds": [
        0.0030931681394577026,
        0.003115549683570862
       ],
       "throughput_ratio": 2.9993365393846303
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009379911236464977,
        0.00928785465657711
       ],
       "candidate_seconds": [
        0.0032752519473433495,
        0.0032789697870612144
       ],
       "throughput_ratio": 2.848204813555642
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009289941750466824,
        0.009318351745605469
       ],
       "candidate_seconds": [
        0.003126535564661026,
        0.003184206783771515
       ],
       "throughput_ratio": 2.9486695017257696
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009340591728687286,
        0.009443311020731926
       ],
       "candidate_seconds": [
        0.003138878382742405,
        0.0032408060505986214
       ],
       "throughput_ratio": 2.944331016006402
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009317465126514435,
        0.009343956597149372
       ],
       "candidate_seconds": [
        0.003112160600721836,
        0.003130926750600338
       ],
       "throughput_ratio": 2.9891335285756737
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009372354485094547,
        0.009404503740370274
       ],
       "candidate_seconds": [
        0.0031085284426808357,
        0.003317875787615776
       ],
       "throughput_ratio": 2.921829619267223
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009335017763078213,
        0.009292354807257652
       ],
       "candidate_seconds": [
        0.0032268445938825607,
        0.0030904021114110947
       ],
       "throughput_ratio": 2.9486536523461573
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009355970658361912,
        0.009413531050086021
       ],
       "candidate_seconds": [
        0.003139953128993511,
        0.0032702572643756866
       ],
       "throughput_ratio": 2.928063286013723
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009296104311943054,
        0.009288009256124496
       ],
       "candidate_seconds": [
        0.0031493864953517914,
        0.0030986983329057693
       ],
       "throughput_ratio": 2.974369599468164
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009338516741991043,
        0.009400648064911366
       ],
       "candidate_seconds": [
        0.003162117674946785,
        0.0032646693289279938
       ],
       "throughput_ratio": 2.915790549088427
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009285666048526764,
        0.009356236085295677
       ],
       "candidate_seconds": [
        0.003097918815910816,
        0.003151443786919117
       ],
       "throughput_ratio": 2.9830085592058184
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.009272748604416847,
        0.00930580124258995
       ],
       "candidate_seconds": [
        0.003268337808549404,
        0.0032635200768709183
       ],
       "throughput_ratio": 2.8442979276196048
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.009300393052399158,
        0.009292196482419968
       ],
       "candidate_seconds": [
        0.0031665852293372154,
        0.0031275032088160515
       ],
       "throughput_ratio": 2.9539765317111324
      }
     ],
     "native_median_seconds": 0.009317908436059952,
     "candidate_median_seconds": 0.003156780730932951,
     "geomean_paired_ratio": 2.9357662636467206,
     "every_ABBA_round_faster": true,
     "native_CV": 0.005527822651287754,
     "candidate_CV": 0.02472616203042904,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 1312817152
    },
    {
     "shape": [
      1,
      14,
      4096,
      4096
     ],
     "dtype": "torch.bfloat16",
     "seed": 63102,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04444677475839853,
        0.044267102144658566
       ],
       "candidate_seconds": [
        0.009389708749949932,
        0.009223290719091892
       ],
       "throughput_ratio": 4.766232172875251
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04425491392612457,
        0.04428216256201267
       ],
       "candidate_seconds": [
        0.009663291275501251,
        0.0095845777541399
       ],
       "throughput_ratio": 4.599837849675346
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04416187107563019,
        0.04431809484958649
       ],
       "candidate_seconds": [
        0.009673808701336384,
        0.009347493760287762
       ],
       "throughput_ratio": 4.651624992753612
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04419485479593277,
        0.04426714405417442
       ],
       "candidate_seconds": [
        0.009603804908692837,
        0.009574959985911846
       ],
       "throughput_ratio": 4.612497172588683
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.0441593574360013,
        0.044141442514956
       ],
       "candidate_seconds": [
        0.009561906568706036,
        0.00938897579908371
       ],
       "throughput_ratio": 4.659455862648357
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044141860678792,
        0.044319380074739456
       ],
       "candidate_seconds": [
        0.0095652611926198,
        0.009593061171472073
       ],
       "throughput_ratio": 4.617379281566579
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04409747011959553,
        0.04434350039809942
       ],
       "candidate_seconds": [
        0.009555844590067863,
        0.009251861833035946
       ],
       "throughput_ratio": 4.702379361316066
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044261026196181774,
        0.044830894097685814
       ],
       "candidate_seconds": [
        0.009493901394307613,
        0.01002087164670229
       ],
       "throughput_ratio": 4.565357747519928
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04448969569057226,
        0.04432318266481161
       ],
       "candidate_seconds": [
        0.009692497551441193,
        0.00947008840739727
       ],
       "throughput_ratio": 4.634702150646857
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04430158901959658,
        0.04414749052375555
       ],
       "candidate_seconds": [
        0.0096496706828475,
        0.009517516940832138
       ],
       "throughput_ratio": 4.61460915810517
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04426724091172218,
        0.044239701703190804
       ],
       "candidate_seconds": [
        0.009547253139317036,
        0.009425511583685875
       ],
       "throughput_ratio": 4.6649470389313175
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.04417342972010374,
        0.0441231420263648
       ],
       "candidate_seconds": [
        0.00936606153845787,
        0.00949665904045105
       ],
       "throughput_ratio": 4.681009368563519
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.04423831216990948,
        0.04420510679483414
       ],
       "candidate_seconds": [
        0.009389877319335938,
        0.009346342645585537
       ],
       "throughput_ratio": 4.72045157082539
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.044219400733709335,
        0.04412646126002073
       ],
       "candidate_seconds": [
        0.00941415410488844,
        0.00947173684835434
       ],
       "throughput_ratio": 4.677876315840993
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.044278730638325214,
        0.04419516958296299
       ],
       "candidate_seconds": [
        0.009423104114830494,
        0.009299874305725098
       ],
       "throughput_ratio": 4.725418052298476
      }
     ],
     "native_median_seconds": 0.04424730781465769,
     "candidate_median_seconds": 0.009495280217379332,
     "geomean_paired_ratio": 4.659290799509308,
     "every_ABBA_round_faster": true,
     "native_CV": 0.0031389687475462187,
     "candidate_CV": 0.016462491143198737,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 5540675584
    },
    {
     "shape": [
      1,
      14,
      4096,
      4096
     ],
     "dtype": "torch.float32",
     "seed": 63102,
     "rounds": [
      {
       "round": 0,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03670516423881054,
        0.03659041132777929
       ],
       "candidate_seconds": [
        0.011789878830313683,
        0.011438858695328236
       ],
       "throughput_ratio": 3.1553835194736584
      },
      {
       "round": 1,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036360123194754124,
        0.036378142423927784
       ],
       "candidate_seconds": [
        0.011676104739308357,
        0.011852910742163658
       ],
       "throughput_ratio": 3.0914283547460726
      },
      {
       "round": 2,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03636227361857891,
        0.036551604978740215
       ],
       "candidate_seconds": [
        0.011813419871032238,
        0.011313142254948616
       ],
       "throughput_ratio": 3.1528196106331854
      },
      {
       "round": 3,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03627246990799904,
        0.03635714668780565
       ],
       "candidate_seconds": [
        0.011778121814131737,
        0.0113916527479887
       ],
       "throughput_ratio": 3.134670835966814
      },
      {
       "round": 4,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03629778232425451,
        0.03639273438602686
       ],
       "candidate_seconds": [
        0.011622677557170391,
        0.011278911493718624
       ],
       "throughput_ratio": 3.1740381223659937
      },
      {
       "round": 5,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03630373813211918,
        0.03633146081119776
       ],
       "candidate_seconds": [
        0.011486100032925606,
        0.011487607844173908
       ],
       "throughput_ratio": 3.1616663418847
      },
      {
       "round": 6,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036345682106912136,
        0.03644814062863588
       ],
       "candidate_seconds": [
        0.011499082669615746,
        0.011372676119208336
       ],
       "throughput_ratio": 3.1826945801438566
      },
      {
       "round": 7,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.036344158463180065,
        0.03635198995471001
       ],
       "candidate_seconds": [
        0.01145009696483612,
        0.011526772752404213
       ],
       "throughput_ratio": 3.1638839107549823
      },
      {
       "round": 8,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03636342193931341,
        0.036366877146065235
       ],
       "candidate_seconds": [
        0.011483601294457912,
        0.01140157412737608
       ],
       "throughput_ratio": 3.1780529423422754
      },
      {
       "round": 9,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03638284374028444,
        0.03637261223047972
       ],
       "candidate_seconds": [
        0.01147901825606823,
        0.011239889077842236
       ],
       "throughput_ratio": 3.202418800404571
      },
      {
       "round": 10,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036345974542200565,
        0.03643221128731966
       ],
       "candidate_seconds": [
        0.01162608340382576,
        0.011349347420036793
       ],
       "throughput_ratio": 3.1676527150878035
      },
      {
       "round": 11,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03637011907994747,
        0.03643016144633293
       ],
       "candidate_seconds": [
        0.011467250995337963,
        0.011365490965545177
       ],
       "throughput_ratio": 3.188416032161237
      },
      {
       "round": 12,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.03626581281423569,
        0.03638094570487738
       ],
       "candidate_seconds": [
        0.011455223895609379,
        0.011470967903733253
       ],
       "throughput_ratio": 3.1687233167610542
      },
      {
       "round": 13,
       "order": [
        "candidate",
        "native",
        "native",
        "candidate"
       ],
       "native_seconds": [
        0.03655021172016859,
        0.03652821760624647
       ],
       "candidate_seconds": [
        0.011540870182216167,
        0.011722041293978691
       ],
       "throughput_ratio": 3.1414137220612672
      },
      {
       "round": 14,
       "order": [
        "native",
        "candidate",
        "candidate",
        "native"
       ],
       "native_seconds": [
        0.036359439603984356,
        0.03645259700715542
       ],
       "candidate_seconds": [
        0.0116996169090271,
        0.01145151350647211
       ],
       "throughput_ratio": 3.1450747891944664
      }
     ],
     "native_median_seconds": 0.03636849811300635,
     "candidate_median_seconds": 0.011481309775263071,
     "geomean_paired_ratio": 3.160452163283666,
     "every_ABBA_round_faster": true,
     "native_CV": 0.0026547526538284495,
     "candidate_CV": 0.013925209954592293,
     "timing_scope": "dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls",
     "max_GPU_reserved_bytes": 6480199680
    }
   ]
  }
 ],
 "trace_key_evidence_SHA256": "3b997e47e1cd59fb7841b8583e6db65d7869ec6437084ffc6489486e3ee25bda",
 "trace_observer_source_SHA256": {
  "corex_process_guard_data.py": "a8f86db9e75eb353121c346011428400c4d4d1208e35a5d6c12ce48009dfd058",
  "full_reference.py": "7043fab97c80736e68c4dd343956383964bdb4fef3a053b4bf4901ada8c3873e",
  "fused_scale_mask_softmax.py": "093e3a90fc652b7652938ced212107ada5861bceaccbe8d8bdcc008b29e6cd58",
  "fusion_worker.py": "b9f4c61f06a3a76566eae5adbf686f2f55f8b5b856e69e6a94502695aa7050f1",
  "supervise_fusion.py": "fd7e73f68eed857c34b28a98facee2503fef4abba64e6e6e41939fadcb439498"
 },
 "uncompleted": [
  "Real attention integration including dropout/recomputation",
  "New full-model three-arm numerical/SFT and native-resume gates",
  "Full-model latency and effective-token/answer-target throughput"
 ]
}
```

</details>


<a id="appendix-attention-integration"></a>

## 真实attention接入：精度边界与保真候选

| 实现/阶段 | 检查对象 | 实际结果 |
|---|---|---|
| 原全融合直接接入 | MCore原生QK→FP32 softmax→BF16→PV及反向 | 未过原容差；输出最大差0.001953125，Q/K/V梯度最大差至0.00390625 |
| 原生softmax保真边界融合 | 两卡×层1/14/28×层缩放开关×重计算开关，共24组 | 输出与Q/K/V共96项全元素比较及SHA均相同，最大差0 |
| 共同Node0状态布局 | 两rank小型CPU AdamW各3步、每步4项状态 | 与原生逐值相同；moment页位于Node0，训练亲和性恢复；不是完整容量证明 |
| 本轮完整7B更新、原生恢复与端到端 | 完整权重与实际SFT、三臂及原生保存恢复 | 尚未开始，等待新布局容量证明与内存准入确认 |

真实MCore调用已在QK `baddbmm`中应用缩放，并将softmax概率转回BF16；适配器只使用原实例的附加层系数，不重复乘`1/√128`，保留输出dtype、dropout/RNG及重计算边界。

在固定首例中，原全融合与原生QK分数逐值相同；FP32概率最大差约1.19e-7，经BF16转换后有528/58,720,256个概率值不同，后续attention输出和Q/K/V梯度超原门槛。差异发生位置有完整元素计数与SHA，不据此指认某条硬件指令。

因此当前候选保留已安装原生softmax规约算法，以两个Triton kernel融合前后的cast/scale/mask及相应反向。它保留了真实attention组件的逐值结果；其性能尚未测定，不能继承前一节独立FP32输出子链的4.16–4.66×加速。

本阶段是实际MCore模块、真实TP2分组与7B形状的合成QKV检查，没有加载完整7B权重或执行CPU offload更新。`atol=3e-4, rtol=1e-3`保持不变。完整验收仍要求三臂各3更新、每rank3585字段、完整SFT语义、原生iteration2保存和真实step3恢复；这些通过后才进行端到端配对计时。

仅绑CPU不足以保证内存页归属：小型CPU检查捕获了分配器复用其他节点页面的问题，随后使用已有源码绑定的两处moment零值工厂实现定点分配。完整容量仍须单独验证，整模型原80GiB Node1启动要求当前保留。

[MCore适配器](experiments/jiuge_training/attention_integration_20261028_v2/attention_fusion_adapter.py) · [边界融合实现](experiments/jiuge_training/attention_integration_20261028_v2/fused_scale_mask.py) · [真实attention检查入口](experiments/jiuge_training/attention_integration_20261028_v2/actual_attention_probe.py)

[独立JSON文件](docs/jiuge7b/evidence/attention-integration.json)

<details>
<summary>展开完整复核数据</summary>

```json
{
"schema": "real-MCore-attention-component-independent-review-v1",
"date": "2026-10-28",
"component_accepted": true,
"full_model_accepted": false,
"native_resume_accepted": false,
"formal_e2e_performance_accepted": false,
"scope": "actual MCore DotProductAttention on TP2 with syntheticQKV at real7B shape [2048,1,14,128]; no full7B weights or CPUoptimizer updates",
"candidate": "native-softmax with fused cast/scale/mask forward/backward boundaries",
"full_fusion_initial_candidate_accepted": false,
"full_fusion_failure_preserved": true,
"atol": 0.0003,
"rtol": 0.001,
"numerical_tolerances_not_relaxed": true,
"closed_cases": 24,
"checked_output_QKV_fields": 96,
"checked_elements": 352321536,
"max_abs": 0.0,
"all_output_QKV_SHA_equal": true,
"details": [
{
"rank": 0,
"cases": 12,
"output_and_QKV_exact": true
},
{
"rank": 1,
"cases": 12,
"output_and_QKV_exact": true
}
],
"boundary_diagnostic": [
{
"rank": 0,
"full_fusion": {
"boundaries": {
"QK_scores": {
"elements": 58720256,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "d684621d85bc202fe9ebf7501d49f78a67ae695240aed49ec4d0bf7c10f5c39e",
"fused_SHA256": "d684621d85bc202fe9ebf7501d49f78a67ae695240aed49ec4d0bf7c10f5c39e",
"passed": true
},
"FP32_probabilities": {
"elements": 58720256,
"exact_different_elements": 20744589,
"max_abs": 1.1920928955078125e-07,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4a790cb6fea166e9ab5b3d7d55e1321cca6e58fbcb9ce956999aba2e39848dbc",
"fused_SHA256": "631716ba35b3d66eecdd05ad643c54e30550cc7ed429c3b975163d205f2df96e",
"passed": true
},
"BF16_probabilities": {
"elements": 58720256,
"exact_different_elements": 528,
"max_abs": 6.103515625e-05,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "cb7847ae94fd41c3728b598a9401c96c73e180d79a8f3f589f3e7bc63fb04f1f",
"fused_SHA256": "6d3edb15abde560ad69b69f30b2a1f48ef98d982be96054f6d8d22c9aa3a98b2",
"passed": true
}
},
"attention_output": {
"elements": 3670016,
"exact_different_elements": 2121,
"max_abs": 0.001953125,
"bad_original_tolerance": 106,
"nonfinite": 0,
"native_SHA256": "786718ee172201d4bc7882837cdf419aba6e43754603e4f8a384e5555fa90231",
"fused_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"passed": false
},
"QKV_gradients": {
"dq": {
"elements": 3670016,
"exact_different_elements": 1964,
"max_abs": 0.001953125,
"bad_original_tolerance": 108,
"nonfinite": 0,
"native_SHA256": "8948319caaebc0a30749e14565cc45e7f7ff29631239575757777a9a08d05e03",
"fused_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"passed": false
},
"dk": {
"elements": 3670016,
"exact_different_elements": 1752,
"max_abs": 0.00390625,
"bad_original_tolerance": 124,
"nonfinite": 0,
"native_SHA256": "776385bd8a519ed9b506aa5a0b2336be19cbca11f6305f14465ce420814244e5",
"fused_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"passed": false
},
"dv": {
"elements": 3670016,
"exact_different_elements": 2717,
"max_abs": 0.00390625,
"bad_original_tolerance": 124,
"nonfinite": 0,
"native_SHA256": "b29f2ee8f1d0ec69cd1bdef7df0f56df7e8185a560e2a97430463118206efaed",
"fused_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"passed": false
}
},
"RNG_equal": true
},
"native_softmax_boundary_fusion": {
"boundaries": {
"QK_scores": {
"elements": 58720256,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "d684621d85bc202fe9ebf7501d49f78a67ae695240aed49ec4d0bf7c10f5c39e",
"fused_SHA256": "d684621d85bc202fe9ebf7501d49f78a67ae695240aed49ec4d0bf7c10f5c39e",
"passed": true
},
"FP32_probabilities": {
"elements": 58720256,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "631716ba35b3d66eecdd05ad643c54e30550cc7ed429c3b975163d205f2df96e",
"fused_SHA256": "631716ba35b3d66eecdd05ad643c54e30550cc7ed429c3b975163d205f2df96e",
"passed": true
},
"BF16_probabilities": {
"elements": 58720256,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "6d3edb15abde560ad69b69f30b2a1f48ef98d982be96054f6d8d22c9aa3a98b2",
"fused_SHA256": "6d3edb15abde560ad69b69f30b2a1f48ef98d982be96054f6d8d22c9aa3a98b2",
"passed": true
}
},
"attention_output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"fused_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"passed": true
},
"QKV_gradients": {
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"fused_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"fused_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"fused_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"passed": true
}
},
"RNG_equal": true
}
},
{
"rank": 1,
"full_fusion": {
"boundaries": {
"QK_scores": {
"elements": 58720256,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "d684621d85bc202fe9ebf7501d49f78a67ae695240aed49ec4d0bf7c10f5c39e",
"fused_SHA256": "d684621d85bc202fe9ebf7501d49f78a67ae695240aed49ec4d0bf7c10f5c39e",
"passed": true
},
"FP32_probabilities": {
"elements": 58720256,
"exact_different_elements": 20744589,
"max_abs": 1.1920928955078125e-07,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4a790cb6fea166e9ab5b3d7d55e1321cca6e58fbcb9ce956999aba2e39848dbc",
"fused_SHA256": "631716ba35b3d66eecdd05ad643c54e30550cc7ed429c3b975163d205f2df96e",
"passed": true
},
"BF16_probabilities": {
"elements": 58720256,
"exact_different_elements": 528,
"max_abs": 6.103515625e-05,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "cb7847ae94fd41c3728b598a9401c96c73e180d79a8f3f589f3e7bc63fb04f1f",
"fused_SHA256": "6d3edb15abde560ad69b69f30b2a1f48ef98d982be96054f6d8d22c9aa3a98b2",
"passed": true
}
},
"attention_output": {
"elements": 3670016,
"exact_different_elements": 2121,
"max_abs": 0.001953125,
"bad_original_tolerance": 106,
"nonfinite": 0,
"native_SHA256": "786718ee172201d4bc7882837cdf419aba6e43754603e4f8a384e5555fa90231",
"fused_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"passed": false
},
"QKV_gradients": {
"dq": {
"elements": 3670016,
"exact_different_elements": 1964,
"max_abs": 0.001953125,
"bad_original_tolerance": 108,
"nonfinite": 0,
"native_SHA256": "8948319caaebc0a30749e14565cc45e7f7ff29631239575757777a9a08d05e03",
"fused_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"passed": false
},
"dk": {
"elements": 3670016,
"exact_different_elements": 1752,
"max_abs": 0.00390625,
"bad_original_tolerance": 124,
"nonfinite": 0,
"native_SHA256": "776385bd8a519ed9b506aa5a0b2336be19cbca11f6305f14465ce420814244e5",
"fused_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"passed": false
},
"dv": {
"elements": 3670016,
"exact_different_elements": 2717,
"max_abs": 0.00390625,
"bad_original_tolerance": 124,
"nonfinite": 0,
"native_SHA256": "b29f2ee8f1d0ec69cd1bdef7df0f56df7e8185a560e2a97430463118206efaed",
"fused_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"passed": false
}
},
"RNG_equal": true
},
"native_softmax_boundary_fusion": {
"boundaries": {
"QK_scores": {
"elements": 58720256,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "d684621d85bc202fe9ebf7501d49f78a67ae695240aed49ec4d0bf7c10f5c39e",
"fused_SHA256": "d684621d85bc202fe9ebf7501d49f78a67ae695240aed49ec4d0bf7c10f5c39e",
"passed": true
},
"FP32_probabilities": {
"elements": 58720256,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "631716ba35b3d66eecdd05ad643c54e30550cc7ed429c3b975163d205f2df96e",
"fused_SHA256": "631716ba35b3d66eecdd05ad643c54e30550cc7ed429c3b975163d205f2df96e",
"passed": true
},
"BF16_probabilities": {
"elements": 58720256,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "6d3edb15abde560ad69b69f30b2a1f48ef98d982be96054f6d8d22c9aa3a98b2",
"fused_SHA256": "6d3edb15abde560ad69b69f30b2a1f48ef98d982be96054f6d8d22c9aa3a98b2",
"passed": true
}
},
"attention_output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"fused_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"passed": true
},
"QKV_gradients": {
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"fused_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"fused_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"fused_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"passed": true
}
},
"RNG_equal": true
}
}
],
"source_SHA256": {
"actual_attention_probe.py": "eb4b7094ced682760f1ee81eaef2d85f7c73c1a46bbb352480410c644e608e24",
"attention_fusion_adapter.py": "c1d30f33be782757ae5118511c9cfd84514078cba3735b1317319076a8d4133b",
"corex_process_guard_data.py": "a8f86db9e75eb353121c346011428400c4d4d1208e35a5d6c12ce48009dfd058",
"fused_scale_mask.py": "d1c9b7ea0f5b4e019854d20beb3f3e47d52ace2cd93254fbac771510978c178b",
"fused_scale_mask_softmax.py": "093e3a90fc652b7652938ced212107ada5861bceaccbe8d8bdcc008b29e6cd58",
"supervise_component.py": "9a420a085e05d50143ea53a028c0c8790946acb649372e817997ee0982852d41"
},
"both_ends_key_SHA256": "4301f21b7c20609388de96ac1bfb2856f1ffdbd8ec73fc6571184fa371f02576",
"all_expected_recomputation_and_layer_scaling_cases_passed": true,
"first_observed_divergence_in_captured_component": "QK scores identical; FP32 softmax probabilities differ; BF16 probabilities differ; downstream attention output and QKV gradients exceed original tolerance. Does not identify a specific hardware instruction.",
"raw_tensor_values_downloaded": false,
"local_independent_review_scope": "coverage,source bindings,all comparison summaries/SHA and original thresholds; numeric full arrays compared remotely on CPU float64",
"scope_note": "Actual unmodified MCore attention module and TP2 groups, syntheticQKV; complete official7B weights not loaded in this stage.",
"case_details": [
{
"rank": 0,
"layer_scaling": false,
"layer": 1,
"recompute": false,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": null,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"fused_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"fused_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"fused_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"fused_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 0,
"layer_scaling": false,
"layer": 1,
"recompute": true,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": null,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"fused_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"fused_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"fused_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"fused_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 0,
"layer_scaling": false,
"layer": 14,
"recompute": false,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": null,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "0c38e6541f6bdae19b7458455d2911b76276e24b00df9194c3e51f0f731744da",
"fused_SHA256": "0c38e6541f6bdae19b7458455d2911b76276e24b00df9194c3e51f0f731744da",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "f1d5ed7eb70599c4b4a628eaedcc77afe8593d4a7a5588ab9c8da9e2802253e6",
"fused_SHA256": "f1d5ed7eb70599c4b4a628eaedcc77afe8593d4a7a5588ab9c8da9e2802253e6",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4a5aeb1536d7096d5fbdf9c92ffb57d743b70bb2d4c6e8218ee631ad07581167",
"fused_SHA256": "4a5aeb1536d7096d5fbdf9c92ffb57d743b70bb2d4c6e8218ee631ad07581167",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "489a962c4a272bedd3c5f67dcb42b50a98d681d407a21fba6ac57fa1cf2574a4",
"fused_SHA256": "489a962c4a272bedd3c5f67dcb42b50a98d681d407a21fba6ac57fa1cf2574a4",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 0,
"layer_scaling": false,
"layer": 14,
"recompute": true,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": null,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "0c38e6541f6bdae19b7458455d2911b76276e24b00df9194c3e51f0f731744da",
"fused_SHA256": "0c38e6541f6bdae19b7458455d2911b76276e24b00df9194c3e51f0f731744da",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "f1d5ed7eb70599c4b4a628eaedcc77afe8593d4a7a5588ab9c8da9e2802253e6",
"fused_SHA256": "f1d5ed7eb70599c4b4a628eaedcc77afe8593d4a7a5588ab9c8da9e2802253e6",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4a5aeb1536d7096d5fbdf9c92ffb57d743b70bb2d4c6e8218ee631ad07581167",
"fused_SHA256": "4a5aeb1536d7096d5fbdf9c92ffb57d743b70bb2d4c6e8218ee631ad07581167",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "489a962c4a272bedd3c5f67dcb42b50a98d681d407a21fba6ac57fa1cf2574a4",
"fused_SHA256": "489a962c4a272bedd3c5f67dcb42b50a98d681d407a21fba6ac57fa1cf2574a4",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 0,
"layer_scaling": false,
"layer": 28,
"recompute": false,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": null,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "2ac030de6d99dc9a2b41ae17664b9b80e666eb6a2765f78d5354e23703b36043",
"fused_SHA256": "2ac030de6d99dc9a2b41ae17664b9b80e666eb6a2765f78d5354e23703b36043",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "947e7acfcca64f108fe4059436b5814309c640f19cb138ba42fa4a021a87c2b7",
"fused_SHA256": "947e7acfcca64f108fe4059436b5814309c640f19cb138ba42fa4a021a87c2b7",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "0312b9819b4dd7310abf5da3d397604a38c733c891c3ce2dd5c5d69be31a0049",
"fused_SHA256": "0312b9819b4dd7310abf5da3d397604a38c733c891c3ce2dd5c5d69be31a0049",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "9f6284554fb1c9cfda1b05714dc4f1459b6c50a18f56151e55abd4aa59f57858",
"fused_SHA256": "9f6284554fb1c9cfda1b05714dc4f1459b6c50a18f56151e55abd4aa59f57858",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 0,
"layer_scaling": false,
"layer": 28,
"recompute": true,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": null,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "2ac030de6d99dc9a2b41ae17664b9b80e666eb6a2765f78d5354e23703b36043",
"fused_SHA256": "2ac030de6d99dc9a2b41ae17664b9b80e666eb6a2765f78d5354e23703b36043",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "947e7acfcca64f108fe4059436b5814309c640f19cb138ba42fa4a021a87c2b7",
"fused_SHA256": "947e7acfcca64f108fe4059436b5814309c640f19cb138ba42fa4a021a87c2b7",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "0312b9819b4dd7310abf5da3d397604a38c733c891c3ce2dd5c5d69be31a0049",
"fused_SHA256": "0312b9819b4dd7310abf5da3d397604a38c733c891c3ce2dd5c5d69be31a0049",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "9f6284554fb1c9cfda1b05714dc4f1459b6c50a18f56151e55abd4aa59f57858",
"fused_SHA256": "9f6284554fb1c9cfda1b05714dc4f1459b6c50a18f56151e55abd4aa59f57858",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 0,
"layer_scaling": true,
"layer": 1,
"recompute": false,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": 1,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"fused_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"fused_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"fused_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"fused_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 0,
"layer_scaling": true,
"layer": 1,
"recompute": true,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": 1,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"fused_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"fused_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"fused_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"fused_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 0,
"layer_scaling": true,
"layer": 14,
"recompute": false,
"actual_QK_alpha": 0.006313453403451317,
"additional_softmax_scale": 14,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "5c84777c8bc3664e72c28d370af9583542d7ae17c1108fbecb054376836bb2a4",
"fused_SHA256": "5c84777c8bc3664e72c28d370af9583542d7ae17c1108fbecb054376836bb2a4",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "f0dddab72f9f36a2b0f0920c8aeff683915b06f564b588eccfa242feeeff5cd2",
"fused_SHA256": "f0dddab72f9f36a2b0f0920c8aeff683915b06f564b588eccfa242feeeff5cd2",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "c8ccc59c99f2ce2c6459aad46a4587c712570fa29bef4eca7b7b65cf4e4bdf2e",
"fused_SHA256": "c8ccc59c99f2ce2c6459aad46a4587c712570fa29bef4eca7b7b65cf4e4bdf2e",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "99a4df03fb5d918c0ef4606e31423c921872c0fa27579c9886fb3de1b1cb2877",
"fused_SHA256": "99a4df03fb5d918c0ef4606e31423c921872c0fa27579c9886fb3de1b1cb2877",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 0,
"layer_scaling": true,
"layer": 14,
"recompute": true,
"actual_QK_alpha": 0.006313453403451317,
"additional_softmax_scale": 14,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "5c84777c8bc3664e72c28d370af9583542d7ae17c1108fbecb054376836bb2a4",
"fused_SHA256": "5c84777c8bc3664e72c28d370af9583542d7ae17c1108fbecb054376836bb2a4",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "f0dddab72f9f36a2b0f0920c8aeff683915b06f564b588eccfa242feeeff5cd2",
"fused_SHA256": "f0dddab72f9f36a2b0f0920c8aeff683915b06f564b588eccfa242feeeff5cd2",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "c8ccc59c99f2ce2c6459aad46a4587c712570fa29bef4eca7b7b65cf4e4bdf2e",
"fused_SHA256": "c8ccc59c99f2ce2c6459aad46a4587c712570fa29bef4eca7b7b65cf4e4bdf2e",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "99a4df03fb5d918c0ef4606e31423c921872c0fa27579c9886fb3de1b1cb2877",
"fused_SHA256": "99a4df03fb5d918c0ef4606e31423c921872c0fa27579c9886fb3de1b1cb2877",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 0,
"layer_scaling": true,
"layer": 28,
"recompute": false,
"actual_QK_alpha": 0.0031567267017256584,
"additional_softmax_scale": 28,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "7200f1a4b77894b4c8ffa1dbd92fc96f8b3ebfebcf7d04e50c353cfbac1439c4",
"fused_SHA256": "7200f1a4b77894b4c8ffa1dbd92fc96f8b3ebfebcf7d04e50c353cfbac1439c4",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "b4ece9962c2b979d1f767ed2a80c62041d168ad9184d60e858a289fdcd5f20bb",
"fused_SHA256": "b4ece9962c2b979d1f767ed2a80c62041d168ad9184d60e858a289fdcd5f20bb",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "a94ecac04fa895c26a7cbb943b6f68a15145a9e3dcc9254791a530a3bc3a8a60",
"fused_SHA256": "a94ecac04fa895c26a7cbb943b6f68a15145a9e3dcc9254791a530a3bc3a8a60",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "c1adbed7eccbe91bd4266e7fd42f13341cf28831148e3676f165a0e2ce2ee13a",
"fused_SHA256": "c1adbed7eccbe91bd4266e7fd42f13341cf28831148e3676f165a0e2ce2ee13a",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 0,
"layer_scaling": true,
"layer": 28,
"recompute": true,
"actual_QK_alpha": 0.0031567267017256584,
"additional_softmax_scale": 28,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "7200f1a4b77894b4c8ffa1dbd92fc96f8b3ebfebcf7d04e50c353cfbac1439c4",
"fused_SHA256": "7200f1a4b77894b4c8ffa1dbd92fc96f8b3ebfebcf7d04e50c353cfbac1439c4",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "b4ece9962c2b979d1f767ed2a80c62041d168ad9184d60e858a289fdcd5f20bb",
"fused_SHA256": "b4ece9962c2b979d1f767ed2a80c62041d168ad9184d60e858a289fdcd5f20bb",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "a94ecac04fa895c26a7cbb943b6f68a15145a9e3dcc9254791a530a3bc3a8a60",
"fused_SHA256": "a94ecac04fa895c26a7cbb943b6f68a15145a9e3dcc9254791a530a3bc3a8a60",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "c1adbed7eccbe91bd4266e7fd42f13341cf28831148e3676f165a0e2ce2ee13a",
"fused_SHA256": "c1adbed7eccbe91bd4266e7fd42f13341cf28831148e3676f165a0e2ce2ee13a",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 1,
"layer_scaling": false,
"layer": 1,
"recompute": false,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": null,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"fused_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"fused_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"fused_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"fused_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 1,
"layer_scaling": false,
"layer": 1,
"recompute": true,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": null,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"fused_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"fused_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"fused_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"fused_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 1,
"layer_scaling": false,
"layer": 14,
"recompute": false,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": null,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "0c38e6541f6bdae19b7458455d2911b76276e24b00df9194c3e51f0f731744da",
"fused_SHA256": "0c38e6541f6bdae19b7458455d2911b76276e24b00df9194c3e51f0f731744da",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "f1d5ed7eb70599c4b4a628eaedcc77afe8593d4a7a5588ab9c8da9e2802253e6",
"fused_SHA256": "f1d5ed7eb70599c4b4a628eaedcc77afe8593d4a7a5588ab9c8da9e2802253e6",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4a5aeb1536d7096d5fbdf9c92ffb57d743b70bb2d4c6e8218ee631ad07581167",
"fused_SHA256": "4a5aeb1536d7096d5fbdf9c92ffb57d743b70bb2d4c6e8218ee631ad07581167",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "489a962c4a272bedd3c5f67dcb42b50a98d681d407a21fba6ac57fa1cf2574a4",
"fused_SHA256": "489a962c4a272bedd3c5f67dcb42b50a98d681d407a21fba6ac57fa1cf2574a4",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 1,
"layer_scaling": false,
"layer": 14,
"recompute": true,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": null,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "0c38e6541f6bdae19b7458455d2911b76276e24b00df9194c3e51f0f731744da",
"fused_SHA256": "0c38e6541f6bdae19b7458455d2911b76276e24b00df9194c3e51f0f731744da",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "f1d5ed7eb70599c4b4a628eaedcc77afe8593d4a7a5588ab9c8da9e2802253e6",
"fused_SHA256": "f1d5ed7eb70599c4b4a628eaedcc77afe8593d4a7a5588ab9c8da9e2802253e6",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4a5aeb1536d7096d5fbdf9c92ffb57d743b70bb2d4c6e8218ee631ad07581167",
"fused_SHA256": "4a5aeb1536d7096d5fbdf9c92ffb57d743b70bb2d4c6e8218ee631ad07581167",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "489a962c4a272bedd3c5f67dcb42b50a98d681d407a21fba6ac57fa1cf2574a4",
"fused_SHA256": "489a962c4a272bedd3c5f67dcb42b50a98d681d407a21fba6ac57fa1cf2574a4",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 1,
"layer_scaling": false,
"layer": 28,
"recompute": false,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": null,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "2ac030de6d99dc9a2b41ae17664b9b80e666eb6a2765f78d5354e23703b36043",
"fused_SHA256": "2ac030de6d99dc9a2b41ae17664b9b80e666eb6a2765f78d5354e23703b36043",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "947e7acfcca64f108fe4059436b5814309c640f19cb138ba42fa4a021a87c2b7",
"fused_SHA256": "947e7acfcca64f108fe4059436b5814309c640f19cb138ba42fa4a021a87c2b7",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "0312b9819b4dd7310abf5da3d397604a38c733c891c3ce2dd5c5d69be31a0049",
"fused_SHA256": "0312b9819b4dd7310abf5da3d397604a38c733c891c3ce2dd5c5d69be31a0049",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "9f6284554fb1c9cfda1b05714dc4f1459b6c50a18f56151e55abd4aa59f57858",
"fused_SHA256": "9f6284554fb1c9cfda1b05714dc4f1459b6c50a18f56151e55abd4aa59f57858",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 1,
"layer_scaling": false,
"layer": 28,
"recompute": true,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": null,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "2ac030de6d99dc9a2b41ae17664b9b80e666eb6a2765f78d5354e23703b36043",
"fused_SHA256": "2ac030de6d99dc9a2b41ae17664b9b80e666eb6a2765f78d5354e23703b36043",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "947e7acfcca64f108fe4059436b5814309c640f19cb138ba42fa4a021a87c2b7",
"fused_SHA256": "947e7acfcca64f108fe4059436b5814309c640f19cb138ba42fa4a021a87c2b7",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "0312b9819b4dd7310abf5da3d397604a38c733c891c3ce2dd5c5d69be31a0049",
"fused_SHA256": "0312b9819b4dd7310abf5da3d397604a38c733c891c3ce2dd5c5d69be31a0049",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "9f6284554fb1c9cfda1b05714dc4f1459b6c50a18f56151e55abd4aa59f57858",
"fused_SHA256": "9f6284554fb1c9cfda1b05714dc4f1459b6c50a18f56151e55abd4aa59f57858",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 1,
"layer_scaling": true,
"layer": 1,
"recompute": false,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": 1,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"fused_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"fused_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"fused_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"fused_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 1,
"layer_scaling": true,
"layer": 1,
"recompute": true,
"actual_QK_alpha": 0.08838834764831843,
"additional_softmax_scale": 1,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"fused_SHA256": "4f6ba59ed2a20d1c93fb9c1879520e050efcb25c1f261656711a79194c1b56b3",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"fused_SHA256": "64a7795df7bc962254ae008bbce26086d44944422a26171f9c2f28150f4b1fd5",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"fused_SHA256": "6c583fba0b161dd4348f4e41f11179af06fdbbdf15f6341934fc9614ee11a7d9",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"fused_SHA256": "1349d3122df8d20c7e7ca2c72a449cf0e6f874f69163c65466574fd4739bdb46",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 1,
"layer_scaling": true,
"layer": 14,
"recompute": false,
"actual_QK_alpha": 0.006313453403451317,
"additional_softmax_scale": 14,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "5c84777c8bc3664e72c28d370af9583542d7ae17c1108fbecb054376836bb2a4",
"fused_SHA256": "5c84777c8bc3664e72c28d370af9583542d7ae17c1108fbecb054376836bb2a4",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "f0dddab72f9f36a2b0f0920c8aeff683915b06f564b588eccfa242feeeff5cd2",
"fused_SHA256": "f0dddab72f9f36a2b0f0920c8aeff683915b06f564b588eccfa242feeeff5cd2",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "c8ccc59c99f2ce2c6459aad46a4587c712570fa29bef4eca7b7b65cf4e4bdf2e",
"fused_SHA256": "c8ccc59c99f2ce2c6459aad46a4587c712570fa29bef4eca7b7b65cf4e4bdf2e",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "99a4df03fb5d918c0ef4606e31423c921872c0fa27579c9886fb3de1b1cb2877",
"fused_SHA256": "99a4df03fb5d918c0ef4606e31423c921872c0fa27579c9886fb3de1b1cb2877",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 1,
"layer_scaling": true,
"layer": 14,
"recompute": true,
"actual_QK_alpha": 0.006313453403451317,
"additional_softmax_scale": 14,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "5c84777c8bc3664e72c28d370af9583542d7ae17c1108fbecb054376836bb2a4",
"fused_SHA256": "5c84777c8bc3664e72c28d370af9583542d7ae17c1108fbecb054376836bb2a4",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "f0dddab72f9f36a2b0f0920c8aeff683915b06f564b588eccfa242feeeff5cd2",
"fused_SHA256": "f0dddab72f9f36a2b0f0920c8aeff683915b06f564b588eccfa242feeeff5cd2",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "c8ccc59c99f2ce2c6459aad46a4587c712570fa29bef4eca7b7b65cf4e4bdf2e",
"fused_SHA256": "c8ccc59c99f2ce2c6459aad46a4587c712570fa29bef4eca7b7b65cf4e4bdf2e",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "99a4df03fb5d918c0ef4606e31423c921872c0fa27579c9886fb3de1b1cb2877",
"fused_SHA256": "99a4df03fb5d918c0ef4606e31423c921872c0fa27579c9886fb3de1b1cb2877",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 1,
"layer_scaling": true,
"layer": 28,
"recompute": false,
"actual_QK_alpha": 0.0031567267017256584,
"additional_softmax_scale": 28,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "7200f1a4b77894b4c8ffa1dbd92fc96f8b3ebfebcf7d04e50c353cfbac1439c4",
"fused_SHA256": "7200f1a4b77894b4c8ffa1dbd92fc96f8b3ebfebcf7d04e50c353cfbac1439c4",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "b4ece9962c2b979d1f767ed2a80c62041d168ad9184d60e858a289fdcd5f20bb",
"fused_SHA256": "b4ece9962c2b979d1f767ed2a80c62041d168ad9184d60e858a289fdcd5f20bb",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "a94ecac04fa895c26a7cbb943b6f68a15145a9e3dcc9254791a530a3bc3a8a60",
"fused_SHA256": "a94ecac04fa895c26a7cbb943b6f68a15145a9e3dcc9254791a530a3bc3a8a60",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "c1adbed7eccbe91bd4266e7fd42f13341cf28831148e3676f165a0e2ce2ee13a",
"fused_SHA256": "c1adbed7eccbe91bd4266e7fd42f13341cf28831148e3676f165a0e2ce2ee13a",
"passed": true
}
},
"RNG_equal": true,
"passed": true
},
{
"rank": 1,
"layer_scaling": true,
"layer": 28,
"recompute": true,
"actual_QK_alpha": 0.0031567267017256584,
"additional_softmax_scale": 28,
"metrics": {
"output": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "7200f1a4b77894b4c8ffa1dbd92fc96f8b3ebfebcf7d04e50c353cfbac1439c4",
"fused_SHA256": "7200f1a4b77894b4c8ffa1dbd92fc96f8b3ebfebcf7d04e50c353cfbac1439c4",
"passed": true
},
"dq": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "b4ece9962c2b979d1f767ed2a80c62041d168ad9184d60e858a289fdcd5f20bb",
"fused_SHA256": "b4ece9962c2b979d1f767ed2a80c62041d168ad9184d60e858a289fdcd5f20bb",
"passed": true
},
"dk": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "a94ecac04fa895c26a7cbb943b6f68a15145a9e3dcc9254791a530a3bc3a8a60",
"fused_SHA256": "a94ecac04fa895c26a7cbb943b6f68a15145a9e3dcc9254791a530a3bc3a8a60",
"passed": true
},
"dv": {
"elements": 3670016,
"exact_different_elements": 0,
"max_abs": 0.0,
"bad_original_tolerance": 0,
"nonfinite": 0,
"native_SHA256": "c1adbed7eccbe91bd4266e7fd42f13341cf28831148e3676f165a0e2ce2ee13a",
"fused_SHA256": "c1adbed7eccbe91bd4266e7fd42f13341cf28831148e3676f165a0e2ce2ee13a",
"passed": true
}
},
"RNG_equal": true,
"passed": true
}
],
"memory_placement_fixture": {
"CPU_rank_fixtures": 2,
"parameter_elements": 131072,
"steps_each": 3,
"fields_each_step": 4,
"max_abs": 0,
"moments_sampled_on_node0": true,
"affinity_restored": true,
"CUDA_initialized": false,
"both_ends_key_SHA256": "7a0d415aa9a5e8118b237116f78228c632ae0113cf971e46355b0d4f96b10ee6",
"source_SHA256": {
"node0_optimizer_state.py": "0f45bb906ceacbf1df1e3d3f785595238aa1edd75f57d50c56d04293fd0d45da",
"corex_process_guard_data.py": "a8f86db9e75eb353121c346011428400c4d4d1208e35a5d6c12ce48009dfd058",
"corex_adamw_local_moments.py": "beacd9b0a3004e99407cd1d60cb681c4fb426bad7fc132f85bbff2f4f8ae490d"
},
"scope": "small native-AdamW API/math/locality fixture; not7B or full-capacity acceptance"
},
"full_model_validation_status": {
"weights_and_three_real_updates_started": true,
"native_checkpoint_resume_started": false,
"e2e_measurement_started": false,
"full_model_Node1_initial_admission_GiB": 28,
"resource_layout_capacity_and_admission_review_pending": false,
"original_numeric_gate_unchanged": true,
"full_model_Node0_initial_admission_GiB": 291,
"explicit_capacity_based_admission_authorized": true,
"current_blocker": null,
"current_stage": "native real7B initialization and full-state audit running; three-arm completion pending",
"direct_launch_user_authorized": true
},
"measured_CPU_state_capacity": {
"capacity_passed": true,
"groups_per_rank": 199,
"elements_per_rank": 3834518016,
"actual_CPU_AdamW_updates_per_rank": 1,
"all_sampled_master_gradient_398_moment_pages_node0": true,
"peak_own_RSS_bytes": 176126173184,
"startup_required_GiB": {
"0": 291,
"1": 28
},
"derivation": {
"node0": "max(prior actual full-model own RSS peak, new capacity own RSS peak) +16GiB reserve +4*largest FP32 group, rounded up GiB",
"node1": "16GiB reserve +2*(sum rank node1 mapped milestones peak +coordinator entire RSS +64MiB supervisor) +4*largest FP32 group +8GiB allowance for later model/driver transients, rounded up GiB"
},
"runtime_NUMA_reserve_GiB": 16,
"own_RSS_cap_GiB": 292,
"per_rank_DATA_GiB": 144,
"host_reserve_GiB": 8,
"both_ends_key_SHA256": "63a8ff0857b8c07568a53875d50df3bd3e4e5dd567936d8e8cfe192462c0f6a0",
"source_SHA256": {
"corex_adamw_local_moments.py": "beacd9b0a3004e99407cd1d60cb681c4fb426bad7fc132f85bbff2f4f8ae490d",
"corex_process_guard_data.py": "a8f86db9e75eb353121c346011428400c4d4d1208e35a5d6c12ce48009dfd058",
"node0_capacity_worker.py": "f6a5e83d9bfd38955a6f0a86b47375d2ec755b2cdfe5607786b0a5a0b15eed89",
"node0_optimizer_state.py": "0f45bb906ceacbf1df1e3d3f785595238aa1edd75f57d50c56d04293fd0d45da",
"supervise_component.py": "f9d27959aeced06bad0cb0ad21cbaa31b3c4c869b6a58aeb7224452d57924e70"
},
"scope": "Full optimizer state capacity, no official7B model forward/backward or performance acceptance"
}
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
| softmax-fusion | `725d2ad52bd5dd4a4e5c875acefcc14410c61a3e3b62eb5b361e2b48773bc4f5` | 460320 |
| attention-integration | `c2f7b8f99d44adce01eaa9e945874f00a970da936110a66ba42392da1193ed3d` | 65112 |

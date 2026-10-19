# Thesis figure, table and narrative review

审阅对象：`/root/NTU_Dissertation_Reviewed_v6_editable.docx`、其24个原始嵌入PNG、`thesis_paragraphs.txt`。本报告只做原稿图表/叙事审阅；未修改docx，未读取实验代码或运行训练。预览仅写入`/root/dissertation_local_audit/thesis_review/`。

**定位方式：下文p编号为原稿零起始、Word body直接paragraph索引，与`thesis_paragraphs.txt`一致；原稿共836段。插入后索引会改变，执行时应先保留原XML节点引用或按原锚点文字匹配。不能把插入后的动态段号当原段号。**

## 1. 主要结论

1. 原稿24张图像并非24幅Figure：**5个独立编号Figure、18个Table图像对象（其中一个混入重复Figure）、1个封面校徽**。原docx的原生`w:tbl`数量是**0**。Table标题、表头和数字大量被烘焙进PNG，读屏、检索、编辑和表目录均受影响。
2. 原稿实际研究对象是**分层diagnostic evaluation、π0.5 flow-matching VLA、一个checkpoint/配置、5个inference flow seeds**。ACT仅为文献背景；不得加入ACT网络结构、CVAE损失、训练随机种子置信区间或end-to-end HRC成功图。
3. 最适合补强的是**benchmark如何构造**：真实head-camera任务storyboard、implemented FSM与replay/policy替换边界、QC与有效监督分布。建议核心总数为8幅：保留现有5图，新增上述3图。现有main results图不必全部推倒重画。
4. 紧急排版修复：p550的`image8.png`包含Table5.1和一幅截断的Figure5.1，p551的`image9.png`又把同一Figure完整重复一遍。p550应只放原生Table5.1，p551保留单幅Figure5.1。不能直接保留整张image8再插入image9。
5. Appendix A中Table A.1与A.3只剩图片表头，数据散成普通段落；Table A.2还落在A.2标题之后。应先修成完整原生表，再添加新图。

## 2. 全部现有图片与具体位置

| 图片 | 像素尺寸 | 原稿位置 | 实际类别/内容 | 建议 |
|---|---:|---|---|---|
| image1.png | 836×305 | p0，封面 | NTU校徽/校名，不是研究Figure | 保留，不计入List of Figures |
| image2.png | 943×627 | p262；caption p263；§3.1引入p261 | Figure3.1 分层诊断框架 | 保留；Level5虚线且注明historical supporting evidence |
| image3.png | 943×775 | p279；caption p280；§3.2.1引入p278 | Figure3.2 两轮任务与决策窗口/chunk示意 | 保留；不把宽度解释成时长；这是schematic |
| image4.png | 999×267 | p294；§3.2.3正文p291–293 | Table3.1 controller scope | 转原生Word表；caption从图内移到独立表题 |
| image5.png | 879×327 | p304；§3.2.4 | Table3.2 source-level split | 转原生表；当前割裂p303/305一句话，移至p306之后 |
| image6.png | 1058×462 | p472；§4.2末尾，§4.3前 | Table4.1 main policy configurations | 转原生表；保留A→B双改变、B→C state/action共同改变 |
| image7.png | 1058×462 | p518；§4.8末尾 | Table4.2 protocol populations | 转原生表；108 validation与216 train coverage不可合并成validation |
| image8.png | 1058×741 | p550；§5.1末尾 | **混合截图：Table5.1 + 重复/被截断的Figure5.1** | 仅重建Table5.1；删除该对象中的图部分 |
| image9.png | 880×440 | p551；caption p552；§5.1 | Figure5.1 frustum coverage单图 | 保留该单图；与image8重复；可提高输出分辨率 |
| image10.png | 1058×333 | p560；§5.2 | Table5.2 unified audit | 转原生表；当前割裂p559/561一句话，移至p556之后或p554之后 |
| image11.png | 902×396 | p564；caption p565；§5.2 | Figure5.2 pickup/radio arm skill | 保留；横轴两panel不同scale必须说明 |
| image12.png | 1058×446 | p570；§5.3 | Table5.3 gripper precision/recall/F1 | 转原生表；注意每seed仅14 pickup/9 radio expert transitions，来自重叠chunks的pool |
| image13.png | 1058×333 | p580；§5.4 | Table5.4 lookup/persistence baselines | 转原生表；Lookup B也对应C20 |
| image14.png | 1058×311 | p590；§5.5 | Table5.5 conditioning mean与anchor bootstrap interval | 转原生表；当前割裂p589/591一句话，移至p588后、p589前 |
| image15.png | 902×469 | p596；caption p597；§5.5 | Figure5.3 conditioning ratios | 保留；n=96/12是anchors，不是独立trajectories |
| image16.png | 1057×137 | p789；§A.1 | Table A.1 **只有caption与表头** | 与p790–791数据重组为完整原生表 |
| image17.png | 853×327 | p794；原稿却已进入§A.2 | Table A.2 Round1 durations | 转原生表，并移至p792之后、p793标题之前 |
| image18.png | 1003×133 | p795；§A.2 | Table A.3 **只有caption与表头** | 与p796–802数据重组；p801为重复“Parameter Value”表头文本 |
| image19.png | 1059×268 | p804；§A.2 | Table A.4 config/experiment IDs | 转原生表；等宽字体并保持下划线；directory=20000 |
| image20.png | 611×327 | p810；§A.3 | Table A.5 frozen audit manifest | 转原生表；652匹配验证anchors |
| image21.png | 1058×301 | p815；§A.4 | Table A.6 raw arm MAE | 转原生表；保留±SD；wait/follow无动作分母极小 |
| image22.png | 1058×421 | p820；§A.5 | Table A.7 predicted/matched gripper transitions | 转原生表；不要从mean counts重新构造mean F1 |
| image23.png | 946×397 | p823；§A.6 | Table A.8 full conditioning summary | 转原生表；表内“95% anchor BI”可在caption解释为bootstrap interval |
| image24.png | 983×234 | p825；§A.6 | Table A.9 extra image perturbations | 转原生表；仅validation pickup，不混入coverage subset |

## 3. 核心8幅Figure：位置、英文caption与正文引用

这是**8幅核心成稿图**，包含5幅现有图和3幅新增图，不是要求再加8幅。Chapter3新增图不改变Chapter5现有编号。所有英文文案都以当前稿件内容为依据；新截图、源级统计和实际FSM节点名称须经主代理核验再落稿。

### 1) 保留Figure3.1：diagnostic hierarchy

- **位置**：继续使用p262，caption p263；正文引入p261。不另画第二个诊断阶梯。
- **目的**：开宗明义呈现本论文贡献，区分controlled final audit与historical rollout。
- **Caption**: “Hierarchical diagnostic evaluation framework. Levels 1–4 form the controlled final audit of supervision, behavioural baselines, stochastic inference and observation conditioning. Level 5 summarises supporting historical closed-loop evidence and is not a matched A20/B20/C20 rollout comparison.”
- **正文引用段落**（可替换p261）: “Figure 3.1 organises the evaluation by increasing evidential demands. The first four levels test alternative explanations for favourable teacher-forced scores under a controlled final protocol. Historical closed-loop diagnostics provide a separate behavioural check at the fifth level.”

### 2) 保留Figure3.2：task sequence与监督定义

- **位置**：继续p279/p280，引入p278。
- **目的**：解释两轮、三次收罐循环、shift-1 target、±16 decision window。它不是实际场景照片，也不等于implemented FSM。
- **Caption**: “Task structure and supervision annotation. (a) Fixed task order in the two rounds of each retained HRC source; segment widths do not represent duration. (b) An illustrative expert gripper transition, its ±16-frame decision window and a 32-step future action target beginning at t+1. A B/C chunk may cross a local sub-instruction boundary within an unchanged coarse context.”
- **正文引用段落**: “Figure 3.2 separates task order from supervision construction. The task sequence is fixed, whereas segment durations vary across sources. The lower panel illustrates how one anchor supplies a future action chunk and how gripper transitions define decision-frame annotations.”

### 3) 新增Figure3.3：真实head-camera task storyboard

- **位置**：原p282之后、p283（3.2.2 Demonstration Provenance）之前；引文放p282后、新图前。不要抢占p279现有schematic的位置。
- **素材**：用户明确有收拾易拉罐、跟随人、开收音机视频，可从中选**4–6张真正录制的head-camera帧**。优先四个coarse contexts：wait、pickup、follow、radio；若4张不足以说明pickup，可6panel含approach、grasp、deposit。标注round/task/sub-instruction，同时保存source ID、raw frame/time、图像通道与抽帧位置。
- **重要边界**：只有确认视频属于CLEAN/source数据后，才能称“retained dataset”。未知来源时写“recorded development demonstrations”，并说明是否计入45条。不拼接成同一连续episode；不同source/round明确标注。画面上的成功行为来自recorded replay/scripted controller，不能写成learned policy成功。
- **Caption（已核验为dataset视频时）**: “Representative head-camera observations from the recorded HRC demonstrations. The panels illustrate the wait, pickup, follow and radio contexts used for diagnostic grouping. These frames document the dataset and scripted/replayed task execution; they are not evidence of successful control by the fine-tuned policies. Source, round and frame identifiers are provided for each panel.”
- **正文引用段落**: “Figure 3.3 provides representative recorded observations for the four coarse contexts. The images show the visual setting and task variation underlying the benchmark. They are examples of dataset execution rather than a qualitative evaluation of the learnt controllers.”
- **若截图来自第三人称视频**：不能沿用“head-camera observations”。改成“Representative recorded views of the HRC task”，并单独注明viewpoint；最好另配真实模型head-camera crop，不误称模型输入。

### 4) 新增Figure3.4：implemented FSM、controller boundary与数据流

- **位置**：原p294的Table3.1之后、p295（3.2.4 Population and Quality Control）之前；现有p291引Table3.1可保留，新增引文紧跟表后。
- **内容约束**：高层FSM节点名称/边必须取自实际实现。重点画：scripted human-facing/transitions；ExecuteBehaviorTask接口；dataset时BEHAVIOR state/action replay、rollout时π0.5 action server；30Hz observations/state/actions到export→source-level QC/split→训练/audit。只画能核实的接口，避免凭叙事臆造FSM节点。
- **不重复原则**：Figure3.1回答“如何诊断”；Figure3.2回答“任务按什么顺序/怎么标chunk”；Figure3.4回答“实现里谁控制/数据从哪来/哪里换成policy”。若实现证据不足，缩成controller boundary/data provenance图，或只保留原生Table3.1，不硬添FSM。
- **Caption**: “Implemented controller scope and data flow. Scripted finite-state logic controls the human-facing and transition phases. During dataset construction, ExecuteBehaviorTask replays upstream BEHAVIOR state–action trajectories; during policy rollout, the same interface receives predicted action chunks from the π0.5 controller. Recorded observations, robot states and actions are exported with source identity for dataset construction and diagnostic evaluation.”
- **正文引用段落**: “Figure 3.4 locates the learnt controller within the implemented HRC system. The policy replaces replay only inside ExecuteBehaviorTask, while the surrounding interaction and transition logic remains scripted. The data-flow branch shows how source identity is retained from recording to the diagnostic benchmark.”

### 5) 新增Figure3.5：benchmark构造、时长与有效监督

- **位置**：原p311之后、p312（3.2.5 Observation, State and Action Spaces）之前。该位置结束dataset construction/split说明；正文引文可置p311后。
- **建议3panel**：(a) 65 collected→45 CLEAN/19 REJECT/1 TRIM→37 train/4 val/4 test，明确只有45 CLEAN进入数据；(b) 45 sources的round/segment duration分布；(c) raw frames→valid anchors→sampling target的区别，可用context/segment的raw-vs-valid数量或3.955% natural decision anchors→20% configured B/C decision target。若(c)实测valid counts未核验，仅画已知natural-vs-configured target，绝不标作realised training frequencies。
- **数据范围**：distribution必须来自源级真实记录；只有min/median/p90/max时用range/quantile plot，不能伪造boxplot quartile或individual points。不同segment中位数不应相加冒充median total。validation候选15311/2498/11526/52与frustum valid pickup12302有不同population，不能强行画同一漏斗。
- **Caption（3panel均取得核验数据时）**: “Dataset construction and effective supervision. (a) Source-level quality-control outcomes and the frozen split of the 45 retained sources. (b) Recorded task-duration distributions across retained sources. (c) The distinction between the exported anchor population and the configured B/C sampling target. The 20% decision-frame share is a sampler target rather than a measured frequency of realised training draws.”
- **正文引用段落**: “Figure 3.5 summarises the construction of the case-study dataset and the supervision available to the learner. Quality control and splitting operate at source level, whereas chunk validity and weighted sampling operate on derived anchors. These stages preserve source separation while changing the distribution of training examples.”

### 6) 保留Figure5.1：validation geometric frustum coverage

- **位置**：p551图、p552题；p550只留独立原生Table5.1。引文可加在p549末尾，当前正文未明确点名Fig5.1。
- **Caption**: “Validation pickup target-centre geometric frustum coverage by task segment. Bars show the proportion of valid anchors whose assigned target centre lies inside the head-camera frustum. This geometric test does not model occlusion, object extent or rendered-pixel visibility.”
- **正文引用段落**: “Figure 5.1 shows that out-of-frustum anchors are concentrated in the walk/acquire and soda-can pickup segments. Overall validation coverage is 88.69% among the defined pickup anchors. This is a statement about target-centre geometry, not guaranteed pixel visibility.”
- **限制**：图本身是segment aggregate，**不能仅凭此图证明coverage随时间恢复**。正文34/45 onset与overall anchor coverage已经支持不同population的temporal描述，但若升级为early/middle/late curve，须真实记录并清楚标population；不是最优先新增图。

### 7) 保留Figure5.2：arm skill vs no motion

- **位置**：p564/p565；正文新增引用在p563末尾。Table5.2修复到p554或p556之后，勿打断句子。
- **Caption**: “Arm skill relative to the no-motion baseline on the frozen validation audit. Points and error bars show the mean and sample standard deviation over five matched inference flow-noise seeds for one trained checkpoint per configuration. Pickup and radio panels use different horizontal scales; zero denotes no-motion performance.”
- **正文引用段落**: “Figure 5.2 places the arm predictions against a meaningful no-motion reference. Pickup skill remains close to zero for all three checkpoints, while radio skill is negative. Variation across the five flow seeds describes stochastic inference, not uncertainty from independent training runs.”

### 8) 保留Figure5.3：conditioning ratios

- **位置**：p596/p597；正文引用在p595后、新图前。Table5.5移到p588后，修复句子。
- **Caption**: “Validation observation-conditioning ratios for torso and arm outputs. Points show mean per-anchor perturbation/noise response ratios, with 95% anchor-level percentile-bootstrap intervals. Pickup and radio contain 96 and 12 anchors, respectively. The dashed line marks the internal flow-noise reference and is not a universal significance threshold.”
- **正文引用段落**: “Figure 5.3 compares local image- and state-replacement responses with the policy's own flow-noise response. Mean image/noise ratios remain below one in both contexts. The intervals describe the frozen anchor set and do not establish that vision is unnecessary for task completion.”

## 4. 两项候选图：最多按需要选，勿挤压benchmark主线

### 9) 训练曲线：优先附录而非main results

- **推荐位置**：Appendix A.2，完整原生Table A.3结束后（原p802之后、p803之前），临时编号Figure A.1。若需主文展示训练过程，可放§4.2原p471后、Table4.1之前，编号Figure4.1。
- **画什么**：真实training/monitored validation flow-loss vs optimiser updates，标单次seed42、checkpoint20k、训练记录的真实采样频率；A/B/C各panel或明确独立纵轴。不能把logged loss当physical MAE或task-success曲线；B/C的normalisation不同、padding全部32维参与loss，故纵轴不构成统一物理performance尺度。
- **Caption**: “Training diagnostics for the three 20,000-update runs. Curves show the logged normalised flow-matching objective and the monitored validation objective for one training run per configuration. All 32 model output dimensions, including padding, contribute to this objective. Different normalisation pipelines prevent interpreting the loss magnitudes as directly comparable physical-action errors or task-success measures.”
- **正文引用段落**: “Figure A.1 records the optimisation history of the archived runs. It documents training progress and the fixed 20,000-update endpoint; the paper-facing comparison is instead defined by the frozen physical-action audit.”
- **若没有完整训练日志**：不按Table4.1造曲线，也不把checkpoint文件编号伪装成loss。

### 10) 事件prediction图：如需要增强结果可视化，优先替代而非重复主文表

- **位置**：§5.3原p577之后、p578（§5.4）之前，若主文采用编号Figure5.3、后续conditioning改为Figure5.4；若保持原3个main results Figure编号，则放Appendix A.5原p820表后，编号Figure A.2（若A.1存在）。
- **画法**：pickup/radio两panel，precision/recall/F1作为不同metric点或柱；各checkpoint mean±inference-seed SD。为了避免重复信息，主文只画F1+precision重点，完整P/R/F1留原生Table5.3或附录。不画grasp success。
- **Caption**: “Gripper-event prediction under the frozen validation protocol. Values are means with sample standard deviations over five inference flow-noise seeds, after pooling event counts within each context. Higher event F1 does not establish reliable grasp timing or successful object manipulation.”
- **正文引用段落**: “The event comparison shows a narrow improvement over A20: B20 and C20 detect more expert transitions, while most predicted pickup transitions remain unmatched. This supports improved event prediction within the frozen protocol, rather than a claim of successful grasping.”

其他统计候选（base-persistence比较、normalisation motion-vs-skill、seed scatter）在现有Table5.2–5.4与Figure5.2已基本覆盖。若篇幅有限，不必另增。特别是base persistence是重要**结论**，但不必为每个结论再加图。

## 5. 原生Word表格替换：恢复策略与可准确转录范围

所有18个表对象都适合替换成原生表；其中**16个有完整、可直接目视准确转录的表格**；A.1/A.3的PNG仅表头，完整内容已存在于普通段落，可据该文本恢复，不能声称图像里已有这些数据。此次查看是原图目视核对与XML正文读取，**未凭空调用OCR补值**。数值的实验真实性仍以主代理的归档证据核查为准；这里的“准确”仅指忠实恢复原稿。

| 表 | 转录来源 | 特殊处理 |
|---|---|---|
| 3.1, 3.2 | image4, image5全文清晰 | 3.2移出跨段句子；训练/验证/测试用途不能缩成全体policy test |
| 4.1, 4.2 | image6, image7全文清晰 | 原生表允许行内换行，保留长单元格完整语义 |
| 5.1 | image8上半部 | 下半图删除；标题与Figure5.1题分离 |
| 5.2, 5.3, 5.4, 5.5 | image10,12,13,14全文清晰 | ±使用真实字符；负号不漏；5.5用[lower, upper]独立列亦可 |
| A.1 | image16表头+p790–791文本 | 10行含total；完整恢复见下 |
| A.2 | image17全文清晰 | 放回§A.1内，位于p792与p793之间 |
| A.3 | image18表头+p796–802文本 | 合并跨页数据；删除p801重复表头；完整恢复见下 |
| A.4 | image19全文清晰 | config/experiment ID原样，注意“0”与“o”、下划线 |
| A.5 | image20全文清晰 | selected total=652 |
| A.6 | image21全文清晰 | 保留6位小数；正文更高精度数值不是与表矛盾 |
| A.7 | image22全文清晰 | mean predicted/TP不可直接反算mean precision/F1；应保留原有指标 |
| A.8 | image23全文清晰 | 8行；mean/median/std/BI/n不混列 |
| A.9 | image24全文清晰 | B20/C20各1行；4类image perturbation |

### 可直接恢复的Table A.1

表题：Round 0 expert segment durations.（45 retained sources；frames at 30Hz）

| Segment | Min | Median | 90th percentile | Max |
|---|---:|---:|---:|---:|
| Walk to / acquire trash can | 437 | 880 | 1242 | 1479 |
| Pick up soda can 1 | 446 | 992 | 1472 | 1974 |
| Deposit can 1 | 105 | 182 | 292 | 332 |
| Pick up soda can 2 | 254 | 599 | 941 | 1409 |
| Deposit can 2 | 88 | 201 | 284 | 368 |
| Pick up soda can 3 | 278 | 517 | 900 | 1189 |
| Deposit can 3 | 79 | 182 | 321 | 379 |
| Put trash can down | 50 | 240 | 338 | 533 |
| Stand by | 46 | 397 | 544 | 784 |
| Round 0 total | 2505 | 4555 | 5115 | 5197 |

### 可直接恢复的Table A.2

| Segment | Min | Median | 90th percentile | Max |
|---|---:|---:|---:|---:|
| Walk to / acquire trash can | 797 | 1429 | 1906 | 2321 |
| Turn on radio receiver | 72 | 146 | 256 | 372 |
| Put trash can down | 185 | 342 | 488 | 701 |
| Stand by | 63 | 324 | 500 | 548 |
| Round 1 total | 1547 | 2277 | 2655 | 3227 |

### 可直接恢复的Table A.3

| Parameter | Value |
|---|---|
| Base model | π0.5 flow-matching VLA |
| Physical state/action dimensions | 23 |
| Model action dimensions | 32 |
| Action horizon | 32 |
| Head-image model input | 224 × 224, resized with padding |
| Combined prompt/state token budget | 200 tokens |
| State input | Discrete-state input enabled |
| Fine-tuning | LoRA variants with frozen visual tower |
| EMA | Disabled |
| Training seed | 42; one run per configuration |
| Optimiser | AdamW |
| AdamW β1, β2 | 0.9, 0.95 |
| AdamW ε | 10^-8 |
| Weight decay | 10^-10 |
| Global gradient-norm clipping | 1.0 |
| Warm-up | 1,000 updates |
| Peak learning rate | 2.5 × 10^-5 |
| Cosine schedule length | 30,000 updates, including warm-up |
| End-of-schedule learning rate | 2.5 × 10^-6 |
| Physical microbatch size | 2 |
| Gradient accumulation | 4 microbatches per update |
| Effective batch size | 8 |
| Main training length | 20,000 optimiser updates |
| Flow-loss dimensions in A/B/C | All 32, including padding |
| Training-time validation | Every 1,000 updates; 10 × 2 examples |
| Checkpoint save / retention intervals | 1,000 / 5,000 updates |
| Offline action integration | 10 Euler steps |
| Main audit flow-noise seeds | 0, 1, 2, 3, 4 |
| Conditioning reference / alternate seeds | 0 / 1 |
| Protocol seed | 20260907 |
| Conditioning bootstrap resamples | 10,000 |

### 关键原生表恢复的数值检查点

- Table3.2：train 37 / 341269 frames；validation 4 / 31084；test 4 / 37317；total45 / 409670。
- Table5.1：walk/acquire 3569 valid、2916 in-frustum、0.8170；soda pickup6792、6054、0.8913；deposit1560、1560、1.0000；put-down381、381、1.0000。合计12302、10911，与正文88.69%一致。
- Table5.2 pickup arm skill：A20 −0.0107±0.0194；B20 0.0122±0.0231；C20 0.0144±0.0274。pickup motion ratio 0.4034±0.0231 / 0.3792±0.0220 / 0.4736±0.0390。radio arm skill −0.1752±0.0336 / −0.1670±0.0212 / −0.3099±0.0394。
- Table5.3 pickup F1：0.1120±0.0445 / 0.3358±0.0810 / 0.3825±0.0461。最高pickup precision0.3163±0.0427，不能说“可靠抓取”。radio F1：0.0235±0.0526 / 0.1768±0.1025 / 0.2181±0.1293。
- Table5.4 persistence pickup/radio/follow base skill：0.5402 / 0.5787 / 0.5093（Lookup A/B + persistence均相同）。不是只比zero base。
- Table5.5 image/noise B20 pickup0.3581[0.2969,0.4257]、C20 pickup0.4414[0.3349,0.5852]；B20 radio0.2851[0.2103,0.3723]、C20 radio0.4378[0.2873,0.6245]。对应n96、96、12、12。
- Table A.5 candidates/selected：pickup15311/200；follow2498/200；radio11526/200；wait52/52；total652。与Table5.1的12302不同是population定义不同，不应自动当算错。

## 6. 章节结构与主张校核

原稿结构清楚：Ch1背景/RQ；Ch2文献与gap；Ch3框架、dataset、监督、baseline、audit方法；Ch4实际π0.5配置与冻结protocol；Ch5结果；Ch6按RQ解释与限制；Ch7结论；Appendix细节。新增真实任务图与构造图应集中Ch3，训练日志放Ch4或Appendix；不要把所有技术图堆Ch5。

必须在新图/表/引文中保留的事实边界：

- **数据来源**（p283–289）：project-constructed HRC = scripted collaborative phases + replayed upstream human-teleoperated BEHAVIOR manipulation。不是新招募参与者的45条完整teleoperation，也不是全部fully scripted expert。
- **policy控制范围**（p291–293、441–447）：仅ExecuteBehaviorTask；human-facing/navigation transitions由scripted FSM控制。Figure3.3照片不能变成end-to-end自主HRC成功证据。
- **split**（p302–306）：37/4/4按source，继承原split；不是随机frame split。test只用于明确标注的descriptive geometry，不用于paper-facing policy performance。
- **A/B/C干预**（p474–496）：A→B同时变prompts、sampling、export eligibility与各自统计配置，不是单一language ablation；B→C state和action normalisation一起变，不是仅state tokenizer。
- **输出与loss**（p321–322、466–469）：32 model dims、23 physical dims、18 torso/arm group；32维包括padding进loss，23维用于paper metrics。训练曲线不能标“23-DoF MAE”。
- **采样**（p348–355、542）：3.955%是natural exported anchor fraction；20%是target distribution，不是已证明每个训练run实际恰好抽到20%。160000 sampled examples是draws，不是unique examples或epochs。
- **stochastic unit**（p395–397）：5个flow seeds，one checkpoint/config。误差棒不代表重复training runs；没有training-seed reliability结论。
- **protocol**（p401–426、501–517）：main manifest652、conditioning validation108（pickup96/radio12）、coverage216。conditioning要求prompt-stable；teacher-forced允许coarse context内local prompt切换；图不能画成同一anchor population。
- **frustum**（p428–437）：target centre几何覆盖，不等于occlusion-free visibility或detector success；34/45 onset out-of-frustum与overall86–91% in-frustum并不矛盾。
- **arm**（p558–567）：near-zero pickup skill；radio negative。更多motion≠更准确。低raw MAE≠学会控制。
- **events**（p572–577）：F1提升是窄结论；多数predicted transitions unmatched；不能从图中推task completion/grasp success。
- **conditioning**（p591–603）：mean image response < flow-noise response，是局部敏感度；不能写“vision unused/unnecessary”。anchor bootstrap不是source-level statistical population confidence。
- **rollouts**（p612–616）：5个historical metric files，zero contact；单例8→1 execution prefix减少collision，不是final A/B/C matched trial，不可画横向排名图。

## 7. Word落地检查建议

- 每张图首次在正文点名引用后再出现；避免在句子中间插表。上文已指出p304、p560、p590的跨段句子问题。
- 图下caption、表上caption；Table标题应成为独立可编辑文字，不能继续藏在PNG里。图内无需再重复完整“Figure X:”标题。
- 新增Figure3.3–3.5后更新List of Figures与交叉引用；若无Chapter5新增主图，原Figure5.1–5.3编号保持。
- 截图带panel字母和简短task标签；细节放caption。保存未裁原帧与source/frame表便于追溯。不要用AI生成机器人场景冒充实验截图。
- 优先PNG高分辨率实拍/录制帧，统计图用矢量原始输出或足够分辨率PNG；现有约900像素宽统计图放满页会偏软，应从源记录重导出，而非插值锐化。
- 原稿AI disclosure p19–21已说明schematic和统计重绘；若使用实际视频抽帧，更新为明确区分recorded screenshots、schematic diagrams与plots from frozen records，不能把新截图笼统称AI-generated figures。

本报告的截图分类、段落位置、原生表数量与现有表格读数均已直接检查原docx/XML及原PNG。新增FSM、视频截图、duration/source分布与training curves尚须主代理按实际归档材料核验后实现。

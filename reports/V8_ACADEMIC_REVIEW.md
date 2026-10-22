# v8 academic review: minimal figure and cross-reference edits

审阅对象：`/root/dissertation_figures_delivery/NTU_Dissertation_Reviewed_v7_figures_editable.docx`。

SHA256：`e24f23f635b43b97297273ae1f85b66d97b2b09a39e971dee9725b76694afbb9`。读取结果：821个直接body paragraphs、18张原生表、9个现有Figure captions（3.1–3.5、5.1–5.3、A.1）。仅审阅v7正文／表格及用户附件；未修改docx，也未重复数据／代码研究审计。

**所有p索引均为当前v7的`Document(...).paragraphs`零起始索引，不是v6索引，且不包含table cells。新图插入后应以精确文本／原XML节点定位，不能继续使用变化后的索引。**

## 1. 核心结论与最小修改集合

当前Chapter3–7的主要结论和证据限制已经明确，不需要重写研究主线。所需改动是：

1. 在§3.3末插入Figure3.6与一段简短引用／解释；在§6.2补一次RQ1回引。
2. 在§5.3结果解释结束后插入event图与短引用／caption；在§6.3补一次RQ2回引，保留原表5.3全部数值。
3. 处理新增event图与原conditioning图的编号顺序冲突，并同步figure captions、正文、TC图目录域及traceability文件；不能改错section/table编号。
4. 修复Figure5.1前一句的validation／all-source指代歧义；caption去除未由图表示的early-position暗示。
5. 用最小词语修改把重复出现的3.955%明确为**valid B/C training anchors**，避免与新增图的all45描述统计混用。
6. AI disclosure只同步真实图号／来源；官方EEE最新版仍按独立模板报告标Pending，保留真实使用事实和签字占位。

第1、2项是新增图的必要同步；第3项取决于主代理向用户提出的编号选项；第4、5项为小范围准确性修正。Chapter7、abstract、References、18表均无需因新增图而改结论或重编号。

## 2. Figure3.6的精确锚点与英文文案

### 推荐位置：§3.3末，现p348之后、p349之前

当前p348完整文本：

> The distinction is central to RQ1. The policy is trained on a sampler distribution, not directly on the raw trajectory distribution.

后继p349完整文本：

> 3.4 Diagnostic Baselines

此时§3.3已经解释shift-1、32-step chunk、coarse context、local prompt、decision frames与sampler；新统计图能把这些定义落到真实监督结构。放在p315标题前会先给出尚未定义的anchor／decision分布，逻辑较弱；也不应紧贴Figure3.5再画一次QC／总时长／3.955→20%的相同图。

建议结构：保留p348→增加引用解释段→Figure3.6→caption→原p349。

### 英文引用解释段

以下文字适合**segment-duration＋raw-frame/valid-anchor**的最小两panel方案；最终须与实际panel和统计总体一致：

“Figure 3.6 relates the recorded task structure to the supervision defined in this section. Segment durations and the availability of valid anchors determine how much usable supervision each phase contributes. These descriptive characteristics motivate RQ1: recorded trajectory length, eligible training anchors and the sampler distribution describe different aspects of the learning problem.”

若第三panel确实展示gripper events／decision anchors，可在第二句后加：

“Gripper-transition annotations additionally identify brief decision regions within the long trajectories.”

不要在未取得action arrays统计时加入“most arm actions are zero”“the model learns inactivity”等经验结论。时长、anchors或event的描述统计只支持监督结构诊断，不单独证明checkpoint行为的因果来源。

### 英文caption：两panel可落稿版本

**下文以两个panel都覆盖同一45条CLEAN sources为条件；如果实际脚本采用training-only或不同mask，必须替换相应总体说明。**

“Figure 3.6: Temporal and action characteristics of the HRC demonstration dataset. (a) Recorded task-segment durations across the 45 retained sources, converted from frames to seconds at 30 Hz. (b) Raw recorded frames and valid B/C anchors by coarse context over the same source population. Segment durations use segments as the observation unit; frame and anchor counts use the populations stated in the panel labels. Chunk validity and idle-tail exclusions follow the B/C export recipe. These are descriptive dataset statistics.”

如果(a)画的是同一segment跨45sources的boxplot，应明确每组45个**source-segment durations**，并确认脚本确实取得全部单源数值。若仅有min／median／p90／max，只能画相应quantile/range图，不能把它们当quartiles或伪造individual dots。

如果真实图增加第三panel，caption相应加一种**与实图完全一致**的句子：

- event counts版本：“(c) Recorded gripper-transition counts per source and task round.”
- decision-anchor版本：“(c) The fraction of valid anchors carrying decision-frame annotations within each labelled population.”

如果最终第二panel改成motion magnitude，须另外定义：absolute command还是anchor-relative delta、哪些物理维度、聚合维度／时间方法、阈值与单位。不能保留上面raw-frame/valid-anchor版本caption。

### RQ1回引：只加一句，不改原论证

§6.2现p630结束于：

> visible in the standby phases, where raw duration and usable-anchor coverage differ strongly. Episode length alone is therefore a poor measure of useful supervision.

可在该段末加：

“The descriptive distributions in Figure 3.6 make this distinction between recorded duration and eligible supervision explicit.”

若新图未实际展示raw-vs-valid，改在p628末回引**实际展示的segment duration或event统计**，不要让引用承载图中没有的证据。

## 3. Event figure的精确锚点、文案与编号映射

### 推荐位置：§5.3末，现p568之后、p569之前

p567–568是一句跨段文本，不能在两者之间插图。p568精确文本：

> of 0.0810 across the five flow seeds. Radio F1 varies even more. One uncontrolled policy sample would therefore be a weak basis for an event-level conclusion.

后继p569：

> 5.4 Comparison with Trivial Predictors

这样插图位于Table5.3（caption p561及随后table）、事件改善解释p563–565、precision限制p566以及noise说明p567–568之后，完全符合用户要求的“结果解释之后”，无需挪动原表或删原结论。

### 英文引用段

以下暂以用户指定的**Figure5.4**称呼；若采用顺序编号方案，应把新event引用全部改成Figure5.3。

“Figure 5.4 compares the event-detection profiles of the three checkpoints in pickup and radio. Displaying F1 together with precision separates the increased detection of expert transitions from the remaining unmatched predictions, while the error bars show variation across inference flow-noise seeds.”

若只画F1，改为：

“Figure 5.4 compares event F1 across the three checkpoints in pickup and radio, with error bars showing variation across inference flow-noise seeds. The precision values in Table 5.3 and the unmatched-transition counts remain necessary for interpreting the increase in F1.”

如果画precision／recall／F1三项，应将第一版“F1 together with precision”替换成“precision, recall and F1”。不应在图上使用task success／grasp success标签。

### 英文caption

“Figure 5.4: Gripper-event prediction across A20, B20 and C20. Panels show (a) pickup and (b) radio under the frozen audit of 652 matched validation anchors. Points and error bars represent the mean and sample standard deviation over five inference flow-noise seeds for one trained 20,000-step checkpoint per configuration. Event counts are pooled within each context before scoring; complete precision, recall and F1 values are retained in Table 5.3. The checkpoint differences do not isolate single-factor effects or establish successful grasping.”

若图采用bar而非point，改“Points”为“Bars”。不能将errorbar称95%CI或training-seed SD。所谓652是**整份manifest**，不是两个manipulation contexts各652；panel标签最好注明pickup200 anchors／radio200 anchors，caption保留“audit of652”。

### 保留／补充的因果边界

- 现p565已说明A→B同时改变prompt和sampling、每配置只有1个training run，应保留。
- 现p566已说明precision最高仅0.3163、多数prediction unmatched、不能推成功抓取，应保留。
- 现p595–600及§6.8 p690–691已说明B→C同时改变state／action normalisation。为让图周边也清楚，可在新引用段后只加一句：“B20 to C20 changes state and action normalisation together.” 不需要重写整节。
- 不增加statistically significant／causal improvement等措辞。F1／precision／recall的每seed聚合定义与误差棒须由主代理对照冻结records，不能从mean counts重新反推mean F1。

### 现有Table5.3一致性检查（已直接读取v7原生表）

| Context | A20 F1 | B20 F1 | C20 F1 |
|---|---|---|---|
| Pickup | 0.1120 ± 0.0445 | 0.3358 ± 0.0810 | 0.3825 ± 0.0461 |
| Radio | 0.0235 ± 0.0526 | 0.1768 ± 0.1025 | 0.2181 ± 0.1293 |

这些mean与用户附件sanity check相符。此处是**文稿表格一致性检查**，不是新增冻结records审计；实际绘图数值仍由主代理／绘图任务对源数据验证。

### 编号冲突：两种映射供主代理落稿

现有v7阅读顺序为Figure5.1（§5.1）→5.2（§5.2）→5.3 conditioning（§5.5）。用户要求新增event Figure5.4放§5.3，会产生5.1→5.2→5.4→5.3。

| 方案 | §5.3 event图 | §5.5 conditioning图 | 必要同步 |
|---|---|---|---|
| 按正文出现顺序（推荐，待用户选择） | **5.3** | **5.4**（由原5.3改号） | 新event全部引用用5.3；现p585末句及p587caption改5.4；旧conditioning TC图目录域改5.4；图文件／映射更新 |
| 保留用户明确指定图号 | **5.4** | **5.3**保持 | 新event用5.4；保留conditioning；报告图号首次出现顺序例外，不能宣称按出现顺序连续 |

两个方案都不把section“5.3 Manipulation Event Prediction”、section“5.4 Comparison with Trivial Predictors”、Table5.3、Table5.4改号。不得全局把字符串5.3替换成5.4。

v7现p585末句精确文本：

> Figure 5.3 visualises the perturbation/noise ratios and their anchor-level intervals.

v7现p587caption开头：

> Figure 5.3: Validation observation-conditioning ratios.

Word中图目录使用`TC ... \\f F`域；存在原conditioning TC项`5.3 Validation observation-conditioning ratios.`。新增图除了visible caption，还应增加相应TC条目，并刷新LoF缓存／页码。现有section／table的TC采用其他标记，不能混改。

### RQ2回引

§6.3现p645完整文本：

> between single trained checkpoints evaluated under matched inference noise. They do not measure training-seed reliability. They also do not establish successful grasping, force control, or object transport.

可末尾加一句（按最终event编号替换）：

“Figure 5.4 summarises this event-prediction comparison while retaining its inference-variation scale.”

无需改§7.1现p715、§7.2 p722–724，它们已经保留“事件提升、arm近no-motion、base弱于persistence”的混合结论。

## 4. 现有文本只需修正的两处小歧义

### 4.1 Figure5.1总体指代

现p541先报告all-source45条的86.18%／90.06%，末句是：

> Figure 5.1 summarises these geometric coverage ratios by task segment.

但图与Table5.1实际为validation pickup populations。只替换末句：

“Figure 5.1 gives the segment-level geometric coverage for validation pickup anchors.”

现p544caption中的“early walk/acquire and soda-can pickup segments”建议改为“walk/acquire and soda-can pickup segments”。原图不是early／middle／late分箱图，去掉early能避免与新时间统计混淆。保留target-centre **geometric frustum**限定，不改成visibility／detection。

### 4.2 3.955%的分母

现p303、p340已明确是valid B/C **training** anchors；以下位置同一比例省略training，新增all45统计图后宜补齐：

| v7位置 | 原关键短语 | 最小替换 |
|---|---|---|
| p534 | “In the B/C export, decision frames account for 3.955% of valid anchors.” | “In the B/C training export, decision frames account for 3.955% of valid anchors.” |
| p627–628跨段句 | “In the B/C export, decision frames make up only 3.955% of valid anchors.” | 仅把“B/C export”改“B/C training export” |
| p720 | “3.955% of exported B/C anchors” | “3.955% of valid B/C training anchors” |

保留20%为**configured target**；不把它称成observed draws，亦不修改原比例。

## 5. Chapter3–7证据边界审阅结果

以下均为v7已有、应保留的明确限定。没有发现需要因两张新图而推翻主结论的问题。

| 限定 | 当前v7证据位置 | 审阅结果／v8要求 |
|---|---|---|
| 45 retained，37／4／4 source split | p292、p714及Table3.2 | 保留；新增dataset图若all45含test，须标descriptive，不把它当test policy performance |
| project-constructed scripted＋upstream replay provenance | §3.2.2 p269–275 | 保留；视频帧不是learnt rollout成功 |
| policy仅ExecuteBehaviorTask | p282／284、p521–522、p702 | 保留；FSM／新数据图不暗示learned navigation或端到端HRC |
| L1–4 controlled audit，L5 historical支持 | p244／246、p252–253 | 保留；图3.1重绘不能把5层都画成同强度final evidence |
| 652 matched validation anchors | p372、p493–499、p546 | 保留；新event两个panel各是相应context subset |
| 5 inference seeds，1 trained checkpoint/config | p387–389、p454、p611–612、p685–687 | 保留；误差棒不转写成training uncertainty或population CI |
| conditioning val108／coverage216，prompt-stable条件不同 | p393–397、p504–509及Table4.2 | 保留；renumber不改变conditioning population |
| image/noise是local response，非“vision unused” | p591–593、p653、p694–695 | 保留；conditioning图只是改号或排版，不新增必要性结论 |
| A→B bundle，B→C state/action normalisation bundle | p338、p472–476、p487–488、p565、p595–600、p688–691 | 保留；event排序不可写成单因素因果效果 |
| near-zero pickup arm skill；base低于persistence | p551–558、p576–577、p639–647 | 保留；event图不应把benchmark描述成控制成功 |
| frustum是几何、不是pixel visibility | p427–429、p517–518、p634–635、p697–699 | 保留；仅修正上文总体指代 |
| historical rollout未与final audit匹配 | p520–522、p602–606、p700–701、p726 | 保留；不把历史zero-contact与final eventF1建立定量关系 |
| case-study而非广泛generalisation | p684、p702、p736 | 保留；新图说明结构，不夸大规模／多样性 |

§7结论与future work现已保留限制，无需为了“图更多”增加新贡献或实验结论。主代理应在最终v8再次做cross-reference／caption／table consistency及实际PDF页面检查；本报告不替代渲染QA。

## 6. 声明与交付同步

- 现p19区分3.1／3.2／3.4 schematic、3.3真实帧、3.5记录统计；v8增加3.6的数据统计类别，并让最终event／conditioning图号与正文一致。
- 最新EEE适用声明模板与2026-07-06生效日仍**Pending official template verification**，详见`V8_NTU_TEMPLATE_VERIFICATION.md`。已获取的public25Nov2025模板只能作为真实参考，不能自动替换为学校最新强制措辞。
- TC图目录、visible captions、正文引用、figure-to-thesis mapping、traceability、AI disclosure按同一最终映射更新；18张原生表及表号保留。
- 本代理未修改v7、图像、renderer、git或主代理文件。

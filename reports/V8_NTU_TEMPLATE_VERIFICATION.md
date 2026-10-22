# v8 NTU official declaration-template verification

**Status: Pending official template verification.**

已取得一个NTU官网公开下载的真实声明DOCX；其明确版本为**25 Nov 2025**，不是已确认的2026-07-06后新版。EEE／MSc in Signal Processing and Machine Learning是否采用该模板、以及用户所述2026-07-06生效要求，均未由本次公开资料核实。不得把本报告视为学校提交合规认证，也不得据此删除真实AI使用记录、填写未知工具版本或代签。

## 官方来源与版本

| 来源 | 精确URL | 可核对日期与范围 | 本次结果 |
|---|---|---|---|
| NTU Research：Theses / Dissertations | https://www.ntu.edu.sg/research/theses-dissertations | 页面把AI披露标为Nov 2025新增；声明下载标updated 25 Nov 2025。该页说明学校可能采用学科适配版本。 | 已打开，HTTP200，已保存HTML |
| 上述页的实际官方模板链接 | https://entuedu.sharepoint.com/:w:/r/sites/Student/research/rieo/Shared%20Documents/Research%20Integrity/Thesis%20Declaration%20Statements_25Nov2025.docx | 文件名25Nov2025；网页明确标Download (Intranet) | web工具未成功取得；本机urllib因TLS hostname certificate mismatch失败。未关闭证书校验，未尝试绕过登录 |
| NTU MAE MSc：Dissertation Information | https://www.ntu.edu.sg/mae/admissions/current-students/graduate-studies/dissertation-information | MSc dissertation公开页面，含三项声明及模板链接。**属于MAE，不是EEE。** | 已打开，已保存HTML；用于追溯公开模板入口 |
| MAE页面链接的NTU官方DOCX | https://www.ntu.edu.sg/media/docs/librariesprovider122/curriculum/thesis-declaration-statements_25nov2025.docx?sfvrsn=1d36a3ee_1 | 文件名25nov2025；HTTP Last-Modified: Tue, 02 Jun 2026 10:48:45 GMT。传输时间戳不等于政策生效日期 | HTTP200，实际DOCX已下载并读取，37,338 bytes |
| NTU Policy on the Use of Generative Artificial Intelligence in Research | https://www.ntu.edu.sg/research/resources/use-of-gai-in-research | Date issued: 04 Jul 2023；Last reviewed: 07 Nov 2025 | 已打开，HTTP200，已保存HTML；用于核对一般披露要求，不代替声明模板 |

2026-07-06后的新版、生效通知和EEE专用版本：**未查实**。这不证明其不存在。对EEE、MSc、originality／AI disclosure、6/06 July 2026等组合进行官方域名检索后，未取得能证明该具体日期要求的文件。搜索返回的旧EEE招生册、MAE旧指南、一般Library coursework disclosure guidance，均未被当成EEE新版强制模板。

## 真实公开模板中可核对的结构

公开DOCX首部分标题是Statement of Originality，之后为“Disclosure on AI Use”。作者按真实情况保留A，或B及／或C：A指未使用；B指准备文稿时使用，要求工具名称／版本、用途、作者复核与责任以及遵守NTU政策；C指AI是研究设计／方法的必要组成，需说明Methods位置。以上为结构说明，不是供直接复制的学校指定替代措辞。[官方DOCX](https://www.ntu.edu.sg/media/docs/librariesprovider122/curriculum/thesis-declaration-statements_25nov2025.docx?sfvrsn=1d36a3ee_1)

NTU Research公开页要求三项声明，并提醒院系版本可能不同；无已发表材料时仍需保留相应Authorship Attribution选项。签名应由本人／导师处理；公开页建议使用NTU水印签名区域。[官方提交声明说明](https://www.ntu.edu.sg/research/theses-dissertations)

一般研究AI政策要求超出基础拼写／语法检查的使用得到披露，包含工具名称、版本、目的及使用方式；thesis/dissertation披露放在Statement of Originality。AI不能承担作者责任；图像不得把生成内容冒充实验记录。[官方AI研究政策](https://www.ntu.edu.sg/research/resources/use-of-gai-in-research)

## 与当前v7的实际差异

审阅文件：`/root/dissertation_figures_delivery/NTU_Dissertation_Reviewed_v7_figures_editable.docx`。

SHA256：`e24f23f635b43b97297273ae1f85b66d97b2b09a39e971dee9725b76694afbb9`。

| 项目 | v7现状（零起始paragraph index） | 与已取得公开模板的差异／处理建议 |
|---|---|---|
| Originality正文 | p13–14已有原创、无剽窃、未用于其他学位的声明 | 主旨与公开模板相同；不要把小幅thesis/dissertation用词差异当已核实的EEE要求 |
| 披露标题 | p12组合标题；p15为Disclosure of AI assistance | 公开模板使用独立AI披露小节及A/B/C结构；不能据此认定当前EEE格式不合规 |
| 工具与用途 | p16–18写OpenAI ChatGPT/Codex及撰稿、文献核对、代码检查、制图等用途 | 已真实披露较广泛协助，应保留；工具**具体版本尚未写出**，不能推测填入 |
| 模板结构／承诺 | p16–22为自述披露；写作者负责、AI不是作者 | 没有公开模板的A/B/C结构和明确遵守NTU政策表述。若学校确认采用该模板，再按实际情况套用；当前只记录差异 |
| Figures来源 | p19–21区分AI-assisted schematic、视频帧、记录统计与实验重绘 | v8应增加3.6及event图的来源类别说明，但不能把真实帧称AI生成。具体按主代理最终图号更新 |
| C选项 | 当前无模板C选项 | 不自动替作者决定。π0.5作为研究对象与写作AI协助的披露关系，应由作者／导师依据适用模板判断；不能只因有robot VLA就机械勾选C |
| Supervisor／Authorship | p27–33已保留相应声明和未签名占位 | 保留；签字、日期及适用模板仍须有权本人处理，不能由本代理代填确认 |

## 本地公开证据

目录：`/root/dissertation_local_audit/v8/official_templates/`

- `thesis-declaration-statements_25nov2025_public_MAE.docx`：真实NTU公开模板；SHA256 `af1e5f06041cef7e1a839fef92e3f1815ec6d181753e394ba09afdbce586c8fc`。
- `ntu_theses_dissertations.html`：NTU Research公开页快照。
- `ntu_genai_research_policy.html`：NTU研究AI政策快照。
- `ntu_mae_msc_dissertation_information.html`：公开模板的MSc页面入口快照。
- `retrieval_manifest.json`、`public_template_retrieval.json`：精确URL、HTTP信息、实际抓取UTC时间、大小与SHA256。
- `intranet_template_access.json`：真实失败原因；未保存登录页冒充模板。

抓取日期以manifest记录的主机实际UTC时间为准；HTML的HTTP Last-Modified不被用来宣称模板有新版本。报告不复制整份官方声明；原始下载文件供获授权人员逐字核对。

## 主代理可立即执行的最小动作

保留现有真实披露及签名占位；同步v8真实图号／来源。提交检查报告继续标**Pending official template verification**。待作者或学校提供EEE适用的最新版后，逐字核对并再处理选项、版本、排版和签名；不要把2025公开模板直接标成July2026后的已核准版本。

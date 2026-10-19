# 依赖、版本与授权边界审计

2026-10-19（Asia/Singapore）静态读取本地git和安装元数据；没有访问远端、导入Isaac/OmniGibson或执行安装。

## 1. 本地源码快照

| 组件 | 本地证据 | 值 |
|---|---|---|
| hrc_pipeline | `git rev-parse HEAD` | `94e89df6c8db41d23aa4d1a3f3be2907663ba478` |
| hrc_pipeline工作区 | `git status --short` | 审计前已有` M docs/notion.md`；本次未修改 |
| 嵌套OmniGibson monorepo | `git rev-parse HEAD` | `88454bd04f75dc57c00ab1f1a00bcde1ff505950` |
| 嵌套OmniGibson工作区 | `git status --short` | `?? OmniGibson/appdata`；无被跟踪修改 |
| OmniGibson包 | `OmniGibson/OmniGibson/setup.py:15–16` | `omnigibson 3.7.2` |
| BDDL包 | `OmniGibson/bddl3/setup.py:9–10` | `bddl 3.7.0` |
| converter版本 | `scripts/habitat_humanoid_bridge.py:98–105` | bridge15、layout6、Y-up→Z-up修正 |

根仓库没有通过git submodule登记嵌套上游，因此仅发布根commit不足以固定第三方版本；必须另列commit。

## 2. 当前behavior虚拟环境的文本元数据

读取路径：`/root/autodl-tmp/behavior-venv/pyvenv.cfg`和`lib/python3.10/site-packages/*.dist-info/METADATA`；这些数值证明当前安装，不证明所有历史数据都由同一环境生成。

| 包 | 实际安装 |
|---|---|
| Python | 3.10.12 |
| torch | 2.6.0+cu124 |
| torchvision / torchaudio | 0.21.0+cu124 / 2.6.0+cu124 |
| isaacsim | 4.5.0.0 |
| omnigibson / bddl | 3.7.2 / 3.7.0 |
| numpy / scipy | 1.26.4 / 1.15.3 |
| h5py | 3.16.0 |
| opencv-python | 4.11.0.86 |
| imageio-ffmpeg | 0.6.0 |
| PyYAML | 6.0.3 |
| telemoma | 0.3.0 |

OmniGibson和BDDL的`direct_url.json`分别指向原项目`OmniGibson/OmniGibson`、`OmniGibson/bddl3`，均editable=true。此环境会受原源码变化影响。没有执行pip freeze，未读取任何认证或私有索引配置。

## 3. 文档漂移与复现边界

`docs/behavior_venv_requirements.txt`是标注2026-05-25的快照，Python3.10/Isaac4.5/torch2.6与当前主要栈一致，但h5py写3.15.1，当前已3.16.0。README另有Python3.11/Isaac5.1/torch2.7/OG3.8的安装叙述，与当前本地setup.py和环境不一致。论文应以数据生成时的日志/manifest固定环境，不把README安装建议当历史事实。

`patches/data_wrapper.patch:9–19`把两处h5py输入句柄由r改为r+；当前上游`OmniGibson/omnigibson/envs/data_wrapper.py:758、897`仍是r，即该patch未应用于当前快照。它为写源文件补注服务，不属于必须无条件应用的只读采集前置条件；注意源数据写入副作用。

`OmniGibson/OmniGibson/setup.py:77–84`另列可选cuRobo commit `cbaf7d32436160956dad190a9465360fad6aba73`及LeRobot commit `577cd10974b84bea1f06b6472eb9e5e74e07f77a`。这是上游extras声明，不证明本项目安装或使用了这些版本；不能据此填入训练实验版本。Isaac依赖未在此setup install_requires中固定，须单独记录。

## 4. 第三方软件/数据权利分界

| 材料 | 本地能确认的权利证据 | 发布处理 |
|---|---|---|
| 本项目原创候选源码和文档 | 根LICENSE及文件头All Rights Reserved，署名Mike Xu | 用户已明确确认权利人身份并授权本论文及指定仓库；可在安全审查后上传原创代码，保留原LICENSE、不重许可 |
| OmniGibson软件 | `OmniGibson/OmniGibson/LICENSE`为MIT、Stanford Vision and Learning Group 2023 | 不整仓拷贝；依赖安装。适配段/patch保留MIT声明 |
| BDDL软件 | `OmniGibson/bddl3/LICENSE`为MIT、Stanford Vision and Learning Lab 2021 | 依赖引用，不捆绑完整库 |
| JOYLO软件 | `OmniGibson/joylo/LICENSE`为MIT、Philipp Wu 2023 | 核心未证明直接用到JOYLO；单独列为上游可选包 |
| Isaac Sim/Omniverse及其角色资产 | 上游专有运行时和素材；本次没有取得允许重新分发的许可证证据 | 仅写安装依赖与官网入口，不捆绑wheel/资产；默认角色URL见utils:40–50 |
| BEHAVIOR资产、场景和2025 Challenge原始示教 | 软件MIT不能推及数据；本次本地未确认逐项可再分发条款 | 不随原创代码发布；只留数据获取指引与需用户自行接受的来源条款 |
| Habitat male_0网格/纹理/动作及USD派生文件 | source_asset.json:6–8指向Habitat humanoid与SMPL-X动作；未找到覆盖这些派生物的可再分发授权 | assets全排除；公开转换器候选源码与占位说明 |
| SMPL-X相关数据 | 文件名证明依赖来源线索，不足以证明具体权利归属、授权版本或再分发范围 | 不分发pkl/npz/派生动作资产，保留许可待确认项 |

这里的“待确认”指本地证据不足，不是断言第三方禁止一切使用。此次审计没有取得第三方许可证权利人新增的素材再分发授权；用户本人原创代码的指定用途授权已另确认。

## 5. 原创性准确表述

`hrc_utils.py:481–569`与上游`data_wrapper.py:996–1059`的禁用控制、transition重建、load_state和对象keep_still结构明显对应，应标为“基于OmniGibson回放接口的改写与扩展”。`patches/data_wrapper.patch`含上游上下文，按第三方衍生适配保留其MIT条款。人物转换代码为本项目原创候选，但资产内容不能因格式转换变成原创；脚本Copyright头也不足以独立证明无任何第三方片段，故不作百分比原创判定。

## 6. 追加：相邻训练仓库许可核对

主代理要求追加核对，范围仅根LICENSE与git commit，未审查其全部源码。

| 路径 | 本地commit | 根许可与风险 |
|---|---|---|
| `/root/autodl-tmp/hrc_pipeline_traintest` | `6980ad469c775a4da508dbdccc8465ab9b1df1e5` | 与hrc_pipeline相同的Mike Xu 2026 All Rights Reserved，限制再分发及论文使用；现用户已确认权利人身份并授权本论文及指定仓库；保留原许可，不推及第三方材料 |
| `/root/autodl-tmp/openpi` | `c23745b5ad24e98f66967ea795a07b2588ed6c79` | 根LICENSE为Apache2.0；另有LICENSE_GEMMA.txt（Gemma Terms of Use，2024-02-21），因此模型/派生模型条款不能用Apache源码许可替代 |
| `/root/autodl-tmp/openpi_hrc` | `cfd097d6ca0f957b68d05616a3b71b86dabdf2cf` | 根LICENSE为Apache2.0，同样附Gemma条款；HRC本地改动的具体权属与新增文件逐项许可未在本补充中完成证明 |

整棵openpi/openpi_hrc不是本项目从零原创，不应以原创仓库名义重发完整上游。用户授权范围内优先发布改动/适配代码及依赖commit，保留Apache/Gemma/NOTICE等实际适用声明；不包含checkpoint、Gemma权重、缓存或训练数据。

授权状态更新：原先的原创代码权属待确认项已由用户明确确认解除；本论文及指定仓库获授权。原LICENSE不更换，不授予第三方素材再分发权。

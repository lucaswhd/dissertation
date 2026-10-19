# 本地源码暂存布局与使用边界

**授权状态：用户已明确确认本人为权利人，授权原创代码用于本论文及指定仓库。代码、统计与报告可在最终安全审查后上传；保留原LICENSE，不重许可。第三方资产、人物动作、权重和原始数据仍排除，storyboard及Word渲染材料仅本地保留。**

| root_alias | 快照路径 | 角色与范围 |
|---|---|---|
| hrc_pipeline | `source_snapshots/hrc_pipeline/` | 历史采集主树：scripts、configs、patches及必要shell。不要当作最终研究实现。 |
| hrc_pipeline_traintest | `source_snapshots/hrc_pipeline_traintest/` | 当前研究分支：保留collection/preprocess/training/evaluation/debug/legacy真实scripts层级与根兼容脚本、configs、patches；含在线物理导航和policy injection。 |
| openpi_hrc | `source_snapshots/openpi_hrc/` | 相对本地openpi基线不同或新增的小型scripts/src/packages代码、必要pyproject及LICENSE/Gemma条款；不是完整上游或独立可安装软件包。 |

保留每root原始LICENSE和已有NOTICE/COPYING（若有），没有用OmniGibson MIT覆盖全部项目。没有复制assets、第三方整仓、原始数据、权重、checkpoint、运行日志、缓存、内部交接/Notion文档、编辑器配置或.env。

## OpenPI恢复方法

`source_snapshots/openpi_hrc/patches/against_local_openpi_baseline.patch`以本机`openpi`基线的当前文件内容为参照，包含选中源码的新增/修改；不是git历史导出。基线此前读取到的commit为`c23745b5ad24e98f66967ea795a07b2588ed6c79`，本轮没有执行git；工作区内容以`reports/OPENPI_BASELINE_FILES.csv`中每个baseline_sha256为准。新增文件baseline_exists=False。

先自行取得匹配的完整上游工作副本并核验baseline SHA256，再在其根目录做`patch -p1 --dry-run < against_local_openpi_baseline.patch`；检查成功后才在那个独立副本中应用。也可按相对路径叠加快照，但不能仅把snapshot当作完整openpi安装。uv.lock未复制，依赖版本应结合原上游、pyproject变更及实际训练环境锁定；不承诺仅凭该patch复现全部实验。模型权重受独立条款，不随代码提供。

## 清单工具与重跑

运行`python tools/release_inventory.py`刷新元数据清单；运行`python tools/release_inventory.py --snapshot`同时刷新选中源码和OpenPI差异patch。脚本只使用Python标准库，无网络、git、仿真或训练操作，源root只读，不跟随symlink、不删除文件。默认SHA256上限为1MiB且仅小源码/配置/许可，数据与大文件不hash。对历史私人工作站路径前缀只在暂存副本脱敏；FILE_CLASSIFICATION同时记录源sha256、snapshot_sha256和sanitization_policy。

`reports/FILE_CLASSIFICATION.csv`以root_alias/relative_path记录逐文件元数据；私有绝对路径完整版独立放在本地审计目录，不应入发布包。缓存/环境/VCS等剪枝子树只记directory_skipped，不枚举内部。被secret正则匹配的候选只报告文件路径并排除；零匹配不等于证明无秘密。最终外发仍需主代理复检全部staging，并限定于用户指定仓库与授权范围。

工具不会删除已有暂存，因此若源文件后续删除或被排除，旧快照可能仍存在。每次外发应从干净发布目录按当次manifest重建，不能直接打包长期复用的staging。

# 源码暂存验证

验证日期：2026-10-19（Asia/Singapore；host UTC日期为2026-10-18）。原实验/commit时间未改。

- 清单工具多次复跑均24,045条记录。
- 复制223个源码/配置/许可文件：旧core32、研究分支107、OpenPI差异84；其中217文件与原文件字节一致，6个历史YAML模板仅在暂存副本脱敏私人路径前缀，共18处。全部223快照均与生成器的预期转换逐字节一致；manifest同时保留source sha256与snapshot_sha256。
- OpenPI恢复patch对本地基线执行`patch --dry-run -p1`，exit=0，检查82个新增/修改文件；基线已有文件SHA256仍与manifest相符。
- 生成器正确处理基线缺尾换行文件；没有修补或修改原项目。
- FILE_CLASSIFICATION.csv的relative_path全部为相对路径；私有绝对完整清单在staging之外，文件权限0600。
- 待复制文本的secret正则匹配0项；这不是无秘密的完整证明，仍需主代理最终复检。
- 权利人已明确授权本论文和指定仓库使用原创代码，原先权属HOLD已解除。保留原LICENSE，不重许可；本代理未执行push。第三方资产、human motion、权重与raw datasets继续排除，storyboard与Word渲染材料仅本地保留。

脱敏文件（只列alias/relative path）：

- hrc_pipeline/configs/hrc_closed_loop.yaml（3处）
- hrc_pipeline/configs/hrc_task.yaml（3处）
- hrc_pipeline/configs/hrc_v01.yaml（3处）
- hrc_pipeline_traintest/configs/hrc_closed_loop.yaml（3处）
- hrc_pipeline_traintest/configs/hrc_task.yaml（3处）
- hrc_pipeline_traintest/configs/hrc_v01.yaml（3处）

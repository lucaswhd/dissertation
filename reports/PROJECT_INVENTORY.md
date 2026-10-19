# 研究文件清单与分类

生成时间：2026-10-19T09:43:36.366278+08:00

本清单仅覆盖三个明确研究root；发布候选版只含root_alias/relative_path，私有绝对路径清单独立保管。
权利人已明确授权本论文及指定仓库使用原创代码；保留原LICENSE、不重许可。第三方资产/权重/原始数据仍排除，代码待最终安全审查后可上传。分类A/B本身不证明逐段原创。未执行网络、git、仿真、训练。

## 分类定义

| 类别 | 含义 |
|---|---|
| A | 项目研究代码/配置候选；分类不证明权属 |
| B | OpenPI相对本地基线的改动/新增适配代码 |
| C | 第三方软件或未改动OpenPI上游代码 |
| D | 输入数据/模型资产/转换派生素材；不复制 |
| E | 模型权重/checkpoint；不复制不hash |
| F | 运行输出/日志/结果/归档；不复制 |
| G | 环境、缓存、VCS、编辑器等排除目录/文件 |
| H | 敏感或私有材料/内部文档/未分类材料；不复制 |

## 元数据统计

| root_alias | 类别 | 记录数 | 普通文件字节 |
|---|---:|---:|---:|
| hrc_pipeline | A | 35 | 543688 |
| hrc_pipeline | B | 0 | 0 |
| hrc_pipeline | C | 19490 | 939608091 |
| hrc_pipeline | D | 11 | 10405395 |
| hrc_pipeline | E | 0 | 0 |
| hrc_pipeline | F | 9 | 3799829 |
| hrc_pipeline | G | 27 | 0 |
| hrc_pipeline | H | 12 | 302602 |
| hrc_pipeline_traintest | A | 109 | 1356039 |
| hrc_pipeline_traintest | B | 0 | 0 |
| hrc_pipeline_traintest | C | 987 | 106426475 |
| hrc_pipeline_traintest | D | 11 | 10405395 |
| hrc_pipeline_traintest | E | 0 | 0 |
| hrc_pipeline_traintest | F | 2750 | 7479993762 |
| hrc_pipeline_traintest | G | 8 | 0 |
| hrc_pipeline_traintest | H | 32 | 883337 |
| openpi_hrc | A | 0 | 0 |
| openpi_hrc | B | 82 | 712457 |
| openpi_hrc | C | 134 | 1636010 |
| openpi_hrc | D | 8 | 26589 |
| openpi_hrc | E | 102 | 20448591008 |
| openpi_hrc | F | 217 | 215332796 |
| openpi_hrc | G | 21 | 0 |
| openpi_hrc | H | 0 | 0 |

## 覆盖、hash和复制策略

- 非缓存代码与大资产逐文件记录；symlink仅记自身且不跟随。排除子树以directory_skipped显式登记，其内部不遍历。字节统计不含这些子树，不能称整机磁盘总量。
- SHA256只读取≤1MiB的小型源码/配置/许可；数据、权重、大文件只记元数据，不hash数十GB材料。敏感文件名不读取内容。
- source_snapshots仅含小型研究scripts/configs/patches、必要shell与许可证；不复制assets、模型、数据、日志、内部交接文档、缓存、备份文件或编辑器配置。
- OpenPI快照只包含相对本地openpi基线不同/新增的小源码与许可证，附恢复patch及逐文件baseline校验值；不是可独立安装的上游整仓。
- secret扫描只输出被排除的root_alias/relative_path，不输出匹配内容；正则不能证明绝对无秘密，发布前仍需人工/最终复检。
- 对匹配历史私人工作站路径前缀的小型候选文本，仅在暂存副本脱敏；snapshot_sha256记录脱敏后内容，sha256仍指源文件。
- 原项目不写入、不删除。重复运行会刷新本工具生成的清单和已选快照；不会删除旧暂存，复用目录时应复检是否存在陈旧文件。

## 本次快照

--snapshot：True；复制文件数：{'hrc_pipeline': 32, 'hrc_pipeline_traintest': 107, 'openpi_hrc': 84}。
清单记录数：24045；secret-pattern排除：0。


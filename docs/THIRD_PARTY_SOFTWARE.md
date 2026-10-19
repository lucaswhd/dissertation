# 第三方软件、数据与资产

本地暂存仅整理项目原创候选代码范围及必要的上游适配说明。用户已明确确认本人为权利人，并授权用于本论文及指定仓库；原创代码权属阻碍已解除，保留原LICENSE、不重许可，安全审查通过后可准备上传。软件依赖、数据集与人物模型具有各自的权利边界；公开脚本不表示授予第三方素材再分发权。

| 名称 | 用途与版本证据 | 许可/分发处理 |
|---|---|---|
| OmniGibson / BEHAVIOR软件 | 本地monorepo commit `88454bd04f75dc57c00ab1f1a00bcde1ff505950`；omnigibson3.7.2 | MIT，Stanford Vision and Learning Group 2023。依赖安装，不附上游全仓。`hrc_utils.py`回放逻辑和data_wrapper.patch含上游适配，保留以下声明。 |
| BDDL | 同一monorepo的bddl3，版本3.7.0 | MIT，Stanford Vision and Learning Lab 2021。不随源码副本捆绑。 |
| JOYLO | 上游monorepo可选组件 | MIT，Philipp Wu 2023；不作为本项目原创，不证明核心直接使用。 |
| NVIDIA Isaac Sim / Omniverse | 当前behavior环境4.5.0.0；渲染、仿真、USD工具 | 独立NVIDIA条款。用户自行获取运行时并接受条款，不分发wheel、缓存或NVIDIA角色资产。 |
| Habitat humanoids / SMPL-X相关素材 | 源元数据指向male_0 GLB、URDF、motion_data_smplx.pkl | 本地未确认覆盖再分发的具体授权；模型、纹理、骨骼、运动及其USD/NPZ派生物均不包含在发布中。仅提供转换器源码候选。 |
| BEHAVIOR场景、物体、2025 Challenge原始示教 | 数据构建输入 | 不由软件MIT自动覆盖；不随本代码发布。按来源流程自行申请/下载，遵守对应数据条款。 |
| PyTorch、NumPy、SciPy、h5py、PyYAML、OpenCV、imageio等 | Python依赖；安装版本见reports/DEPENDENCY_AUDIT.md | 由用户环境安装，不捆绑库源码或二进制；各包许可随其自身分发物。 |

项目根LICENSE和源码头原为All Rights Reserved。用户现已确认权利人身份；本次授权仅覆盖其有权授权的原创内容及指定用途；本说明不重新许可第三方软件与数据，也不把所有带本项目版权头的代码一概认定为从零原创。

转换产物不等同于转换器版权：原`assets/`目录虽然在项目git中被跟踪，仍含第三方内容与派生素材。该目录应整体排除。原source_asset.json包含历史机器路径，不宜公开原样复制。

## OmniGibson MIT notice

以下条款用于保留OmniGibson适配段及补丁的来源；不能据此把本项目全部重标为MIT。

```text
MIT License

Copyright (c) 2023 Stanford Vision and Learning Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

BDDL原始版权声明为“Copyright (c) 2021 Stanford Vision and Learning Lab”；JOYLO为“Copyright (c) 2023 Philipp Wu”，其原始LICENSE同为MIT。若未来复制其任何实质代码，应随副本保留各自完整声明。

## 训练代码补充

`hrc_pipeline_traintest`根LICENSE同为Mike Xu 2026 All Rights Reserved；权利人现已授权本论文及指定仓库使用原创代码，保留原许可并经安全审查后可上传。`openpi`与`openpi_hrc`根LICENSE为Apache2.0，同时存在独立Gemma Terms of Use（LICENSE_GEMMA.txt）；源码许可与模型权重/衍生模型条款应分开。其本地commit分别为`c23745b5ad24e98f66967ea795a07b2588ed6c79`、`cfd097d6ca0f957b68d05616a3b71b86dabdf2cf`。这些上游库不能作为本项目全部原创重新署名，模型权重不随发布。

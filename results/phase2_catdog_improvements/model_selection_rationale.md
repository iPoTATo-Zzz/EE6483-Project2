# 模型选择依据与比较边界

## 为什么选择 ResNet-18

ResNet-18 是课程允许的 ResNet 路线中的较浅模型，能复用成熟的 torchvision 实现及 ImageNet 预训练权重。残差连接是其结构特征。本项目选择它作为统一基线，是为了在本机算力与时间预算内建立可复现的迁移学习流程，并为数据增强、学习率和后续任务提供参照；并不是证明它是猫狗识别的最优模型。既有 D1/D2 已完成，继续沿用它可以避免反复更换基线。

## 为什么增加 MobileNetV3-Small

该模型是不同于 ResNet 的轻量网络路线。MobileNetV3 论文将 Small 面向较低资源场景；torchvision 提供正式模型和 ImageNet 权重。因此它适合作为受限时间、显存条件下的补充比较，检验更小的模型是否足以完成当前二分类任务，为报告中“模型选择如何影响效果与成本”提供实际证据。轻量结构不保证在本机GPU上更快，也不保证准确率更高。

## 比较如何成立及如何避免过度解释

两个模型采用同一猫狗训练／验证清单、标签、224×224确定性预处理、冻结特征策略、Adam、batch size32、seed42和10轮预算；分别从各自ImageNet权重开始。ResNet只训练新建512→2线性层；MobileNet保留已预训练分类器前部，仅训练新建1024→2线性层，已有Dropout保持评估状态。因此对比的是两套预训练特征提取方案，架构、特征维度和预训练过程同时不同，不能把准确率差异全部归因于网络结构。相同基础参数也不代表每个架构各自达到最优。

补充试验使用缓存冻结特征；提取与分类层训练耗时分别记录，不直接与D1/D2未缓存执行耗时比较。未做专门的推理基准时，不声称某模型本机推理更快。需要报告参数规模时，应以实际改为二分类后的模型计数，而不是直接引用原始1000类模型的总参数数目。单种子结果只用于初步比较，不声称稳定显著优势。

## 来源

- He et al. (2016), Deep Residual Learning for Image Recognition：https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
- Howard et al. (2019), Searching for MobileNetV3：https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf
- torchvision ResNet-18：https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
- torchvision MobileNetV3-Small：https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html

这些引用仅支撑模型选择理由，不代表文献综述已完成。

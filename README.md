# EE6483-Project2

NTU EE6483 Mini Project (Option 2)：猫狗二分类、CIFAR-10 十分类与类别不均衡实验。

## 实施路线

1. phase1_catdog_baseline：ImageNet 预训练 ResNet-18，冻结骨干，仅训练二分类层。
2. phase2_catdog_improvements：比较数据增强与局部微调，按验证集结果选择模型。
3. phase3_cifar10：适配 32×32 输入的 ResNet-18，从头训练十分类模型，比较基线与增强。
4. phase4_imbalance：固定不均衡训练样本，比较普通训练、类别加权损失与少数类过采样。

## 目录结构

```text
EE6483-Project2/
├── README.md
├── .gitignore
├── Requirement/                  # 保留现有课程材料
├── ReferenceProject/             # 保留现有参考材料
├── datasets/                     # 本地数据，不提交 Git
│   ├── train/{cat,dog}/
│   ├── val/{cat,dog}/
│   └── test/
├── configs/                      # 各阶段实验配置
│   ├── phase1_catdog_baseline/
│   ├── phase2_catdog_improvements/
│   ├── phase3_cifar10/
│   └── phase4_imbalance/
├── src/
│   ├── common/                   # 共享数据读取、训练、指标和辅助工具
│   ├── models/                   # 模型定义，不存放权重
│   ├── phase1_catdog_baseline/
│   ├── phase2_catdog_improvements/
│   ├── phase3_cifar10/
│   └── phase4_imbalance/
├── runs/                         # 本地日志与权重，不提交 Git
├── results/                      # 精选指标、曲线与案例图
│   ├── phase1_catdog_baseline/
│   ├── phase2_catdog_improvements/
│   ├── phase3_cifar10/
│   └── phase4_imbalance/
└── submission/                   # 提交样例与最终 submission.csv
```

空目录中的 `.gitkeep` 用于保留 Git 目录结构。`runs/` 不上传 GitHub，在其他电脑首次运行时建立。报告在仓库外另行存储。

## 当前状态

阶段 0 已完成。阶段 1 已实现数据检查、冻结骨干训练、模型重载评估与预测入口。完整数据检查发现 1 组训练／验证完全重复图片及 4 组训练内部重复。已确认仅从训练清单排除 cat.2339.jpg，保留全部验证图片与训练内部重复，原始图片不删除。正式训练规模由短训练耗时确定。

猫狗数据已检查：训练集每类 10,000 张，验证集每类 2,500 张，测试集 500 张；全部图片可读取，测试 ID 覆盖 1–500。完整文件哈希检查只能识别内容完全相同的文件，不能排除所有视觉近似重复。CIFAR-10 数据准备将在阶段 3 完成。

## 本机已验证环境

- Windows，Python 3.13.4。
- NVIDIA GeForce GTX 1660 Ti，6 GB 显存。
- PyTorch 2.10.0+cu126，torchvision 0.25.0+cu126。
- 独立环境位于仓库外：`D:\EE6483\ee6483-venv`。
- 已验证 GPU 卷积前向计算、反向传播和参数更新。

在 PowerShell 中检查项目解释器：

```powershell
& 'D:\EE6483\ee6483-venv\Scripts\python.exe' --version
```

后续脚本应使用该解释器，避免误用全局 CPU 版 PyTorch。上述路径是本机位置，其他电脑需要自行建立独立环境。`requirements.txt` 记录已安装依赖。

在其他 Windows 电脑建立独立 Python 3.13 环境后，先从官方 CUDA 12.6 索引安装对应构建，再安装依赖：

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

具体 GPU 和驱动仍需检查兼容性；本机环境已经验证可用。

## 阶段 1 运行入口

以下命令在仓库根目录执行，`python` 应指向上述独立环境。训练入口检查数据审查与排除规则是否一致。

```powershell
python -m src.phase1_catdog_baseline.check_data --output results/phase1_catdog_baseline/data_check_new.json
python -m src.phase1_catdog_baseline.train --run-name pilot_001 --epochs 1 --train-per-class 200
python -m src.phase1_catdog_baseline.evaluate --checkpoint runs/phase1_catdog_baseline/pilot_001/best.pth --output results/phase1_catdog_baseline/pilot_001
python -m src.phase1_catdog_baseline.predict --checkpoint runs/phase1_catdog_baseline/pilot_001/best.pth --output submission/pilot_001.csv
```

默认配置为 ImageNet 预训练 ResNet-18，权重标准预处理（短边缩放至 256 后中心裁剪为 224×224，ImageNet 归一化），无随机增强；骨干和 BatchNorm 状态冻结，仅二分类层更新。使用 Adam、学习率 0.001、batch size 32、种子 42。默认 10 轮是初始预算，正式设置按短训练结果确定。完整验证集用于评估；模型按验证准确率优先、验证损失次优保存。运行目录和预测文件拒绝覆盖。

训练会记录实际配置、样本清单、环境、逐轮指标、耗时与最佳权重。短训练用于检查流程和估算时间，其预测不是最终提交结果。

## 实验与交付规则

- 猫狗标签固定为 `cat=0`、`dog=1`；测试 ID 从文件名提取，不使用文件枚举顺序。
- 验证集用于选择模型，测试集不参与调参。
- 每次运行单独保存配置、随机种子、逐轮指标与最佳权重，避免覆盖既有结果。
- GitHub 保存代码、配置、说明和精选结果；不保存数据集、虚拟环境、大型权重和完整运行日志。
- 最终 `submission.csv` 包含 `id,label` 两列和 500 条预测。小组报告另行管理，并按课程要求与 CSV 一并提交。

## 阶段 2：D2 数据增强对比

训练增强配置在 `configs/phase2_catdog_improvements/augmentation.json`：RandomResizedCrop(224)，面积采样范围 0.8–1.0、宽高比 0.9–1.1，水平翻转概率 0.5；保持 ImageNet 归一化。极端长宽比可能触发回退裁剪，面积范围不是每张图片的严格保证。验证与测试继续使用 D1 确定性预处理。骨干和 BatchNorm 状态冻结，仅训练分类层；其他设置与 D1 相同。

正式 D2 从相同的 ImageNet 权重及种子初始化，不接着 D1 最佳权重训练。D2 训练准确率来自增强图片，不能与 D1 干净训练图片准确率直接等同解读。单次种子比较只提供初步证据。

```powershell
python -m src.phase2_catdog_improvements.train --run-name NEW_D2_NAME
python -m src.phase2_catdog_improvements.run_pipeline --run-name NEW_D2_PIPELINE_NAME
```

自动流水线要求已有短训练成功记录，依次完成正式训练、重载评估、D1/D2 对比和候选测试预测。状态及日志存放在 `runs/phase2_catdog_improvements/pipeline_运行名/`。结果会核对数据清单、基础配置、验证图片与标签一致，记录修正、新增和持续错误。D3 局部微调未执行，是否需要将在 D2 完成后评估。

## 阶段 2：增强条件下的学习率对照

最终课程 `submission.csv` 来自猫狗模型对 `datasets/test` 的500张课程测试图片的预测，与CIFAR-10无关。猫狗验证集是课程原有 `datasets/val/cat` 和 `datasets/val/dog` 各2500张；结果目录里的 `predictions.json` 是这5000张验证图片的输出，不是新生成的数据集。

新增配置 `configs/phase2_catdog_improvements/augmentation_lr0003.json` 与原D2仅学习率不同（0.001改为0.0003）。保持完整训练清单、seed42、batch32、10轮、相同增强、冻结骨干和BatchNorm；使用原图流程重新训练，不续训、不缓存随机增强特征。

```powershell
python -m src.phase2_catdog_improvements.augmentation_lr_compare --run-name NEW_AUGMENTATION_LR_RUN
```

入口拒绝覆盖既有目录，顺序执行训练、原图重载验证、500条测试预测，并核对样本清单、最佳轮次、指标和CSV。与原D2的逐图变化保存到该结果目录 `augmentation_lr_comparison.json` 及三份错误清单。单seed结果不能证明稳定改善；新候选CSV不自动成为最终submission.csv。

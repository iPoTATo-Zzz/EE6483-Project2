# 五个核心猫狗实验索引

小组报告统一使用E1–E5。它们对应历史执行顺序2、4、5、6、8；试跑和技术对照不纳入主要结果表。

|编号|配置|结果目录（相对results）|候选CSV（相对submission）|
|---|---|---|---|
|E1|ResNet-18，无增强，lr0.001|phase1_catdog_baseline/baseline_full_20261006_001/|baseline_full_20261006_001.csv|
|E2|ResNet-18，有增强，lr0.001|phase2_catdog_improvements/augmentation_full_20261006_001/|augmentation_full_20261006_001.csv|
|E3|ResNet-18，无增强，lr0.0003|phase2_catdog_improvements/resnet18_lr0003/|resnet18_lr0003.csv|
|E4|MobileNetV3-Small，无增强，lr0.001|phase2_catdog_improvements/mobilenetv3small_baseline/|mobilenetv3small_baseline.csv|
|E5|ResNet-18，有增强，lr0.0003|phase2_catdog_improvements/augmentation_lr0003_20261006_001/|augmentation_lr0003_20261006_001.csv|

## 统一阅读入口

`phase2_catdog_improvements/catdog_complete/catdog_detailed_analysis_report.pdf`及同名HTML为当前综合报告；该目录的`team_progress_brief.md`为当前组员同步文案。综合报告中的缓存对照属于历史技术核验，不是第六个核心实验；其独立文件已删除。旧分阶段HTML与综合目录重复history副本已清理，历史路径说明以本索引为准。

## 每个实验查什么

- config.json：实际训练配置。E1/E2/E5原图逐批；E3/E4缓存冻结特征，执行方式必须注明。
- history.csv：10轮训练与验证记录，检查最佳轮次及趋势。
- metrics.json：最佳权重验证指标、类别P/R/F1和混淆矩阵。
- predictions.json：5000张课程验证图的逐图标签与预测；不是新数据集。
- timing.json：实际计时及其口径，不跨缓存／原图流程直接比较耗时。
- 错误CSV：查看修正、新增和持续错误；E3/E4名称中的vs_cached_control表示历史比较对象，不能改写来源。
- submission/<运行名>.csv：对应模型对课程500张猫狗test图片的候选预测，最终唯一submission.csv尚未确定。

## 试跑和GitHub

两个试跑仅在runs中；.gitignore已忽略整个/runs/，git ls-files runs为空，核对试跑配置确实被忽略。保留本机试跑记录，不需删除，也不再添加重复忽略规则。不要使用git add -f强制加入。当前结果目录不含试跑实验。数据集和权重不上传；代码、实际五组结果和综合报告可上传。本次未推送。

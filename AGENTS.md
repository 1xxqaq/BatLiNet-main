# BatLiNet 项目协作说明

这份文档用于让新的 Codex 对话快速接手当前 BatLiNet 项目。重点是代码结构、运行流程、数据约定、已经完成的修改、实验与评估方式。它应当优先记录已经确认的工程事实，少写未验证的研究设想。

## 1. 当前项目状态

当前代码目录：

```text
D:\GongZuo\Codex\projects\batlinet1\BatLiNet-main
```

外层目录：

```text
D:\GongZuo\Codex\projects\batlinet1
```

注意：

- `BatLiNet-main` 本身是 Git 仓库。
- 截图中 GitLens 显示分支为 `main`，并且已经和 GitHub 上的 `origin/main` 同步。
- 新增本文件之前，`git status --short --branch` 在 `BatLiNet-main` 下显示：

```text
## main...origin/main
```

说明当时工作树与远程 `main` 没有未提交差异。新增本文件后，
`AGENTS.md` 会显示为未跟踪文件，需要提交并推送后才会保存到 GitHub。

历史上外层目录 `batlinet1` 不是 Git 仓库，因此以后执行 git 命令时应进入：

```text
D:\GongZuo\Codex\projects\batlinet1\BatLiNet-main
```

不要在外层目录误判项目没有 Git。


## 2. 本地与服务器的数据约定

当前本地代码副本主要用于：

- 修改代码；
- 阅读源码；
- 做不依赖完整数据的静态检查；
- 下载服务器 workspace 后在本地做结果比较与画图。

当前本地副本里不一定有：

```text
BatLiNet-main\data\processed
```

这是正常的。用户会在服务器上把代码和需要的数据一起上传到同一个项目目录下，因此服务器上应当按如下结构理解：

```text
BatLiNet-main/
    data/
        processed/
            CALCE/
            RWTH/
            UL_PUR/
            SNL/
            MATR/
            HUST/
            HNEI/
    scripts/
    configs/
    src/
```

所以配置文件中的相对路径：

```yaml
data/processed/CALCE
data/processed/RWTH
data/processed/UL_PUR
data/processed/SNL
data/processed/MATR
data/processed/HUST
data/processed/HNEI
```

应当视为服务器正式运行时有效，不要因为本地缺少 `data/processed` 就改掉配置。


## 3. 与之前复现相关的路径

之前已经复现成功的原始 BatLiNet `mix_20` 结果在：

```text
D:\Documents\研一\论文阅读\复现代码\Batlinet\BatLiNet-main\workspaces\mix_20_0424
```

之前用于对比作者结果的作者 workspace：

```text
D:\Documents\研一\论文阅读\复现代码\Batlinet\ds340.batlinet-main\code\workspaces\batlinet\mix_20
```

之前的作者结果 vs 复现结果分析输出：

```text
D:\Documents\研一\论文阅读\复现代码\Batlinet\BatLiNet-main\analysis\mix_20_comparison
```

当前仓库里也带有一份分析结果副本：

```text
BatLiNet-main\analysis\mix_20_comparison
```

后续测试新增方法时，优先用之前的 `mix_20_0424` 作为原始 BatLiNet 基线，不需要重新跑原始版，除非需要完全同一代码环境下的重新对照。


## 4. 项目目录与各部分作用

核心目录：

```text
scripts/
configs/
src/
analysis/
notebooks/
nmi_rebuttal/
nmi_rebuttal_final/
transfer_reproduce/
```

其中：

- `scripts/`：命令行入口、批量运行脚本、下载与预处理脚本、结果对比脚本。
- `configs/`：所有实验配置。模型、数据划分、特征、标签、变换都由 YAML 配置指定。
- `src/`：核心源码。包含数据结构、特征提取、标签构造、划分策略、模型实现、注册器等。
- `analysis/`：结果分析输出目录。
- `notebooks/`、`nmi_rebuttal*`、`transfer_reproduce/`：论文图表、反驳实验、迁移实验等 notebook 或资产。

后续最常改的文件通常是：

```text
src/models/rul_predictors/batlinet.py
configs/ablation/diff_branch/batlinet_weighted/mix_20.yaml
scripts/compare_experiment_results.py
```


## 5. 配置与注册机制

项目通过注册器根据配置文件实例化对象。

注册器定义：

```text
src/builders.py
src/utils/registry.py
```

`src/builders.py` 中定义了这些注册表：

```python
MODELS
LABEL_ANNOTATORS
FEATURE_EXTRACTORS
TRAIN_TEST_SPLITTERS
DATA_TRANSFORMATIONS
```

每个可配置类通过装饰器注册，例如：

```python
@MODELS.register()
class BatLiNetRULPredictor(...)
```

配置中写：

```yaml
model:
    name: 'BatLiNetRULPredictor'
```

`MODELS.build(configs['model'])` 会根据 `name` 找到类，并把除 `name` 以外的键值作为构造参数传入。

配置读取逻辑在：

```text
src/utils/config.py
```

主入口 `scripts/pipeline.py` 使用：

```python
CONFIGS = [
    'model',
    'train_test_split',
    'feature',
    'label',
    'feature_transformation',
    'label_transformation'
]
```

也就是说一个实验配置主要由这六块组成。


## 6. 主运行入口 pipeline.py

主入口文件：

```text
scripts/pipeline.py
```

典型运行命令：

```bash
PYTHONPATH=. python scripts/pipeline.py configs/ablation/diff_branch/batlinet/mix_20.yaml --train True --evaluate True --device cuda:0
```

批量跑多个随机种子：

```bash
./scripts/run_pipeline_with_n_seeds.sh configs/ablation/diff_branch/batlinet_weighted/mix_20.yaml 8
```

`pipeline.py` 主要步骤：

1. `set_seed(seed)`  
   设置 Python、NumPy、PyTorch 随机种子，并设置 cudnn deterministic。

2. `load_config(config_path, workspace)`  
   读取 YAML 配置，并确定 workspace。
   - 如果配置里 `model.workspace` 不为空，优先使用它。
   - 否则如果命令行传了 `--workspace`，使用命令行。
   - 否则默认使用：

```text
workspaces/<config 文件名>
```

3. `MODELS.build(configs['model'])`  
   根据配置实例化模型。

4. 如果 `--train True`，调用：

```python
model.fit(dataset, timestamp=ts)
```

5. 如果 `--evaluate True`，调用：

```python
prediction = model.predict(dataset)
scores = {m: dataset.evaluate(prediction, m) for m in metric}
```

6. 保存预测结果：

```python
{
    'prediction': prediction,
    'scores': scores,
    'data': dataset.to('cpu'),
    'seed': seed,
}
```

文件名类似：

```text
predictions_seed_0_20260510123456.pkl
```

这些 `.pkl` 文件已经包含评估所需的预测、指标、数据和 seed，因此结果比较时通常不需要 `.ckpt`。


## 7. 数据构建 Task.build()

数据任务构建文件：

```text
src/task.py
```

核心类：

```python
Task
```

`Task.build()` 的完整流程：

1. 调用划分器：

```python
train_list, test_list = self.train_test_splitter.split()
```

2. 读取训练电池和测试电池：

```python
train_cells = [BatteryData.load(path) for path in train_list]
test_cells = [BatteryData.load(path) for path in test_list]
```

3. 提取特征：

```python
train_features = self.feature_extractor(train_cells)
test_features = self.feature_extractor(test_cells)
```

4. 构造标签：

```python
train_labels = self.label_annotator(train_cells)
test_labels = self.label_annotator(test_cells)
```

5. 删除 NaN 标签样本：

```python
train_mask = ~torch.isnan(train_labels)
test_mask = ~torch.isnan(test_labels)
```

6. 构造 `DataBundle`：

```python
dataset = DataBundle(
    train_features,
    train_labels,
    test_features,
    test_labels,
    feature_transformation=...,
    label_transformation=...
)
```

`Task.build()` 返回的是训练和测试都已变换好的 `DataBundle`。


## 8. BatteryData 数据结构

数据结构定义：

```text
src/data/battery_data.py
```

核心类：

```python
BatteryData
CycleData
CyclingProtocol
```

`BatteryData` 表示一块电池，关键字段包括：

```python
cell_id
cycle_data
nominal_capacity_in_Ah
charge_protocol
discharge_protocol
max_voltage_limit_in_V
min_voltage_limit_in_V
max_current_limit_in_A
min_current_limit_in_A
anode_material
cathode_material
```

`CycleData` 表示一个循环，关键字段包括：

```python
cycle_number
voltage_in_V
current_in_A
charge_capacity_in_Ah
discharge_capacity_in_Ah
time_in_s
temperature_in_C
internal_resistance_in_ohm
```

处理后的 `.pkl` 文件保存的是字典形式。读取时：

```python
BatteryData.load(path)
```

会重新构造 `BatteryData`、`CycleData`、`CyclingProtocol` 对象。


## 9. 训练/测试划分

划分器基类：

```text
src/train_test_split/base.py
```

基类会读取 `cell_data_path`：

- 如果是目录，就读取目录下的所有 `.pkl`。
- 如果是文件，就把文件中的每一行作为电池路径。

`mix_20` 使用：

```text
src/train_test_split/MIX20_split.py
```

类名：

```python
MIX20TrainTestSplitter
```

它内部写死了一组 `test_ids`。遍历所有 `.pkl` 文件时：

- `filename.stem` 在 `test_ids` 中，则放入测试集；
- 否则放入训练集。

这保证了 `mix_20` 任务的划分可复现，不依赖机器上的随机排列。


## 10. BatLiNet 特征提取

特征文件：

```text
src/feature/batlinet.py
```

类名：

```python
BatLiNetFeatureExtractor
```

基类：

```text
src/feature/base.py
```

`BaseFeatureExtractor.__call__()` 会对电池列表逐个调用：

```python
self.process_cell(cell)
```

然后 `torch.stack(features)` 得到整体特征张量。

### 10.1 输入数据

对每个 cycle，`BatLiNetFeatureExtractor.process_cell()` 会读取：

```python
I = current_in_A
V = voltage_in_V
Qc = charge_capacity_in_Ah / nominal_capacity_in_Ah
Qd = discharge_capacity_in_Ah / nominal_capacity_in_Ah
```

然后用电流正负划分充电和放电段：

```python
charge_mask = I > 0.1
discharge_mask = I < -0.1
```

### 10.2 六个通道

每个 cycle 构造六个通道：

1. `V(Qc)`：充电电压-容量曲线。
2. `V(Qd)`：放电电压-容量曲线。
3. `I(Qc)`：充电电流-容量曲线。
4. `I(Qd)`：放电电流-容量曲线。
5. `delta_V(Q)`：`V(Qc) - reverse(V(Qd))`。
6. `R(Q)`：`delta_V(Q) / (I(Qc) - reverse(I(Qd)) + eps)`。

容量轴统一插值到 `interp_dim`，默认 1000。

### 10.3 输出形状

单个 cycle 的通道堆叠后大致是：

```text
[6, 1000]
```

多个早期 cycle 堆叠后，原始顺序是：

```text
[cycle, channel, width]
```

最后执行：

```python
feature = feature.transpose(1, 0)
```

所以单块电池最终特征形状是：

```text
[channel, cycle, width]
```

在 `mix_20` 中：

```text
[6, 20, 1000]
```

模型配置对应：

```yaml
in_channels: 6
input_height: 20
input_width: 1000
```


## 11. RUL 标签构造

标签文件：

```text
src/label/rul.py
```

类名：

```python
RULLabelAnnotator
```

主要参数：

```python
eol_soh: float = 0.8
pad_eol: bool = True
min_rul_limit: float = 100.0
```

`mix_20` 配置使用：

```yaml
eol_soh: 0.9
```

标签构造逻辑：

1. 遍历 `cycle_data`。
2. 每个 cycle 取最大放电容量：

```python
Qd = max(cycle.discharge_capacity_in_Ah)
```

3. 如果：

```python
Qd <= nominal_capacity_in_Ah * eol_soh
```

则认为达到寿命终止阈值。

4. 输出达到阈值时的 cycle 计数作为寿命标签。
5. 如果没有找到 EOL：
   - `pad_eol=True` 时补一个周期；
   - 否则置为 NaN。
6. 如果寿命过短：

```python
label <= min_rul_limit
```

则置为 NaN，后续在 `Task.build()` 中丢弃。


## 12. DataBundle 与标签变换

文件：

```text
src/data/databundle.py
```

核心类：

```python
Dataset
DataBundle
```

`Dataset` 只是包装：

```python
feature
label
```

并实现 `__getitem__()` 返回：

```python
{
    'feature': self.feature[item],
    'label': self.label[item]
}
```

`DataBundle` 负责：

1. 将特征和标签转为 float。
2. 对训练特征拟合特征变换，并应用到训练/测试特征。
3. 对训练标签拟合标签变换，并应用到训练/测试标签。
4. 构造：

```python
self.train_data
self.test_data
```

`mix_20` 的标签变换是：

```text
LogScaleDataTransformation
ZScoreDataTransformation
```

因此模型训练时的标签不是原始寿命，而是：

```text
log(寿命) 后再标准化
```

评估时：

```python
dataset.evaluate(prediction, metric)
```

会先对 `target` 和 `prediction` 做 `inverse_transform`，再计算原始寿命尺度上的：

```text
RMSE
MAE
MAPE
```


## 13. 原始 BatLiNet 模型

模型文件：

```text
src/models/rul_predictors/batlinet.py
```

类名：

```python
BatLiNetRULPredictor
```

卷积模块来自：

```text
src/models/rul_predictors/cnn.py
```

`ConvModule` 基本结构：

```text
Conv2d
激活函数
AvgPool2d
Conv2d
激活函数
AvgPool2d
```

然后 `build_module()` 会接一个投影卷积：

```python
proj = nn.Conv2d(channels, channels, (H, W))
```

将空间维度压到 `1 x 1`，最后得到长度为 `channels` 的向量。

### 13.1 两个分支

BatLiNet 有两个主要分支：

1. 目标电池自身分支：

```python
x_ori = self.ori_module(feature)
y_ori = self.fc(x_ori.view(B, self.channels)).view(-1)
```

输入是目标电池自己的早期特征。

2. 参考电池差异分支：

```python
x_sup = self.sup_module(support_feature.view(-1, C, H, W))
y_sup = self.fc(x_sup).view(B, S)
y_sup += support_label.view(B, S)
```

输入是目标电池与参考电池的差异特征。

这里的思想是：

```text
目标寿命 ≈ 参考电池寿命 + 模型预测的目标-参考寿命差修正
```

所以每个参考电池都会给出一个单独的目标寿命预测。

### 13.2 参考电池从哪里来

参考电池在：

```python
get_support_set(self, x, sup_feat, sup_label)
```

中随机采样。

训练时：

```python
size = (len(x) * self.train_support_size,)
```

测试时：

```python
size = (len(x) * self.test_support_size,)
```

然后：

```python
indx = torch.randint(len(sup_feat), size, device=x.device)
```

参考电池永远从训练集特征和训练集标签中采样：

```python
dataset.train_data.feature
dataset.train_data.label
```

### 13.3 差异特征如何构造

`get_support_set()` 中：

```python
feature = x.unsqueeze(1) - sup_feat[indx].view(B, -1, C, H, W)
label = sup_label[indx].view(B, -1)
```

其中：

- `x` 是目标电池原始特征；
- `sup_feat` 是训练集中候选参考电池的特征；
- `feature` 就是目标电池减参考电池后的差异特征；
- `label` 是对应参考电池的变换后寿命标签。

### 13.4 cycle diff 数据集

训练和预测前都会调用：

```python
build_cycle_diff_dataset()
```

这里先做：

```python
feature = dataset.feature - dataset.feature[:, :, [self.diff_base]]
```

也就是将每块电池每个 cycle 的特征减去某个基准 cycle。`mix_20` 中：

```yaml
diff_base: 0
```

因此是减第 0 个 cycle。

`raw_feature` 保留为原始特征，用于后续构造目标-参考差异：

```python
raw_feature = dataset.feature
```

`DiffDataset.__getitem__()` 返回：

```python
{
    'feature': cycle_diff_feature,
    'label': label,
    'raw_feature': raw_feature
}
```

训练时：

- `feature` 进入目标电池自身分支；
- `raw_feature` 用来和参考电池特征相减，构造目标-参考差异。

### 13.5 原始聚合与损失

原始 BatLiNet 中，每个目标电池有多个参考电池预测：

```text
y_sup_1, y_sup_2, ..., y_sup_S
```

训练时：

```python
y_sup = y_sup.mean(1).view(-1)
```

测试时：

```python
y_sup = y_sup.median(1)[0].view(-1)
```

损失函数：

```python
loss = (
    (1. - self.alpha) * mse(y_ori, label)
    + self.alpha * mse(y_sup, label)
)
```

其中：

```python
def mse(pred, label):
    return ((pred.view(-1) - label.view(-1)) ** 2).mean()
```

最终预测：

```python
return (1. - self.alpha) * y_ori + self.alpha * y_sup
```

默认 `alpha=0.5`，即目标自身分支和参考电池分支各占一半。


## 14. 已新增方法：参考电池可信度加权

已在：

```text
src/models/rul_predictors/batlinet.py
```

中加入可配置聚合方式。

新增参数：

```python
support_aggregation: str = 'original'
score_head_type: str = 'mlp'
score_hidden_channels: int = None
score_temperature: float = 1.0
teacher_temperature: float = 1.0
score_loss_weight: float = 0.0
warmup_epochs: int = 0
```

支持的聚合模式：

```text
original
mean
median
learned_weighted
supervised_weighted
```

含义：

- `original`：保持原始 BatLiNet 行为，训练均值、测试中位数。
- `mean`：训练和测试都取均值。
- `median`：训练和测试都取中位数。
- `learned_weighted`：学习每个参考电池的权重，用加权平均代替固定均值/中位数。
- `supervised_weighted`：预热阶段保持原始聚合，预热后启用学习权重，并用单独参考预测误差构造教师权重来监督打分头。

`learned_weighted` 的打分器：

```python
self.score_head = nn.Sequential(
    nn.Linear(channels, score_hidden_channels),
    nn.ReLU(),
    nn.Linear(score_hidden_channels, 1)
)
```

输入是 `self.sup_module` 提取出的目标-参考关系向量：

```python
x_sup = self.sup_module(support_feature.view(-1, C, H, W))
x_sup = x_sup.view(B, S, self.channels)
```

打分与加权：

```python
score = self.score_head(x_sup).squeeze(-1)
score = score / self.score_temperature
weight = torch.softmax(score, dim=1)
y_sup = (weight * y_sup).sum(1).view(-1)
```

这里的分数不是人工规则，而是由最终预测误差反向传播学出来的。直观解释：

```text
如果某类目标-参考关系经常让加权后的预测更接近真实寿命，
打分器会倾向于给这类参考电池更高权重。
```

本次修改没有改变损失函数主体。这样对比时更清楚：实验主要检验“参考电池聚合方式”是否优于原来的均值/中位数策略。

后续又新增了 `supervised_weighted`，它会在训练阶段额外计算：

```text
e_i = |y_sup_i - label|
q_i = softmax(-detach(e_i) / teacher_temperature)
```

其中 `q_i` 是教师权重，表示“当前目标电池下哪个参考电池单独预测更准”。模型打分头输出学生权重 `w_i`，并加入：

```text
L_score = KL(q || w)
```

总损失变为：

```text
L = 原始预测损失 + score_loss_weight * L_score
```

`warmup_epochs` 用于前若干 epoch 保持原始训练逻辑，让单独参考预测先具备基本意义，然后再启用加权和打分监督。


## 15. 新增 weighted 配置

新增配置：

```text
configs/ablation/diff_branch/batlinet_weighted/mix_20.yaml
```

它基本复制原始：

```text
configs/ablation/diff_branch/batlinet/mix_20.yaml
```

只在模型配置中加入：

```yaml
support_aggregation: 'learned_weighted'
score_hidden_channels: 16
score_temperature: 1.0
```

服务器测试命令：

```bash
./scripts/run_pipeline_with_n_seeds.sh configs/ablation/diff_branch/batlinet_weighted/mix_20.yaml 8
```

该命令会用 8 个 seed 跑 `mix_20`。如果服务器有多张 GPU，脚本会按 seed 对 GPU 取模分配。

预期输出 workspace：

```text
workspaces/ablation/diff_branch/batlinet_weighted/mix_20
```

具体取决于 `run_pipeline_with_n_seeds.sh` 中根据 config 相对路径生成 workspace 的逻辑。

新增有监督关系可信度加权配置：

```text
configs/ablation/diff_branch/batlinet_supervised_weighted/mix_20.yaml
```

关键差异：

```yaml
train_support_size: 4
test_support_size: 32
support_aggregation: 'supervised_weighted'
score_head_type: 'linear'
score_temperature: 1.0
teacher_temperature: 1.0
score_loss_weight: 0.05
warmup_epochs: 100
```

服务器测试命令：

```bash
./scripts/run_pipeline_with_n_seeds.sh configs/ablation/diff_branch/batlinet_supervised_weighted/mix_20.yaml 8
```


## 16. 结果比较脚本

保留旧脚本：

```text
scripts/compare_mix20_results.py
```

它是之前“作者结果 vs 用户复现结果”的专用比较脚本，默认路径固定，更适合复查之前复现是否成功。

新增通用脚本：

```text
scripts/compare_experiment_results.py
```

用途：

- 比较任意两个包含 `predictions_seed_*.pkl` 的 workspace；
- 默认用于比较原始 BatLiNet 和 `learned_weighted`；
- 不加载模型；
- 不需要 checkpoint；
- 可以处理 CUDA 序列化张量和 `src.*` 对象反序列化问题。

本地比较命令示例：

```powershell
conda run -n batlinet python scripts/compare_experiment_results.py `
  --baseline-workspace "D:\Documents\研一\论文阅读\复现代码\Batlinet\BatLiNet-main\workspaces\mix_20_0424" `
  --candidate-workspace "D:\你的下载路径\workspaces\ablation\diff_branch\batlinet_weighted\mix_20" `
  --baseline-name "batlinet_original" `
  --candidate-name "batlinet_weighted" `
  --experiment-name "mix_20" `
  --output-dir "analysis\mix_20_weighted_comparison"
```

输出：

```text
analysis/mix_20_weighted_comparison/
    tables/
        seed_level_comparison.csv
        summary_statistics.csv
    figures/
        metric_means_bar.png
        metrics_by_seed_lines.png
    summary/
        comparison_summary.txt
```

如果同一个 seed 有多个 `predictions_seed_*.pkl`，脚本会保留修改时间最新的那个。


## 17. 从服务器下载哪些文件

如果只是做指标比较，下载 weighted workspace 中这些文件即可：

```text
predictions_seed_0_*.pkl
predictions_seed_1_*.pkl
...
predictions_seed_7_*.pkl
```

建议同时下载：

```text
log.0
log.1
...
log.7
config_*.yaml
```

暂时不需要下载：

```text
*.ckpt
latest.ckpt
```

只有在后续需要重新推理、查看模型参数、提取每个参考电池权重时，才需要 checkpoint。


## 18. 依赖环境

`requirements.txt` 已补充项目实际导入的依赖：

```text
fire
numpy
scipy
pandas
h5py
addict
scikit-learn
numba
openpyxl
matplotlib
pyyaml
tqdm
requests
```

PyTorch 没有固定写入 `requirements.txt`，因为服务器 CUDA 版本可能不同，应按目标环境单独安装。

本地存在 conda 环境：

```text
batlinet
```

路径：

```text
D:\GongZuo\Anaconda\azb\envs\batlinet
```

之前检查时：

- `compare_experiment_results.py --help` 能在该环境运行；
- 该环境缺 `numba`；
- 该环境也缺 `requests`。

如果要在本地完整导入模型或运行预处理，需要补依赖：

```powershell
conda run -n batlinet pip install -r requirements.txt
```

或者只补缺失项：

```powershell
conda run -n batlinet pip install numba==0.58.1 requests==2.31.0
```

注意：不要在不了解 CUDA 版本的情况下随意用 pip 重装 PyTorch。


## 19. 已做过的检查

已运行并通过：

```powershell
python -m compileall .\src\models\rul_predictors\batlinet.py .\scripts\pipeline.py
python -m py_compile .\src\models\rul_predictors\batlinet.py
python -m py_compile .\scripts\compare_experiment_results.py .\src\models\rul_predictors\batlinet.py
conda run -n batlinet python .\scripts\compare_experiment_results.py --help
```

尝试用随机张量实例化并前向测试 `BatLiNetRULPredictor` 时，本地环境在导入阶段因为缺少 `numba` 失败，还没真正执行到模型前向。因此这不是模型形状错误，而是本地环境依赖不完整。


## 20. 当前推荐下一步

当前最自然的下一步是：

1. 在服务器上使用带 `data/processed` 的完整项目目录。
2. 确认服务器环境依赖完整。
3. 运行 supervised weighted `mix_20`：

```bash
./scripts/run_pipeline_with_n_seeds.sh configs/ablation/diff_branch/batlinet_supervised_weighted/mix_20.yaml 8
```

4. 下载 supervised weighted workspace 的 `predictions_seed_*.pkl`、`log.*` 和 `config_*.yaml`。
5. 本地用 `compare_experiment_results.py` 对比：
   - baseline：之前复现的 `mix_20_0424`；
   - candidate：服务器下载的 `batlinet_supervised_weighted/mix_20`。

如果 supervised weighted 指标有提升，或者某些 seed 表现有明显差异，后续值得新增诊断模式，保存：

- 目标电池自身分支预测 `y_ori`；
- 每个参考电池单独给出的预测 `y_sup_i`；
- 每个参考电池的学习权重；
- 被采样参考电池的索引或 `cell_id`；
- 最终聚合预测与真实标签。

当前的 `predictions_seed_*.pkl` 还不包含这些细粒度信息，所以若要做参考电池可信度分析，需要继续改模型推理和保存逻辑。

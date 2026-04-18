# Pointcept 骨架化与长度计算流程说明

本文档说明当前项目中“按标签骨架化”和“基于骨架计算长度”的完整流程，以及这两个步骤背后的算法原理。

适用脚本：

- [skeletonize_pointcept_instances.py](/f:/CodeProject/Python/UT/pc-skeletor/tools/skeletonize_pointcept_instances.py)
- [compute_skeleton_curve_lengths.py](/f:/CodeProject/Python/UT/pc-skeletor/tools/compute_skeleton_curve_lengths.py)

## 1. 目标

当前流程的目标有两个：

1. 将 Pointcept 的分割结果按标签拆分，并分别骨架化。
2. 基于每个标签的骨架点云，拟合连续曲线并计算长度。

这里的“标签”来自 Pointcept 输出的 `pred.npy`，不是实例 ID。

因此当前流程的语义是：

- 每个标签输出一个骨架
- 整个点云额外输出一个完整骨架
- 每个标签长度基于该标签自己的骨架结果计算
- 完整长度基于完整骨架结果计算

## 2. 输入与输出

### 2.1 骨架化输入

推荐输入：

- `coord.npy`：点云坐标，形状为 `(N, 3)`
- `pred.npy`：逐点标签，形状为 `(N,)`

对应脚本参数：

```bash
python pc-skeletor/tools/skeletonize_pointcept_instances.py \
  --coord-npy Pointcept/data/byme_partseg_36/train/8/coord.npy \
  --pred-npy Pointcept/exp/byme/semseg-pt-v3m1-0-base-36parts/result/8_pred.npy \
  --output-dir pc-skeletor/output/8_pred_label_skeletons \
  --group-mode label \
  --method lbc \
  --down-sample 0.01 \
  --min-points 32
```

### 2.2 骨架化输出

输出根目录形如：

```text
pc-skeletor/output/8_pred_label_skeletons/
```

其中包含：

- `summary.json`：本次骨架化的总汇总
- `groups/label_xx/`：每个标签的骨架结果
- `full/`：整云骨架结果

每个标签目录通常包含：

- `input_points.ply`
- `skeleton.ply`
- `meta.json`

如果运行时显式传入 `--extract-topology`，还会额外输出：

- `topology.ply`
- `skeleton_graph.gpickle`
- `topology_graph.gpickle`

### 2.3 长度计算输入

长度计算脚本不重新骨架化，而是直接读取上一步的输出目录：

```bash
python pc-skeletor/tools/compute_skeleton_curve_lengths.py \
  --skeleton-root pc-skeletor/output/8_pred_label_skeletons
```

### 2.4 长度计算输出

长度脚本会生成：

- `curve_length_summary.json`
- `curve_length_summary.csv`

其中记录：

- 每个标签的长度
- 完整骨架长度
- 连通分量数量
- 各分量长度
- 是否存在回环并经过清理
- 实际使用的 `skeleton.ply` 路径

## 3. 骨架化流程

### 3.1 整体流程

骨架化脚本的流程如下：

1. 读取 `coord.npy` 和 `pred.npy`
2. 将每个点的标签映射为伪彩色，仅用于可视化和构造点云颜色
3. 按标签分组
4. 对每个标签单独调用 `pc-skeletor` 的 LBC 方法
5. 对整个点云再额外运行一次骨架化，输出完整骨架
6. 将结果写入 `groups/` 和 `full/`
7. 写出 `summary.json`

当前脚本默认：

- 分组骨架化用 `LBC`
- 整云骨架化默认也用 `LBC`
- `topology` 默认不提取

### 3.2 为什么按标签骨架化和完整骨架化会略有差异

这两者输入的数据本来就不同：

- 分组骨架化只看到某个标签对应的点
- 完整骨架化看到的是全部点

这会影响：

- 几何收缩路径
- 局部邻域关系
- 噪声与连接结构
- 最终骨架点的位置和密度

所以“完整骨架”和“标签骨架”存在细微差异是正常现象，不表示脚本运行错误。

## 4. 骨架化算法原理

### 4.1 LBC 是什么

`pc-skeletor` 当前使用的核心方法是 LBC，通常可理解为一种基于 Laplacian 收缩的骨架提取方法。

它的大致思想是：

1. 将原始点云视为一个离散几何对象
2. 通过迭代收缩，让点云逐渐向“中心线”靠拢
3. 保留整体分支结构
4. 最终从收缩后的点云中提取骨架点

直观理解：

- 原始点云像是一根有厚度的“实体线束”
- LBC 通过几何收缩，把这根“粗的实体”压缩成“细的中心线”

### 4.2 当前脚本里真正保留了什么

在你的现阶段需求中，真正需要的是：

- `skeleton.ply`

也就是收缩和抽取得到的骨架点云。

当前默认不提取 `topology`，原因是：

- 你现在长度计算主要依赖 `skeleton.ply`
- `topology.ply` 更偏向结构简化，不是计算连续长度的最佳输入
- 某些小标签在 topology 阶段容易遇到邻居数边界问题

### 4.3 topology 原本是做什么的

`topology` 的作用不是重新生成骨架，而是把骨架点进一步压缩成更简洁的结构图。

它更适合：

- 看分叉关系
- 看连通结构
- 做图结构分析

但它不一定适合：

- 直接做连续曲线拟合
- 直接估计精细长度

因为 `topology.ply` 往往更稀疏、更抽象。

## 5. 为什么现在默认不提取 topology

这是基于你当前任务做出的策略调整。

你的目标是：

- 每个标签得到可靠的骨架
- 再基于骨架计算长度

而不是：

- 重点分析分叉图结构

所以脚本现在默认：

- 提取 `skeleton`
- 跳过 `topology`

只有在显式传入 `--extract-topology` 时，才会继续执行 topology 提取。

这样做的好处是：

1. 避免 topology 阶段的边界错误影响 skeleton 输出
2. 使 `label_32` 这样的标签也能正常产出 `skeleton.ply`
3. 更贴合长度计算脚本的真实输入需求

## 6. 长度计算流程

### 6.1 当前长度计算到底基于哪份骨架

这一点很重要：

- 每个标签的长度，基于 `groups/label_xx/skeleton.ply`
- 完整骨架长度，基于 `full/skeleton.ply`

也就是说：

- 标签长度不是从 `full/skeleton.ply` 中切出来的
- 而是基于每个标签独立骨架化后的结果单独计算

这也是为什么标签长度和完整骨架中的局部形态可能存在轻微差异。

### 6.2 长度计算整体流程

长度脚本的流程如下：

1. 读取骨架输出根目录下的 `summary.json`
2. 对每个标签条目，读取对应目录下的 `skeleton.ply`
3. 对骨架点构建局部近邻图
4. 将图拆成多个连通分量
5. 对每个分量做最小生成树清理
6. 从树中提取主路径
7. 沿主路径拟合连续 3D 曲线
8. 对曲线积分得到长度
9. 将各分量长度求和，得到该标签总长度
10. 对 `full/skeleton.ply` 重复同样流程

## 7. 长度算法原理

### 7.1 为什么不能直接对 skeleton 点简单累加

`skeleton.ply` 虽然已经比原始点云细很多，但它仍然只是离散点云，不是天然连好的曲线。

如果直接按文件顺序累加点与点距离，会出现几个问题：

- 点的存储顺序不代表几何顺序
- 局部可能有断点
- 同一标签中可能有多个分支或多个孤立段
- 某些元器件会带来小回环或小分叉

因此必须先“建图”和“排序”，再谈长度。

### 7.2 k 近邻建图

长度脚本先把每个骨架点看成图节点，再让每个点和空间上最近的若干个点建立连接。

当前参数：

- `--k-neighbors`

默认值是 `4`。

这样做的目的，是从离散点中恢复局部连续关系。

### 7.3 连通分量拆分

建图之后，脚本会计算连通分量。

`component_count` 的含义就是：

- 当前标签骨架图被分成了多少个互不连通的部分

例如：

- `component_count = 1`：基本是一整段
- `component_count = 3`：表示这个标签骨架被分成三段

### 7.4 为什么要做最小生成树

原始近邻图可能存在：

- 小回环
- 杂散支路
- 局部多余连边

这些结构会让“长度”变得不稳定。

因此脚本会先将每个连通分量化成最小生成树。

最小生成树的作用是：

- 保持整体连通
- 去掉冗余回环
- 保留主要连接关系

这一步的目的不是完全消灭分叉，而是让主路径提取更稳定。

### 7.5 主路径提取

对最小生成树，脚本会提取一条“主路径”。

当前实现中，这条主路径可以理解为树的直径路径，也就是：

- 在树中距离最远的两个端点之间的路径

它代表该分量最主要的延伸方向。

这样做的原因是：

- 你的标签大体对应线束分支
- 分支上的元器件容易带来小支路
- 如果直接统计所有边长，元器件会把长度放大

主路径方法能更好地聚焦在主干长度上。

### 7.6 为什么还要拟合曲线

即使提取了主路径，路径仍然是离散折线。

如果直接把折线长度当最终结果，会受到这些因素影响：

- 骨架点采样密度不均
- 局部抖动
- 断点附近的尖锐折线

因此脚本会尝试将主路径拟合为平滑 3D 曲线，再沿曲线做弧长积分。

直观理解：

- 折线长度更像“沿着离散点走”
- 曲线长度更像“恢复连续中心线后再测量”

### 7.7 当前脚本里的回退策略

如果运行环境里没有足够依赖，或者点数太少，不适合稳定拟合样条，脚本会回退：

- 单点：长度记为 `0`
- 两点：长度记为两点欧氏距离
- 无法拟合样条时：退化为折线长度

因此结果里会同时保留：

- `curve_length_sum`
- `polyline_length_sum`

用于比较拟合前后的变化。

## 8. 单位说明

当前长度单位完全继承输入坐标单位。

也就是说：

- 如果 `coord.npy` 是米，长度结果就是米
- 如果 `coord.npy` 是毫米，长度结果就是毫米
- 如果坐标没有经过真实物理标定，长度结果就是该坐标系下的数值单位

脚本本身不会自动做单位换算。

## 9. 关键结果字段说明

### 9.1 骨架化 `summary.json`

常用字段：

- `groups`：每个标签的骨架化状态
- `full`：完整骨架化状态
- `status`：`ok`、`failed`、`skipped`
- `topology_status`：topology 是成功、失败还是跳过

### 9.2 长度汇总 `curve_length_summary.json`

常用字段：

- `curve_length_sum`：该标签的最终曲线长度
- `polyline_length_sum`：主路径折线长度
- `largest_component_length`：最长连通分量长度
- `component_count`：连通分量数量
- `component_lengths`：各分量长度
- `had_cycle_before_cleanup`：建图后是否检测到回环
- `source_skeleton_ply`：该长度实际使用的骨架文件路径

## 10. 推荐的理解方式

可以把这套流程理解成两层：

第一层是“几何提取”：

- 从分割点云中提取骨架点云

第二层是“几何测量”：

- 从骨架点云中恢复主路径
- 拟合连续曲线
- 计算长度

对应关系如下：

- `skeletonize_pointcept_instances.py` 负责提取骨架
- `compute_skeleton_curve_lengths.py` 负责测量长度

## 11. 当前项目中的推荐使用方式

对于你现在的任务，推荐做法是：

1. 使用标签模式运行骨架化
2. 默认不要开启 `--extract-topology`
3. 用 `groups/label_xx/skeleton.ply` 作为每个标签长度的输入
4. 用 `full/skeleton.ply` 作为完整长度的输入
5. 优先关注 `curve_length_sum`
6. 同时参考 `component_count` 和 `component_lengths` 判断该标签是否存在断裂或多段

## 12. 一个典型工作流

### 12.1 先做骨架化

```bash
python pc-skeletor/tools/skeletonize_pointcept_instances.py \
  --coord-npy Pointcept/data/byme_partseg_36/train/8/coord.npy \
  --pred-npy Pointcept/exp/byme/semseg-pt-v3m1-0-base-36parts/result/8_pred.npy \
  --output-dir pc-skeletor/output/8_pred_label_skeletons \
  --group-mode label \
  --method lbc \
  --down-sample 0.01 \
  --min-points 32
```

### 12.2 再做长度计算

```bash
python pc-skeletor/tools/compute_skeleton_curve_lengths.py \
  --skeleton-root pc-skeletor/output/8_pred_label_skeletons \
  --k-neighbors 4 \
  --resample-points 200 \
  --spline-smoothing 0.0001
```

## 13. 当前方案的优点与局限

### 13.1 优点

- 流程和 Pointcept 输出直接对接
- 每个标签单独骨架化，语义清晰
- 长度统计基于标签自己的骨架，而不是整云切分
- 对回环和小分叉有一定鲁棒性
- 可以同时得到标签级和整云级结果

### 13.2 局限

- 标签不是实例，无法区分同一标签下多个真实物理实例
- 骨架质量依赖点云分割质量
- 长度估计依赖骨架点的连通恢复质量
- 坐标单位若未标定，长度结果仅是相对量
- 如果骨架点断裂严重，`component_count` 会偏大

## 14. 总结

当前项目中的核心结论是：

- 骨架化阶段，默认只需要 `skeleton.ply`
- 长度计算阶段，使用的是每个标签自己的 `groups/label_xx/skeleton.ply`
- 完整骨架长度单独基于 `full/skeleton.ply`
- `topology` 不是当前长度计算的必要输入
- 曲线长度的本质是“从离散骨架点恢复主路径后，对连续曲线做弧长估计”

如果后续你需要，我们还可以继续在这份文档基础上补两类内容：

1. 常见异常的排查说明
2. 关键参数对结果的影响表

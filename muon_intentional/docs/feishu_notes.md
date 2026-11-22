# Muon × Intentional Updates 实验笔记

> 来源：飞书 wiki 个人实验笔记
> <https://gcn8rjv579wr.feishu.cn/wiki/GqpdwuNcGiHDkEkW66Ic8iF9nBh>
> 整理日期：2026-11-22
>
> 说明：以下内容按原文整理，只做了标题与列表的排版；`⚠️` 处为飞书嵌入块（图表/表格）在文档外无法显示时的占位。

## 两种 sensitivity

1. **Activation sensitivity**：衡量参数更新对本层 activation 输出的影响敏感度。
2. **Function sensitivity**：衡量参数更新对最终 logits 的影响敏感度。

## 小规模实验

小规模实验表明：**activation sensitivity 不能很好地代表 function sensitivity**，
不同层的参数更新对最终模型输出的影响存在显著差异。

定量上：

- Function-space sensitivity 的动态范围是 activation-space 的 **63 倍**；
- 两者层间排序的 **Spearman 相关系数只有 0.24**。

实验发现了一些比较明确的结构性规律：

- **Q projection**：activation sensitivity 较高，但 function sensitivity 处于中游。
  Function-space 规则反而会给它们更大的步长。
- **O projection**：activation sensitivity 较低，但 function-space 规则分配的步长略小。
- **最后一个 block 的 MLP**：尤其是 `b5.down`、`b5.up`、`b5.gate`，function sensitivity 明显偏高。

但效应主要集中在少数层，大部分层的 function-space sensitivity 差异相对温和，
显著差异主要由少数层贡献。

所以目前不能直接推断 layer-wise function-space calibration 一定能带来很大的训练收益。
它可能主要改善少数关键层的更新尺度。

### 小规模实验结论

（原文此处无额外展开。）

## B2 lr / eta / epsilon 调优之后训练表现

> ⚠️ 原文此处是一个无法在飞书文档外展示的嵌入内容（图表/表格），
> 导出时只得到占位文字：「暂时无法在飞书文档外展示此内容」。

- **Intentional global KL 最好**（甚至比 OrScale 好）；
- 按层调步长效果反而更差，目前机制不明确；
- **Activation intentional 基本滚蛋**，甚至存在梯度上升现象。

## B3 参数转移效果

- 大部分组别在 B2 调的参数转移到别的模型架构上之后仍为最优参数
  （猜测由于 lr / eta 的精细度较低）。
- 目前缺失 intentional kl global 的最优参数数据，但转移后的参数 loss 仍为全局最优。
- intentional kl layer 和 intentional activation 出现了训练发散的现象。

# 当前版：纯空间梯度通道选择

只修改此副本的 ECABlock，训练损失、数据处理和模型输出接口保持原样。
attention_gate 的编码器/解码器 ECA 和 RefinementBlock 的 ECA 都使用这个改动。

## 计算流程

1. 移除平均响应分支和可学习的通道 Conv1d，直接使用原始特征计算梯度。
2. 不再除以特征自身的平均幅值，以保留原始梯度大小的通道排序。
3. 固定 Sobel 算子计算 Dx、Dy，使用复制填充，避免常量图边缘产生假边界。
4. 对 |Dx| + |Dy| 求空间均值，获得每个通道的梯度分数。
5. logits = gradient_score，不经过可学习的通道卷积。
6. 沿用原可学习温度 sigmoid、通道均值归一化和 x * att。

均值等于梯度总和除以 H*W。同一层内通道尺寸相同，排序一致；
均值避免特征图尺寸改变分数量级。固定温度下，梯度越大，权重越大，
但 sigmoid 饱和时差异会变小。常量图的通道权重近似 1。

这里的梯度是特征图的空间变化，不是损失对特征的反向梯度。
固定 Sobel 核不更新；温度由优化器更新。梯度计算不使用 detach，
因此学习信号也可以传回输入特征及上游网络参数。

上一版 avg_logits + grad_logits 的两个加数系数都是 1，但不是固定 50%/50%：
梯度分支零初始化且可学习，贡献随训练变化。当前版同时取消幅值归一化和
grad_conv，以直接实现梯度大小决定通道权重；结果差异不能仅归因于取消平均分支。
本版不使用额外边界概率图 B。

## 对照实验

cfgs.py 中 eca_use_spatial_gradient = True 使用纯梯度版；False 使用原版 ECA。
分别从头训练，保持数据划分、随机种子、训练轮数等一致，分别保存实验结果。
比较 Dice、IoU、HD95 和边界可视化，不能把结构检查通过当成精度提升的证据。
当前结构与上一版不同，旧 checkpoint 不能用于本版断点恢复。
切换模型时应使用不同的 output_root，从头开始训练。

## 检查与运行

在 medcagd 环境中，进入此副本目录：

```powershell
cd D:\paper_read_practice\MedCAGD\code_try\MedCAGD-main
python verify_gradient_eca.py
```

检查脚本使用随机初始化编码器，验证模块和完整网络前向/反向，不下载权重，
不启动训练，不修改 output。

正式训练入口仍是 python xrun.py。先检查 cfgs.py 的数据路径和训练配置。
目前 auto_epoch_resume = True，tries = 1，使用独立 output_gradient_only 目录，
自动恢复最后完整 epoch，优先于旧 fresh_training/resume_train 逻辑，不清空已有结果。
恢复步骤见 RESUME_GUIDE.md。不要在原代码目录运行本次实验。

上一版 output_resumable 保留。batch_size=12、num_epochs=100；验证和测试指标
继续自动写入项目根目录运行结果.xlsx，模型名称由用户手填。
梯度也可能反映纹理和噪声，是否改善边界需要实际实验判断。

# 断电后按完整 epoch 恢复训练

已在此副本启用 auto_epoch_resume = True，仅支持当前的 tries = 1。
不改变模型结构。旧 fresh_training 设置在此自动恢复流程中不触发清空 output。
保留 resume_train = 0、resumed_chkpt = ""，无需每天修改这些变量。

## 使用步骤

1. 激活 medcagd 环境，进入副本目录。
2. 第一次检查 cfgs.py 中的数据路径、batch_size、num_epochs 等。
3. 运行 xrun.py。第二天恢复供电后运行相同命令即可。

```powershell
conda activate medcagd
cd D:\paper_read_practice\MedCAGD\code_try\MedCAGD-main
python xrun.py
```

PyCharm 中也可以运行此目录下的 xrun.py；解释器用 medcagd，工作目录设为此目录。
不要同时启动两份训练进程写同一个输出目录。

例如 num_epochs = 100：第 36 个 epoch 保存完成，第 37 个 epoch 中途断电。
再次运行时恢复第 36 个 epoch 的状态，从第 37 个重新开始，最终停在第 100 个。
epoch 内已跑过的 batch 不保留。即使第一个 epoch 就中断，也能从 epoch 0 的初始状态恢复。
训练已完成而验证/测试还未完成时，入口继续剩余评估阶段，不重新训练。
全部完成后再次运行会提示实验已完成。

## 保存了什么

- output_resumable/resume/latest.pt：模型（包括 buffers）、AdamW 状态、学习率调度器、
  AMP GradScaler（启用时）、Python/NumPy/PyTorch CPU/CUDA 随机状态、DataLoader generator、
  已完成 epoch、固定总 epoch 数、训练日志和配置/环境/训练文件标识。
- output_resumable/resume/latest.pt.previous：前一个完整恢复点，供最新文件损坏时回退。
- output_resumable/resume/run_state.json：训练、验证、最佳权重选择、测试和汇总的阶段状态。
- output_resumable/checkpoints/*.pth：已完成 epoch 的评估权重。
- output_resumable/train-logs/train-loss.csv：以已提交恢复点为准恢复日志，避免重复 epoch。
- output_resumable/summary.csv：全部流程结束后的最终汇总。

保存先写临时文件并 flush/fsync，最后用 os.replace 提交。中途失败不会直接截断旧恢复文件。
权重文件先写、latest.pt 后提交；若在这两步之间中断，未提交的权重不会进入验证候选。
原 output 目录保留，自动恢复模式不执行清空目录或剪除 checkpoint。

## 复现与限制

数据顺序使用显式 DataLoader generator。每次重新创建 worker 时从 PyTorch worker seed
同步设置 Python、NumPy、PyTorch 及 Albumentations 的独立随机状态。
完整 epoch 边界恢复后，重跑的 epoch 会使用相同的随机起点。

恢复期间保持模型代码、训练配置、数据和软件/硬件环境不变；变化会拒绝恢复，避免静默混用实验。
如需新实验，修改 cfgs.py 的 output_root 为新的目录名，不必删除旧结果。
num_epochs 表示固定总训练轮数，不表示“恢复后再跑多少轮”。
旧 checkpoint 没有这些完整状态，不能追溯补出随机状态；本功能适用于修改后新产生的恢复点。

已尽量启用确定性算法并关闭 TF32，但当前 CUDA 模型存在不支持确定性实现的运算，
所以 GPU 训练不能承诺与连续运行逐位一致；可能出现浮点误差并随训练放大。
软件恢复也无法保证磁盘损坏、硬件写缓存丢失时完全无损，前一个完整恢复点用于兜底。

## 验证

```powershell
python verify_epoch_resume.py
```

使用合成图片、实际 DriveDataset/双 worker 数据增强、包含 Dropout 和 BatchNorm 的小模型，
对比连续训练与第一/第二个 epoch 内中断再恢复的 CPU 模型参数、优化器、学习率、随机状态和损失。
同时检查保存失败、最新文件损坏回退、训练完成后恢复验证和完整实验不重复启动。
测试产物保存在 .verification 中，不改动实验输出，也不下载数据或预训练权重。

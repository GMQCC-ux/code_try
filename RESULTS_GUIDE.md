# 自动记录最终实验结果

只在 code_try 副本启用。运行 xrun.py，完整测试结束后自动写入项目根目录的运行结果.xlsx。
模型列由用户手动填写；程序从不写入或覆盖此列，也不改其他手工增加的列。

保留原有 数据集、模型、轮次、DICE、Iou 列；DICE 和 Iou 表示测试集结果。
新增测试 HD95/Precision/Recall，验证 Dice/IoU/HD95/Precision/Recall，
以及 Batch size、随机种子、Checkpoint、实验标识、完成时间。
轮次表示设置的总 epochs；百分比保存为 0~1 数字，显示为百分比。
验证和测试指标来自同一个验证最优 checkpoint 的最终 CSV，不能混用不同权重。

每个实验以 checkpoint 的完整位置和文件名识别。同一实验重复运行只更新原行，
不会重复追加；新实验追加新行，模型列留空。已有的手工记录保留。
本次已补齐原有 89.09% 那一行的验证/测试指标，未虚构原模型未提供的指标。

断电恢复模式完成后重新运行 xrun.py，会重新同步 Excel，不重跑训练。
实验期间可以打开 Excel 查看；最终写入时请关闭运行结果.xlsx。
如果 Excel 占用文件导致写入失败，CSV 和训练结果仍保留，关闭表格后重新运行即可。
恢复阶段会保留在 summary 或 complete，不会因此重新训练。

单独补写已有结果（不启动训练）：

```powershell
conda activate medcagd
cd D:\paper_read_practice\MedCAGD\code_try\MedCAGD-main
python experiment_results.py .
```

Excel 导出使用已安装的 Codex 内置 Node/表格运行库，不额外安装 Python 包。
临时请求和导出失败文件保存在 results_support，原表的首次备份保存在
results_support/运行结果.before-auto-write.bak。
写入采用进程锁和临时文件替换，避免同一实验重复记录以及半写入损坏。

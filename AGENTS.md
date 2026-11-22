## 正式训练实验

每次正式实验启动前，必须取得负责人一次明确确认。先列出实验名称、目的、拟上传的文件、关键配置和实际启动命令，请负责人确认本次实验可以归档并启动。确认后再推送代码、创建 SwanLab run 和提交训练作业。学习率扫描、消融和换 seed 复跑纳入正式实验时，同样适用。

只上传本次正式实验实际使用的训练代码、启动脚本和完整配置。调试脚本、临时试跑产物和草稿配置留在本地。密钥、数据集、checkpoint 和大体积日志也留在原处。

- **GitHub** — 推送到 [Liyikang626/llm_project](https://github.com/Liyikang626/llm_project)。确认远端可以访问对应 commit 后，使用该 commit 的代码和配置启动训练。
- **SwanLab** — 在 [liyikang24/llm_from_scratch](https://swanlab.cn/@liyikang24/llm_from_scratch/v1/zl774g/runs) 为每次正式实验创建唯一的 run。记录 Git commit SHA、配置和启动脚本链接、实际启动命令，以及模型架构与尺寸、初始权重、数据集版本与 split、数据混合比例、tokenizer 与词表大小、seed、序列长度、全局 batch size、训练预算、优化器、学习率、调度器、精度和 GPU。提交作业后补上 Slurm Job ID。
- **曲线与结果** — 在同一个 run 中记录训练 loss、验证 loss、学习率、梯度范数、吞吐和累计训练 tokens，使用训练 step 或累计 tokens 对齐曲线。结束后记录实际步数、tokens、耗时、最终和最佳验证 loss，以及结束状态。发散、失败或中止时保留已有曲线，并注明发生步数和原因。补传日志时沿用原 run。
- **交付** — 每次实验结束后给出简短结果摘要，附上 GitHub commit 和 SwanLab run 链接。

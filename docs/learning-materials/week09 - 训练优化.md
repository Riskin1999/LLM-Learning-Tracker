# 第9周 训练优化技巧
| 每日任务 | 学习素材（可直接访问） | 学习时长 | 备注 |
|----------|--------------|----------|------|
| Day1：梯度裁剪原理 | 1. 梯度裁剪教程：https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html <br> 2. 梯度裁剪代码：https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py | 2h | 设置max_norm=1.0解决梯度爆炸 |
| Day2：梯度累积实现 | 1. 梯度累积原理：https://huggingface.co/docs/transformers/training#gradient-accumulation <br> 2. 梯度累积代码：https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py | 2h | 用梯度累积模拟大批次训练 |
| Day3：Dropout在Transformer中的应用 | 1. Dropout论文：https://arxiv.org/abs/1207.0580 <br> 2. Transformer Dropout代码：https://github.com/karpathy/minGPT/blob/master/mingpt/model.py | 2h | 测试不同位置Dropout效果 |
| Day4：文本数据增强 | 1. 数据增强方法：https://zhuanlan.zhihu.com/p/547439458 <br> 2. 同义词替换代码：https://github.com/zhanlaoban/EnhancedBackTranslation | 2h | 人工筛选高质量增强样本 |
| Day5：学习率调度策略 | 1. 学习率调度教程：https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate <br> 2. 余弦退火代码：https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py | 2h | 测试不同调度策略收敛速度 |
| Day6：训练调试指南 | 1. NaN损失排查：https://huggingface.co/docs/transformers/troubleshooting#nan-loss <br> 2. 梯度消失解决：https://zhuanlan.zhihu.com/p/25081671 | 2h | 检查数据异常值+梯度裁剪 |
| Day7：第9周复盘 | 1. 训练优化组合方案：https://huggingface.co/docs/transformers/training <br> 2. 笔记工具Notion：https://www.notion.so/ | 2h | 整理「梯度累积+混合精度+余弦退火」方案 |

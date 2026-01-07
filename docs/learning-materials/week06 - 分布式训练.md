# 第6周 分布式训练技术
| 每日任务 | 学习素材（可直接访问） | 学习时长 | 备注 |
|----------|--------------|----------|------|
| Day1：显存计算原理 | 1. 模型显存计算公式：https://zhuanlan.zhihu.com/p/547439458 <br> 2. 显存计算代码：https://github.com/facebookresearch/fairseq/blob/main/fairseq/utils.py | 2h | 考虑梯度/优化器状态额外占用 |
| Day2：DataParallel实现 | 1. PyTorch DP教程：https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html <br> 2. DP代码示例：https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html | 2h | 用单模型多卡测试数据并行 |
| Day3：ModelParallel实现 | 1. PyTorch MP教程：https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html <br> 2. Transformer层拆分代码：https://github.com/pytorch/examples/blob/main/distributed/ddp/example.py | 2h | 按层拆分到不同GPU |
| Day4：DeepSpeed入门 | 1. DeepSpeed官方文档：https://www.deepspeed.ai/docs/config-json/ <br> 2. DeepSpeed配置示例：https://github.com/microsoft/DeepSpeedExamples/tree/master/training/ZeroExample | 2h | 编写基础配置文件 |
| Day5：ZeRO优化器原理 | 1. ZeRO论文：https://arxiv.org/abs/1910.02054 <br> 2. ZeRO-1/2/3对比：https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training | 2h | 测试不同ZeRO级别显存占用 |
| Day6：混合精度训练 | 1. PyTorch AMP教程：https://pytorch.org/docs/stable/amp.html <br> 2. AMP代码示例：https://pytorch.org/docs/stable/notes/amp_examples.html | 2h | 用AMP自动混合精度训练 |
| Day7：第6周复盘 | 1. 分布式训练常见报错：https://www.deepspeed.ai/docs/troubleshooting/ <br> 2. 笔记工具Notion：https://www.notion.so/ | 2h | 整理报错原因与解决方法 |

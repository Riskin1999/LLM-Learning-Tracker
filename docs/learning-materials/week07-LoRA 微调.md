# 第7周 LoRA微调技术
| 每日任务 | 学习素材（可直接访问） | 学习时长 | 备注 |
|----------|--------------|----------|------|
| Day1：全参数微调原理 | 1. 全参数微调教程：https://huggingface.co/docs/transformers/training <br> 2. 全参数微调代码：https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py | 2h | 减小批次大小+梯度累积解决显存不足 |
| Day2：LoRA原理与秩选择 | 1. LoRA论文：https://arxiv.org/abs/2106.09685 <br> 2. LoRA秩选择指南：https://huggingface.co/docs/peft/conceptual_guides/lora | 2h | 测试r=8/16/32的微调效果 |
| Day3：PEFT库使用 | 1. PEFT官方文档：https://huggingface.co/docs/peft/ <br> 2. PEFT LoRA示例：https://github.com/huggingface/peft/tree/main/examples | 2h | 升级库版本解决兼容性问题 |
| Day4：Llama-2 LoRA微调 | 1. Llama-2 LoRA代码：https://github.com/huggingface/peft/tree/main/examples/int8_training <br> 2. trust_remote_code参数说明：https://huggingface.co/docs/transformers/main/en/model_doc/llama2 | 2h | 解决模型加载报错 |
| Day5：TensorBoard监控 | 1. TensorBoard使用教程：https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html <br> 2. 损失曲线记录代码：https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py | 2h | 分析学习率/批次大小影响 |
| Day6：过拟合解决方法 | 1. 过拟合解决指南：https://huggingface.co/docs/transformers/training#avoiding-overfitting <br> 2. 早停实现代码：https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py | 2h | 监控验证集损失设置阈值 |
| Day7：第7周复盘 | 1. LoRA微调实验总结：https://huggingface.co/docs/peft/conceptual_guides/lora <br> 2. 人工评估指标设计：https://arxiv.org/abs/2306.05685 | 2h | 从流畅度/相关性评估效果 |

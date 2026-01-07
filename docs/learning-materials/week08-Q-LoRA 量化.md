# 第8周 Q-LoRA量化微调
| 每日任务 | 学习素材（可直接访问） | 学习时长 | 备注 |
|----------|--------------|----------|------|
| Day1：Q-LoRA原理 | 1. Q-LoRA论文：https://arxiv.org/abs/2305.14314 <br> 2. 4bit/8bit量化原理：https://huggingface.co/blog/4bit-transformers-bitsandbytes | 2h | 学习量化感知训练方法 |
| Day2：bitsandbytes安装 | 1. bitsandbytes官方文档：https://github.com/TimDettmers/bitsandbytes/blob/main/README.md <br> 2. CUDA版本对应表：https://github.com/TimDettmers/bitsandbytes#requirements--installation | 2h | 按官方指南安装对应版本 |
| Day3：4bit量化Llama-2微调 | 1. Q-LoRA微调代码：https://github.com/artidoro/qlora <br> 2. 4bit量化配置：https://github.com/artidoro/qlora/blob/main/qlora.py | 2h | 检查量化与PEFT兼容性 |
| Day4：LoRA vs Q-LoRA对比 | 1. 显存占用对比实验：https://github.com/artidoro/qlora/blob/main/scripts/benchmark.sh <br> 2. 效果对比报告：https://arxiv.org/abs/2305.14314 | 2h | 控制变量测试相同数据/超参数 |
| Day5：GPTQ量化 | 1. GPTQ论文：https://arxiv.org/abs/2210.17323 <br> 2. GPTQ量化代码：https://github.com/oobabooga/GPTQ-for-LLaMa | 2h | 用多线程加速量化过程 |
| Day6：量化推理速度对比 | 1. 推理速度测试代码：https://github.com/artidoro/qlora/blob/main/scripts/benchmark.sh <br> 2. 优化推理速度：https://huggingface.co/docs/transformers/performance | 2h | 关闭梯度计算提升速度 |
| Day7：第8周复盘 | 1. 量化方法对比表：https://huggingface.co/blog/4bit-transformers-bitsandbytes <br> 2. 笔记工具Notion：https://www.notion.so/ | 2h | 整理不同量化方法适用场景 |

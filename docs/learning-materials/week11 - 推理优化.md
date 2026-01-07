# 第11周 推理优化技术
| 每日任务 | 学习素材（可直接访问） | 学习时长 | 备注 |
|----------|--------------|----------|------|
| Day1：生成策略（贪婪/束搜索） | 1. 李沐生成策略视频：https://www.bilibili.com/video/BV1Qv411q73c/?p=5 <br> 2. 束搜索代码：https://github.com/karpathy/minGPT/blob/master/mingpt/utils.py#L57 | 2h | 测试beam size=2/4/8的效果 |
| Day2：采样策略（Top-K/Top-P） | 1. Hugging Face生成策略文档：https://huggingface.co/docs/transformers/generation_strategies <br> 2. Top-K采样代码：https://huggingface.co/docs/transformers/generation_strategies#top-k-sampling | 2h | 调整temperature系数 |
| Day3：vLLM原理 | 1. vLLM论文：https://arxiv.org/abs/2309.06180 <br> 2. PagedAttention原理：https://vllm.readthedocs.io/en/latest/concepts/paged_attention.html | 2h | 理解页式注意力核心 |
| Day4：vLLM部署 | 1. vLLM快速入门：https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html <br> 2. Llama-2 vLLM部署代码：https://github.com/vllm-project/vllm/blob/main/examples/llama_example.py | 2h | 参考官方示例快速上手 |
| Day5：TensorRT-LLM入门 | 1. TensorRT-LLM文档：https://docs.nvidia.com/deeplearning/tensorrt/llm-user-guide/index.html <br> 2. Llama-2 TRT-LLM部署：https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama | 2h | 适合高性能推理场景 |
| Day6：推理速度对比 | 1. 推理速度测试代码：https://github.com/vllm-project/vllm/blob/main/examples/benchmark.py <br> 2. 原生推理vs vLLM vs TRT-LLM对比：https://vllm.readthedocs.io/en/latest/performance.html | 2h | 同一GPU+相同输入长度测试 |
| Day7：第11周复盘 | 1. 推理框架对比表：https://www.tablesgenerator.com/markdown_tables <br> 2. 笔记工具Notion：https://www.notion.so/ | 2h | 整理不同框架适用场景 |

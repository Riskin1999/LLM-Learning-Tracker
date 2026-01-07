# 第2周 Transformer核心架构
| 每日任务 | 学习素材（可直接访问） | 学习时长 | 备注 |
|----------|--------------|----------|------|
| Day1：前馈网络+激活函数对比 | 1. 李沐《动手学深度学习》前馈网络：https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp.html <br> 2. ReLU/LeakyReLU对比代码：https://github.com/ShusenTang/Dive-into-DL-PyTorch/blob/master/chapter03_dl-basics/activation.ipynb | 2h | 测试死亡神经元问题 |
| Day2：反向传播算法推导 | 1. 吴恩达反向传播视频：https://www.bilibili.com/video/BV164411b7dx/?p=26 <br> 2. 多层网络梯度计算：https://zhuanlan.zhihu.com/p/25081671 | 2h | 重点掌握链式求导 |
| Day3：Transformer论文精读 | 1. 原论文《Attention Is All You Need》：https://arxiv.org/abs/1706.03762 <br> 2. 李宏毅Transformer详解：https://www.bilibili.com/video/BV1rb411g7nD/ | 2h | 手绘架构图标注每一层 |
| Day4：多头注意力机制实现 | 1. Hugging Face注意力文档：https://huggingface.co/docs/transformers/main/en/glossary#attention <br> 2. 多头注意力PyTorch代码：https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L317 | 2h | 核对Q/K/V维度变换 |
| Day5：位置编码原理+实现 | 1. 李沐位置编码专题：https://zh.d2l.ai/chapter_attention-mechanism/self-attention-and-position-encoding.html <br> 2. 正弦位置编码代码：https://github.com/ShusenTang/Dive-into-DL-PyTorch/blob/master/chapter12_attention-mechanism/self-attention.ipynb | 2h | 对比有无位置编码的效果 |
| Day6：Encoder模块完整实现 | 1. Transformer Encoder代码：https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L39 | 2h | 按「自注意力→Add&Norm→FFN」组装 |
| Day7：Transformer复盘 | 1. 数据流图工具：https://draw.io/ <br> 2. Transformer面试题：https://zhuanlan.zhihu.com/p/441213780 | 2h | 理解注意力的本质 |

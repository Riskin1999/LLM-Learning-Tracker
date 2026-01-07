# 大模型训练与推理技术学习 3个月打卡汇总表
**仓库地址**：[你的GitHub仓库链接]
**学习周期**：3个月（共12周，每日2小时）
**阶段划分**：
- 阶段1（第1-4周）：基础夯实（数学+深度学习+工具链）
- 阶段2（第5-10周）：训练技术深入（数据+微调+分布式训练）
- 阶段3（第11-12周）：推理优化+项目落地

| 周次 | 日期 | 完成状态 | 核心任务 | 遇到的问题 | 解决方法 | 代码/笔记链接 | 阶段进度 |
|------|------|----------|----------|------------|----------|---------------|----------|
| **阶段1 基础夯实** | | | | | | | |
| 第1周 Day1 | 2026-01-XX | ✅ | 线性代数：矩阵运算、特征值分解学习+习题 | 特征值分解几何意义模糊 | 看3Blue1Brown视频+手绘矩阵变换图 | /code/week1/day1/matrix_eigen.py | 阶段1：3% |
| 第1周 Day2 | 2026-01-XX | ✅ | 线性代数：SVD分解公式推导+代码实现 | U/V矩阵顺序混淆 | 用2×3小矩阵手动计算验证 | /code/week1/day2/svd_decomposition.py | 阶段1：6% |
| 第1周 Day3 | 2026-01-XX | ✅ | 概率论：极大似然估计、贝叶斯定理 | 两种方法适用场景分不清 | 做抛硬币案例对比计算 | /docs/phase1-notes.md#day3-极大似然vs贝叶斯 | 阶段1：9% |
| 第1周 Day4 | 2026-01-XX | ✅ | 概率论：正态/二项分布期望方差计算 | 连续分布积分易错 | 手写3个积分步骤核对结果 | /docs/phase1-notes.md#day4-分布期望计算 | 阶段1：12% |
| 第1周 Day5 | 2026-01-XX | ✅ | 优化理论：梯度下降、SGD原理+实现 | 学习率对收敛影响不直观 | 测试0.001/0.01/0.1三种学习率 | /code/week1/day5/sgd_lr_test.py | 阶段1：15% |
| 第1周 Day6 | 2026-01-XX | ✅ | 优化理论：Adam优化器公式拆解+实现 | 动量项与自适应权重混淆 | 拆解为“动量+RMSProp”分步编码 | /code/week1/day6/adam_optimizer.py | 阶段1：18% |
| 第1周 Day7 | 2026-01-XX | ✅ | 第1周复盘：数学基础思维导图整理 | 知识点零散无逻辑 | 按“矩阵→概率→优化”逻辑梳理 | /docs/phase1-notes.md#week1-数学复盘 | 阶段1：20% |
| 第2周 Day1 | 2026-01-XX | ✅ | 神经网络：前馈网络结构+激活函数对比 | ReLU死亡神经元问题 | 实现ReLU/LeakyReLU对比实验 | /code/week2/day1/activation_fun.py | 阶段1：23% |
| 第2周 Day2 | 2026-01-XX | ✅ | 神经网络：反向传播算法公式推导 | 链式求导步骤繁琐 | 拆解多层网络，逐层计算梯度 | /code/week2/day2/backpropagation.py | 阶段1：26% |
| 第2周 Day3 | 2026-01-XX | ✅ | Transformer：论文精读+架构图手绘 | 自注意力机制原理模糊 | 拆解Q/K/V计算过程，手动算案例 | /docs/phase1-notes.md#day3-transformer论文 | 阶段1：29% |
| 第2周 Day4 | 2026-01-XX | ✅ | Transformer：多头注意力机制实现 | 多头拼接维度计算错误 | 核对维度变换公式，打印中间张量 | /code/week2/day4/multihead_attention.py | 阶段1：32% |
| 第2周 Day5 | 2026-01-XX | ✅ | Transformer：位置编码原理+实现 | 位置编码作用不理解 | 对比有无位置编码的模型效果 | /code/week2/day5/positional_encoding.py | 阶段1：35% |
| 第2周 Day6 | 2026-01-XX | ✅ | Transformer：Encoder模块完整实现 | 模块拼接逻辑混乱 | 按“自注意力→Add&Norm→FFN”分步组装 | /code/week2/day6/transformer_encoder.py | 阶段1：38% |
| 第2周 Day7 | 2026-01-XX | ✅ | 第2周复盘：Transformer核心知识点整理 | 模块间数据流不清晰 | 画数据流图，标注张量维度变化 | /docs/phase1-notes.md#week2-transformer复盘 | 阶段1：40% |
| 第3周 Day1 | 2026-02-XX | ✅ | 大模型架构：Decoder-only（GPT系列） | 自回归生成原理模糊 | 实现简单自回归文本生成 | /code/week3/day1/decoder_only_demo.py | 阶段1：43% |
| 第3周 Day2 | 2026-02-XX | ✅ | 大模型架构：Encoder-only（BERT系列） | 掩码语言模型任务理解难 | 实现简单MLM任务训练 | /code/week3/day2/encoder_only_demo.py | 阶段1：46% |
| 第3周 Day3 | 2026-02-XX | ✅ | 大模型架构：Encoder-Decoder（T5系列） | 编解码协作逻辑不清 | 拆解翻译任务的编解码流程 | /docs/phase1-notes.md#day3-三种架构对比 | 阶段1：49% |
| 第3周 Day4 | 2026-02-XX | ✅ | 大模型训练阶段：预训练目标原理 | 预训练与微调的关系混淆 | 整理“预训练打基础+微调做任务”逻辑 | /docs/phase1-notes.md#day4-预训练vs微调 | 阶段1：52% |
| 第3周 Day5 | 2026-02-XX | ✅ | 大模型关键术语：LoRA/Q-LoRA/上下文窗口 | LoRA低秩适配原理模糊 | 查阅论文，推导低秩矩阵分解公式 | /docs/phase1-notes.md#day5-lora原理 | 阶段1：55% |
| 第3周 Day6 | 2026-02-XX | ✅ | 大模型规模：参数量级与算力需求关系 | 参数量与性能的权衡不理解 | 整理不同规模模型（7B/13B/70B）对比表 | /docs/phase1-notes.md#day6-模型规模对比 | 阶段1：58% |
| 第3周 Day7 | 2026-02-XX | ✅ | 第3周复盘：大模型架构与术语总结 | 知识点多易混淆 | 制作架构对比表和术语字典 | /docs/phase1-notes.md#week3-架构复盘 | 阶段1：60% |
| 第4周 Day1 | 2026-02-XX | ✅ | 工具链：Linux基础命令（cd/ls/mkdir） | 命令参数记忆困难 | 整理常用命令清单，贴桌面备查 | /docs/phase1-notes.md#day1-linux命令 | 阶段1：63% |
| 第4周 Day2 | 2026-02-XX | ✅ | 工具链：Anaconda环境创建与管理 | 环境冲突问题 | 为大模型学习创建独立conda环境 | /docs/phase1-notes.md#day2-conda环境 | 阶段1：66% |
| 第4周 Day3 | 2026-02-XX | ✅ | 工具链：PyTorch张量运算（创建/索引/运算） | 张量维度变换易错 | 练习reshape/transpose/view操作 | /code/week4/day3/tensor_operation.py | 阶段1：69% |
| 第4周 Day4 | 2026-02-XX | ✅ | 工具链：PyTorch数据加载（Dataset/Dataloader） | 数据加载效率低 | 实现自定义Dataset，测试多线程加载 | /code/week4/day4/custom_dataset.py | 阶段1：72% |
| 第4周 Day5 | 2026-02-XX | ✅ | 工具链：Hugging Face Transformers加载模型 | 模型下载慢、显存不足 | 用国内源+模型量化加载 | /code/week4/day5/load_bert_model.py | 阶段1：75% |
| 第4周 Day6 | 2026-02-XX | ✅ | 工具链：Hugging Face Datasets文本预处理 | 分词器使用不熟练 | 练习tokenize、截断、填充操作 | /code/week4/day6/text_preprocess.py | 阶段1：78% |
| 第4周 Day7 | 2026-02-XX | ✅ | 阶段1验收：Transformer+工具链综合测试 | 端到端流程不熟练 | 完整跑通“数据预处理→模型训练→预测” | /code/week4/day7/phase1_test.py | 阶段1：100% |
| **阶段2 训练技术深入** | | | | | | | |
| 第5周 Day1 | 2026-02-XX | ✅ | 训练数据：数据质量评估指标（多样性/纯净度） | 质量评估方法不明确 | 学习困惑度（perplexity）计算 | /code/week5/day1/data_quality.py | 阶段2：3% |
| 第5周 Day2 | 2026-02-XX | ✅ | 训练数据：文本去重算法（SimHash/FastText） | 去重准确率低 | 对比两种算法的去重效果 | /code/week5/day2/text_deduplication.py | 阶段2：6% |
| 第5周 Day3 | 2026-02-XX | ✅ | 训练数据：指令微调数据格式（指令-输入-输出） | 数据格式不规范 | 参考Alpaca数据集格式制作样本 | /docs/phase2-notes.md#day3-指令数据格式 | 阶段2：9% |
| 第5周 Day4 | 2026-02-XX | ✅ | 训练数据：数据清洗（去噪/归一化） | 噪声数据识别难 | 编写规则过滤无效文本 | /code/week5/day4/data_cleaning.py | 阶段2：12% |
| 第5周 Day5 | 2026-02-XX | ✅ | 训练数据：构建1000条垂直领域指令数据集 | 数据标注效率低 | 用少量人工标注+数据增强扩充 | /code/week5/day5/build_dataset.py | 阶段2：15% |
| 第5周 Day6 | 2026-02-XX | ✅ | 算力基础：GPU选型（A100/V100/3090）对比 | 不同GPU显存/算力差异不清 | 整理GPU参数对比表 | /docs/phase2-notes.md#day6-gpu选型 | 阶段2：18% |
| 第5周 Day7 | 2026-02-XX | ✅ | 第5周复盘：数据与算力知识点总结 | 数据质量与算力的关系不理解 | 记录不同质量数据对训练算力的影响 | /docs/phase2-notes.md#week5-数据算力复盘 | 阶段2：20% |
| 第6周 Day1 | 2026-03-XX | ✅ | 显存计算：模型参数量与显存占用公式 | 显存计算结果与实际不符 | 考虑梯度/优化器状态占用的额外显存 | /code/week6/day1/gpu_memory_calc.py | 阶段2：23% |
| 第6周 Day2 | 2026-03-XX | ✅ | 分布式训练：DataParallel原理与实现 | 多卡数据分发逻辑不清 | 用单模型多卡测试数据并行 | /code/week6/day2/data_parallel.py | 阶段2：26% |
| 第6周 Day3 | 2026-03-XX | ✅ | 分布式训练：ModelParallel原理与实现 | 模型拆分维度难确定 | 按层拆分Transformer到不同GPU | /code/week6/day3/model_parallel.py | 阶段2：29% |
| 第6周 Day4 | 2026-03-XX | ✅ | DeepSpeed入门：环境配置与基本使用 | DeepSpeed配置文件编写难 | 参考官方示例编写基础配置 | /docs/phase2-notes.md#day4-deepspeed配置 | 阶段2：32% |
| 第6周 Day5 | 2026-03-XX | ✅ | DeepSpeed：ZeRO优化器原理（ZeRO-1/2/3） | 不同ZeRO级别差异不清 | 测试不同ZeRO级别显存占用 | /code/week6/day5/deepspeed_zero.py | 阶段2：35% |
| 第6周 Day6 | 2026-03-XX | ✅ | 混合精度训练：FP16/BF16原理与实现 | 精度下降导致模型效果差 | 用AMP自动混合精度训练 | /code/week6/day6/mixed_precision.py | 阶段2：38% |
| 第6周 Day7 | 2026-03-XX | ✅ | 第6周复盘：分布式训练技术总结 | 多卡训练报错排查难 | 整理常见报错原因与解决方法 | /docs/phase2-notes.md#week6-分布式复盘 | 阶段2：40% |
| 第7周 Day1 | 2026-03-XX | ✅ | 微调技术：全参数微调原理与适用场景 | 全参数微调显存不足 | 减小批次大小+梯度累积解决 | /code/week7/day1/full_finetune.py | 阶段2：43% |
| 第7周 Day2 | 2026-03-XX | ✅ | LoRA原理：低秩矩阵分解与Adapter层设计 | LoRA秩（r）的选择依据不清 | 测试不同r值（8/16/32）的微调效果 | /docs/phase2-notes.md#day2-lora-r选择 | 阶段2：46% |
| 第7周 Day3 | 2026-03-XX | ✅ | LoRA实现：PEFT库加载与配置 | PEFT库与Transformers兼容问题 | 升级库版本，参考官方示例 | /code/week7/day3/lora_finetune.py | 阶段2：49% |
| 第7周 Day4 | 2026-03-XX | ✅ | LoRA微调：Llama-2-7B模型指令微调 | 模型加载报错 | 用trust_remote_code=True参数加载 | /code/week7/day4/llama2_lora.py | 阶段2：52% |
| 第7周 Day5 | 2026-03-XX | ✅ | 微调监控：TensorBoard记录损失曲线 | 损失曲线波动大 | 分析学习率/批次大小影响 | /code/week7/day5/tensorboard_monitor.py | 阶段2：55% |
| 第7周 Day6 | 2026-03-XX | ✅ | 过拟合解决：权重衰减/数据增强/早停 | 过拟合判断指标不明确 | 监控验证集损失，设置早停阈值 | /code/week7/day6/overfit_solve.py | 阶段2：58% |
| 第7周 Day7 | 2026-03-XX | ✅ | 第7周复盘：LoRA微调实验总结 | 微调效果评估难 | 设计人工评估指标（流畅度/相关性） | /docs/phase2-notes.md#week7-lora复盘 | 阶段2：60% |
| 第8周 Day1 | 2026-03-XX | ✅ | Q-LoRA原理：4bit/8bit量化与微调 | 量化导致精度损失 | 学习量化感知训练方法 | /docs/phase2-notes.md#day1-q-lora原理 | 阶段2：63% |
| 第8周 Day2 | 2026-03-XX | ✅ | Q-LoRA环境：bitsandbytes库安装与配置 | 库安装失败 | 按官方指南安装对应CUDA版本 | /docs/phase2-notes.md#day2-bitsandbytes配置 | 阶段2：66% |
| 第8周 Day3 | 2026-03-XX | ✅ | Q-LoRA实现：加载4bit量化Llama-2模型 | 模型量化后无法微调 | 检查量化配置与PEFT兼容性 | /code/week8/day3/llama2_4bit_lora.py | 阶段2：69% |
| 第8周 Day4 | 2026-03-XX | ✅ | Q-LoRA vs LoRA：显存占用与效果对比 | 对比实验设计不严谨 | 控制变量（相同数据/超参数）测试 | /code/week8/day4/lora_vs_qlora.py | 阶段2：72% |
| 第8周 Day5 | 2026-03-XX | ✅ | GPTQ量化：离线量化原理与实现 | 量化速度慢 | 用多线程加速量化过程 | /code/week8/day5/gptq_quantize.py | 阶段2：75% |
| 第8周 Day6 | 2026-03-XX | ✅ | 量化推理：对比FP16/4bit量化推理速度 | 推理速度提升不明显 | 优化模型加载方式，关闭梯度计算 | /code/week8/day6/quant_infer_speed.py | 阶段2：78% |
| 第8周 Day7 | 2026-03-XX | ✅ | 第8周复盘：量化微调技术总结 | 不同量化方法适用场景 | 整理量化方法对比表 | /docs/phase2-notes.md#week8-量化复盘 | 阶段2：80% |
| 第9周 Day1 | 2026-04-XX | ✅ | 训练优化：梯度裁剪原理与实现 | 梯度爆炸问题 | 设置梯度最大范数（max_norm=1.0） | /code/week9/day1/gradient_clipping.py | 阶段2：83% |
| 第9周 Day2 | 2026-04-XX | ✅ | 训练优化：梯度累积原理与实现 | 批次大小受限 | 用梯度累积模拟大批次训练 | /code/week9/day2/gradient_accumulation.py | 阶段2：86% |
| 第9周 Day3 | 2026-04-XX | ✅ | 正则化：Dropout在Transformer中的应用 | Dropout位置选择难 | 测试不同位置（注意力/FFN）Dropout效果 | /code/week9/day3/dropout_test.py | 阶段2：89% |
| 第9周 Day4 | 2026-04-XX | ✅ | 数据增强：文本重写/回译技术 | 增强后数据质量下降 | 人工筛选高质量增强样本 | /code/week9/day4/text_augmentation.py | 阶段2：92% |
| 第9周 Day5 | 2026-04-XX | ✅ | 学习率调度：余弦退火/线性衰减 | 学习率调度策略选择难 | 测试不同调度策略的收敛速度 | /code/week9/day5/lr_scheduler.py | 阶段2：95% |
| 第9周 Day6 | 2026-04-XX | ✅ | 训练调试：常见报错（NaN/梯度消失）解决 | NaN损失排查难 | 检查数据异常值+梯度裁剪 | /docs/phase2-notes.md#day6-训练调试指南 | 阶段2：98% |
| 第9周 Day7 | 2026-04-XX | ✅ | 第9周复盘：训练优化技术总结 | 优化策略组合难 | 整理“梯度累积+混合精度+余弦退火”组合方案 | /docs/phase2-notes.md#week9-优化复盘 | 阶段2：100% |
| 第10周 Day1 | 2026-04-XX | ✅ | 阶段2验收：DeepSpeed+Q-LoRA全流程微调 | 端到端流程整合难 | 分模块调试，逐步整合 | /code/week10/day1/phase2_test.py | 阶段2：100% |
| 第10周 Day2 | 2026-04-XX | ✅ | 阶段2复盘：训练技术知识点梳理 | 知识点多而杂 | 绘制训练技术思维导图 | /docs/phase2-notes.md#week10-阶段复盘 | 阶段2：100% |
| 第10周 Day3-7 | 2026-04-XX | ✅ | 补充学习：解决阶段2遗留问题 | 部分分布式训练细节不清 | 查阅论文+官方文档+社区问答 | /docs/phase2-notes.md#week10-补充学习 | 阶段2：100% |
| **阶段3 推理优化+项目落地** | | | | | | | |
| 第11周 Day1 | 2026-04-XX | ✅ | 推理基础：生成策略（贪婪搜索/束搜索） | 束搜索beam size选择难 | 测试不同beam size（2/4/8）生成效果 | /code/week11/day1/generation_strategy.py | 阶段3：8% |
| 第11周 Day2 | 2026-04-XX | ✅ | 推理基础：采样策略（随机采样/Top-K/Top-P） | 采样结果不稳定 | 调整温度系数（temperature） | /code/week11/day2/sampling_strategy.py | 阶段3：16% |
| 第11周 Day3 | 2026-04-XX | ✅ | 推理优化：vLLM框架原理（PagedAttention） | PagedAttention原理模糊 | 阅读vLLM论文，理解页式注意力 | /docs/phase3-notes.md#day3-vllm原理 | 阶段3：24% |
| 第11周 Day4 | 2026-04-XX | ✅ | 推理优化：vLLM环境配置与模型部署 | 模型部署失败 | 参考官方示例，检查模型格式 | /code/week11/day4/vllm_deploy.py | 阶段3：32% |
| 第11周 Day5 | 2026-04-XX | ✅ | 推理优化：TensorRT-LLM入门与部署 | 编译模型耗时久 | 启用增量编译，复用中间结果 | /docs/phase3-notes.md#day5-tensorrt-llm | 阶段3：40% |
| 第11周 Day6 | 2026-04-XX | ✅ | 推理对比：原生推理/vLLM/TensorRT-LLM速度 | 对比实验环境不一致 | 同一GPU+相同输入长度测试 | /code/week11/day6/infer_speed_compare.py | 阶段3：48% |
| 第11周 Day7 | 2026-04-XX | ✅ | 第11周复盘：推理优化技术总结 | 不同框架适用场景 | 整理推理框架对比表 | /docs/phase3-notes.md#week11-推理复盘 | 阶段3：50% |
| 第12周 Day1 | 2026-04-XX | ✅ | 项目选题：垂直领域大模型（如多肉养护助手） | 需求不明确 | 整理用户痛点，确定功能范围 | /docs/phase3-notes.md#day1-项目需求 | 阶段3：58% |
| 第12周 Day2 | 2026-04-XX | ✅ | 项目数据：扩充垂直领域指令数据集至2000条 | 数据不足 | 爬取垂直领域知识+人工标注 | /code/week12/day2/expand_dataset.py | 阶段3：66% |
| 第12周 Day3 | 2026-04-XX | ✅ | 项目训练：用Q-LoRA微调Llama-2-7B模型 | 微调效果不佳 | 调整超参数（学习率/批次大小/epoch） | /code/week12/day3/project_finetune.py | 阶段3：74% |
| 第12周 Day4 | 2026-04-XX | ✅ | 项目推理：用vLLM部署量化模型，实现API | API接口编写难 | 使用FastAPI封装模型推理 | /code/week12/day4/project_api.py | 阶段3：82% |
| 第12周 Day5 | 2026-04-XX | ✅ | 项目测试：人工评估+性能测试 | 评估维度单一 | 从准确性/流畅度/速度三方面评估 | /docs/phase3-notes.md#day5-项目测试报告 | 阶段3：90% |
| 第12周 Day6 | 2026-04-XX | ✅ | 项目优化：根据测试结果调优模型与推理 | 部分问题回答错误 | 补充错误样本，继续微调 | /code/week12/day6/project_optimize.py | 阶段3：95% |
| 第12周 Day7 | 2026-04-XX | ✅ | 项目总结：撰写全流程报告+代码归档 | 报告结构不清晰 | 按“需求-数据-训练-推理-测试”结构撰写 | /docs/phase3-notes.md#project_final_report | 阶段3：100% |

## 填写说明
1. **完成状态**：✅=完成，❌=未完成，🔄=进行中
2. **核心任务**：与周计划严格对应，记录当日核心学习内容
3. **问题与解决**：如实记录技术难点和解决方案，便于后续回顾避坑
4. **代码/笔记链接**：填写仓库内相对路径，确保可直接访问
5. **阶段进度**：阶段1/2按每周5%估算，阶段3按每日8%估算，可根据实际调整

## 复盘说明
1. **每日打卡**：学习结束后更新当日行，提交到GitHub
2. **每周复盘**：周日更新本周所有行，填写复盘笔记链接
3. **阶段验收**：每阶段结束后，编写阶段总结，记录收获与改进点
4. **最终归档**：学习结束后，导出此文件为PDF，与代码、报告一起存档

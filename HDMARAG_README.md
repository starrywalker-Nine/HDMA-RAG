# HDMARAG (Hierarchical Declarative Memory Augment RAG)

一个先进的检索增强生成系统，集成了分层记忆管理、step-back抽象思考和增强检索功能。

## 🚀 核心特性

### 1. 分层声明式记忆 (Hierarchical Declarative Memory)
- **情景记忆 (Episodic)**: 存储具体的事实和对话信息
- **语义记忆 (Semantic)**: 管理抽象概念和关系
- **程序记忆 (Procedural)**: 保存推理过程和方法
- **元记忆 (Meta)**: 记录关于记忆本身的信息

### 2. Step-back抽象思考 (Step-back Reasoning)
- **概念抽象**: 从具体问题中提取核心概念
- **原理提取**: 识别适用的一般性原理
- **模式识别**: 发现高层次的抽象模式
- **领域映射**: 建立跨领域的知识连接

### 3. 增强检索融合 (Enhanced Retrieval)
- **分层分块**: 多层级的上下文分割策略
- **对比增强**: 生成chunks与embedding chunks的对比分析
- **信息融合**: 智能整合多源信息
- **质量评估**: 动态评估和优化检索结果

## 📁 项目结构

```
HDMARAG/
├── hdmarag_core.py              # HDMARAG核心算法
├── hdmarag_system.py            # 完整系统集成
├── hdmarag_config.json          # 系统配置文件
├── run_hdmarag.py              # 主启动脚本
├── testQA.py                   # 对话式问答系统
├── testChunks.py               # 优化记忆块抽取器
├── testRetriever.py            # 多轮对话检索系统
├── answerModel.py              # 高级答案生成模型
├── local_rag_baselines.py      # 本地RAG基线方法
├── local_data_loader.py        # 本地数据加载器
├── local_longbench_evaluation_v2.py  # 评估框架
├── gpu_server_config.json      # GPU服务器配置
└── README.md                   # 项目说明
```

## 🛠️ 安装和配置

### 环境要求
- Python 3.9+
- PyTorch 2.0+
- transformers
- datasets
- openai
- numpy

### 安装依赖
```bash
pip install torch torchvision torchaudio
pip install transformers datasets openai numpy
pip install -r requirements.txt  # 如果有的话
```

### 配置API密钥
在 `hdmarag_config.json` 中设置您的API配置，或在运行时通过命令行参数指定。

## 🚀 快速开始

### 1. 交互式模式
```bash
python run_hdmarag.py --mode interactive
```

### 2. 快速测试
```bash
python run_hdmarag.py --mode quick --datasets multifieldqa_en hotpotqa --samples 5
```

### 3. 单个问题测试
```bash
python run_hdmarag.py --mode single \
  --question "What are the main challenges in AI development?" \
  --context "AI development faces several challenges including..."
```

### 4. 完整评估
```bash
python run_hdmarag.py --mode full --datasets all --samples 50
```

### 5. 基准测试
```bash
python run_hdmarag.py --mode benchmark
```

## 📊 支持的数据集

HDMARAG支持LongBench基准测试中的多个数据集：

### 问答任务
- **narrativeqa**: 叙事问答
- **qasper**: 科学论文问答  
- **multifieldqa_en**: 多领域英文问答
- **multifieldqa_zh**: 多领域中文问答

### 推理任务
- **hotpotqa**: 多跳推理问答
- **2wikimqa**: 维基百科多跳问答
- **musique**: 多步推理问答

### 摘要任务
- **gov_report**: 政府报告摘要
- **qmsum**: 查询导向摘要
- **multi_news**: 多文档新闻摘要
- **vcsum**: 视频字幕摘要

### 其他任务
- **trec**: 问题分类
- **lsht**: 长文本分类
- **passage_retrieval_en/zh**: 段落检索
- **lcc**: 代码补全
- **passage_count**: 计数任务

## 🧠 HDMARAG算法原理

### 1. 分层记忆管理
```python
# 记忆层次结构
hierarchical_memory = {
    "episodic": {},      # 具体事实
    "semantic": {},      # 抽象概念  
    "procedural": {},    # 推理过程
    "meta": {},          # 元认知
    "index": {           # 多维索引
        "concept_hierarchy": {},
        "temporal_index": {},
        "relevance_graph": {},
        "abstraction_levels": {}
    }
}
```

### 2. Step-back抽象流程
1. **概念提取**: 识别问题的核心概念
2. **原理映射**: 找到适用的一般性原理
3. **模式识别**: 发现抽象模式和结构
4. **维度提升**: 将思考提升到更高维度

### 3. 增强检索过程
1. **分层分块**: 生成多层级的文本chunks
2. **对比分析**: 比较生成chunks与传统embedding chunks
3. **信息融合**: 整合多源信息解决冲突
4. **质量增强**: 基于分层记忆优化最终结果

## 📈 性能指标

HDMARAG使用多维度性能评估：

### 核心指标
- **分层效率 (Hierarchical Efficiency)**: 分层记忆组织的有效性
- **增强分数 (Enhancement Score)**: 相对于基线方法的改进程度
- **抽象质量 (Abstraction Quality)**: Step-back推理的质量
- **融合效果 (Fusion Effectiveness)**: 信息融合的成功程度

### 系统指标
- **处理效率 (Processing Efficiency)**: 整体系统处理速度
- **记忆利用率 (Memory Utilization)**: 记忆资源的有效利用
- **答案质量 (Answer Quality)**: 生成答案的综合质量

## 🔧 配置选项

### 分层记忆配置
```json
{
  "memory_types": {
    "episodic": {"max_capacity": 100, "abstraction_level": 1},
    "semantic": {"max_capacity": 50, "abstraction_level": 4},
    "procedural": {"max_capacity": 30, "abstraction_level": 3},
    "meta": {"max_capacity": 20, "abstraction_level": 5}
  }
}
```

### Step-back推理配置
```json
{
  "abstraction_strategies": {
    "conceptual_abstraction": {"enabled": true, "depth_levels": 3},
    "principle_extraction": {"enabled": true, "generalization_threshold": 0.8},
    "pattern_recognition": {"enabled": true, "pattern_complexity": "adaptive"}
  }
}
```

### 增强检索配置
```json
{
  "chunk_generation": {
    "hierarchical_levels": [2000, 1000, 500, 250, 100],
    "overlap_ratio": 0.1,
    "boundary_detection": "sentence_aware"
  }
}
```

## 📝 使用示例

### Python API使用
```python
from hdmarag_system import HDMARAGSystem

# 初始化系统
hdmarag = HDMARAGSystem(api_key="your-api-key")

# 处理单个样本
sample = {
    "input": "What are the main challenges in AI?",
    "context": "AI development faces several challenges...",
    "answers": ["explainability, bias, efficiency"]
}

result = hdmarag.process_sample(sample, "multifieldqa_en")
print(f"答案: {result['final_answer']}")
print(f"置信度: {result['confidence']}")

# 评估多个数据集
datasets = ["multifieldqa_en", "hotpotqa"]
results = hdmarag.evaluate_multiple_datasets(datasets, max_samples_per_dataset=10)

# 保存结果
hdmarag.save_results(results)
```

### 命令行使用
```bash
# 处理自定义问题
python run_hdmarag.py --mode single \
  --question "Explain quantum computing" \
  --context "Quantum computing uses quantum mechanics..."

# 批量评估
python run_hdmarag.py --mode full \
  --datasets narrativeqa qasper hotpotqa \
  --samples 20 \
  --output-dir results/
```

## 🔍 结果分析

### 输出文件结构
```
hdmarag_results/
├── hdmarag_results_20240101_120000.json    # 完整结果
├── hdmarag_summary_20240101_120000.json    # 摘要报告
└── hdmarag_benchmark_results/               # 基准测试结果
```

### 结果解读
- **成功率**: 成功处理的样本比例
- **平均处理时间**: 每个样本的平均处理时间
- **增强效果**: HDMARAG相对于基线方法的改进
- **记忆效率**: 分层记忆系统的利用效率

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- LongBench基准测试数据集
- OpenAI API支持
- 相关研究工作的启发

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至项目维护者
- 参与项目讨论

---

**HDMARAG**: 让RAG系统具备人类般的分层记忆和抽象思考能力！
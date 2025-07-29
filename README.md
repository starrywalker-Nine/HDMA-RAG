# HDMARAG - Hierarchical Declarative Memory Augment RAG

一个先进的检索增强生成系统，专为超长上下文处理设计，支持本地vLLM部署和A100显卡优化。

## 🚀 核心特性

### 🧠 分层声明式记忆
- **情景记忆**: 存储具体事实和对话信息
- **语义记忆**: 管理抽象概念和关系
- **程序记忆**: 保存推理过程和方法
- **元记忆**: 记录元认知信息

### 🔄 Step-back抽象思考
- **概念抽象**: 从具体问题提取核心概念
- **原理提取**: 识别适用的一般性原理
- **模式识别**: 发现高层次抽象模式
- **维度提升**: 将思考提升到更高维度

### ⚡ 增强检索融合
- **分层分块**: 多层级上下文分割 [2000→1000→500→250→100]
- **对比增强**: 生成chunks vs embedding chunks智能对比
- **信息融合**: 基于分层记忆的冲突解决
- **质量优化**: 动态评估和结果增强

## 🛠️ 环境要求

### 硬件配置
- **GPU**: 1-4张 A100 40GB 显卡
- **内存**: 64GB+ 系统内存
- **存储**: 500GB+ SSD空间

### 软件环境
- **Python**: 3.12+
- **CUDA**: 12.1+
- **vLLM**: 0.3.0+

## 📦 快速安装

### 1. 克隆项目
```bash
git clone <repository-url>
cd HDMARAG
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 下载模型
```bash
./download_models.sh
```

### 4. 快速启动
```bash
./quick_start.sh
```

## 🎯 使用方法

### 交互式模式
```bash
python run_hdmarag.py --mode interactive
```

### 快速测试
```bash
python run_hdmarag.py --mode quick --model qwen2.5-7b-instruct --samples 5
```

### 批量评估
```bash
python run_hdmarag.py --mode full --model qwen2.5-14b-instruct --samples 50
```

### 单个问题
```bash
python run_hdmarag.py --mode single \
  --question "解释量子计算的基本原理" \
  --context "量子计算利用量子力学原理..."
```

## 🔧 模型配置

### 支持的模型
- **Qwen2.5-7B-Instruct**: 1张A100 (推荐入门)
- **Qwen2.5-14B-Instruct**: 2张A100 (推荐性能)
- **Qwen2.5-32B-Instruct**: 4张A100 (最佳效果)

### 自动GPU分配
系统会根据可用GPU自动选择最优模型配置：
```python
# 1张A100 -> Qwen2.5-7B
# 2张A100 -> Qwen2.5-14B  
# 4张A100 -> Qwen2.5-32B
```

## 📊 性能指标

### HDMARAG专用指标
- **分层效率**: 分层记忆组织效果
- **增强分数**: 相对基线的提升程度
- **抽象质量**: Step-back推理质量
- **融合效果**: 信息融合成功率

### 系统监控
```bash
# 查看GPU使用情况
nvidia-smi

# 运行系统测试
python test_hdmarag.py

# 查看性能报告
ls results/
```

## 🏗️ 项目结构

```
HDMARAG/
├── hdmarag_core.py              # 核心算法
├── hdmarag_system.py            # 系统集成
├── local_model_interface.py     # 本地模型接口
├── run_hdmarag.py              # 主启动脚本
├── test_hdmarag.py             # 系统测试
├── configs/                    # 配置文件
│   ├── hdmarag_config.json
│   └── local_model_config.json
├── models/                     # 本地模型
├── data/                       # 数据集
├── results/                    # 结果输出
└── logs/                       # 日志文件
```

## 🔬 算法原理

### 1. 分层记忆管理
```python
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
1. **概念提取** → 识别核心概念
2. **原理映射** → 找到一般性原理
3. **模式识别** → 发现抽象模式
4. **维度提升** → 提升思考层次

### 3. 增强检索过程
1. **分层分块** → 生成多层级chunks
2. **对比分析** → 比较不同检索结果
3. **信息融合** → 整合多源信息
4. **质量增强** → 优化最终输出

## 📈 性能优化

### vLLM优化配置
```json
{
  "max_model_len": 131072,
  "gpu_memory_utilization": 0.9,
  "tensor_parallel_size": "auto",
  "dtype": "bfloat16",
  "enable_prefix_caching": true,
  "enable_chunked_prefill": true
}
```

### 内存管理
- **自动垃圾回收**: 定期清理无用记忆
- **分层压缩**: 压缩低频访问记忆
- **缓存策略**: LRU增强缓存机制

## 🧪 测试和验证

### 运行测试套件
```bash
# 完整测试
python test_hdmarag.py

# 特定组件测试
python test_hdmarag.py core
python test_hdmarag.py system
python test_hdmarag.py integration
```

### 基准测试
```bash
python run_hdmarag.py --mode benchmark
```

## 📝 API使用

### Python API
```python
from hdmarag_system import HDMARAGSystem

# 初始化系统
hdmarag = HDMARAGSystem(
    model_name="qwen2.5-14b-instruct",
    use_local=True
)

# 处理样本
sample = {
    "input": "解释深度学习的工作原理",
    "context": "深度学习是机器学习的一个分支...",
    "answers": ["神经网络、反向传播、梯度下降"]
}

result = hdmarag.process_sample(sample, "single_doc_qa")
print(f"答案: {result['final_answer']}")
print(f"置信度: {result['confidence']}")
```

## 🔧 故障排除

### 常见问题

**Q: 模型加载失败**
```bash
# 检查模型路径
ls models/
# 检查GPU内存
nvidia-smi
# 重新下载模型
./download_models.sh
```

**Q: 内存不足**
```bash
# 使用更小的模型
python run_hdmarag.py --model qwen2.5-7b-instruct
# 调整GPU内存使用率
# 编辑 configs/local_model_config.json
```

**Q: vLLM安装问题**
```bash
# 重新安装vLLM
pip uninstall vllm
pip install vllm --no-cache-dir
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [Issues]
- 技术讨论: [Discussions]

---

**HDMARAG**: 让RAG系统具备人类般的分层记忆和抽象思考能力！🚀